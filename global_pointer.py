from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

INFINITY = 1e12

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, output_dim, merge_mode='add', custom_position_ids=False):
        super().__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape

        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.devices)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


def apply_rotary_position_embeddings(pos, qw, kw):
    cos_pos = pos[..., None, 1::2].repeat(1, 1, 1, 2)
    sin_pos = pos[..., None, ::2].repeat(1, 1, 1, 2)
    qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
    qw2 = torch.reshape(qw2, qw.shape)
    qw = qw * cos_pos + qw2 * sin_pos

    kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
    kw2 = torch.reshape(kw2, kw.shape)
    kw = kw * cos_pos + kw2 * sin_pos

    return qw, kw


def sequence_masking(x, mask, value, dim):
    if mask is None:
        return x

    assert dim > 0, 'dim must > 0'
    for _ in range(dim - 1):
        mask = torch.unsqueeze(mask, 1)
    for _ in range(x.ndim - mask.ndim):
        mask = torch.unsqueeze(mask, mask.ndim)
    return x * mask + value * (1 - mask)


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """

    def __init__(
        self,
        heads,
        head_size,
        hidden_size,
        RoPE=True,
    ):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.RoPE = RoPE
        # TODO: 2 的含义？
        self.dense = nn.Linear(hidden_size, heads * head_size * 2)

    def forward(self, inputs, mask=None):
        # 输入变换
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_size * 2, dim=-1)
        inputs = torch.stack(inputs, dim=-2)
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # 排除padding
        logits = sequence_masking(logits, mask, -INFINITY, 2)
        logits = sequence_masking(logits, mask, -INFINITY, 3)

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * INFINITY
        return logits / self.head_size ** 0.5


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * INFINITY
    y_pred_pos = y_pred - (1 - y_true) * INFINITY
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred):
    """

    :param y_true: [batch_size, num_classes, max_length, max_length]
    :param y_pred:
    :return:
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return 2 * (y_true * y_pred).sum() / (y_true + y_pred).sum()


class BertGPForTokenClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = GlobalPointer(config.num_labels, 64, config.hidden_size)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output, mask=attention_mask)

        loss = None
        if labels is not None:
            loss = global_pointer_crossentropy(labels, logits)
            # f1 = global_pointer_f1_score(labels, logits)
            # logger.info(f"Train f1: {f1}")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

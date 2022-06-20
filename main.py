#!/usr/bin/env python
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import rich
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from global_pointer import BertGPForTokenClassification, global_pointer_f1_score

logger = logging.getLogger(__name__)

tags = [
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene"
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    global_pointer_head: str = field(
        default="GlobalPointer",
        metadata={"help": "GlobalPointer or EfficientGlobalPointer"}
    )
    rope: bool = field(
        default=True,
        metadata={"help": "Whether or not to add SinusoidalPositionEmbedding"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./cluener_dataset.py", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert model_args.global_pointer_head in ["GlobalPointer", "EfficientGlobalPointer"]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
    )

    label_list = tags
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )
    config.global_pointer_head = model_args.global_pointer_head
    config.RoPE = model_args.rope

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    model = BertGPForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    logger.info(f"Model parameters: {model.num_parameters()}")

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        labels = []
        for i, label in enumerate(examples["span_tags"]):
            offset_mapping = tokenized_inputs["offset_mapping"][i]
            mapping = {}
            for token_idx, val in enumerate(offset_mapping):
                start = val[0].item()
                end = val[-1].item()
                if start == end == 0:  # [CLS]
                    continue
                for ci in range(start, end):
                    mapping[ci] = token_idx

            label_ids = np.zeros((num_labels, data_args.max_seq_length, data_args.max_seq_length))

            text = examples["text"][i]
            # rich.print("-" * 80)
            # rich.print(text)
            # rich.print(label)
            # tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[i])
            for tag_id, start, end in zip(label['tag'], label['start'], label['end']):
                if start in mapping and end in mapping:
                    token_start = mapping[start]
                    token_end = mapping[end]
                    # if "".join(tokens[token_start: token_end + 1]).replace("#", "") != text[start:end + 1].replace("#", ""):
                    #     rich.print(tokens[token_start: token_end + 1])
                    #     rich.print(text[start:end + 1])
                    #     import ipdb;
                    #     ipdb.set_trace()
                    label_ids[tag_id, token_start, token_end] = 1

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.with_transform(tokenize_and_align_labels)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.with_transform(tokenize_and_align_labels)

    def compute_metrics(p):
        predictions, labels = p
        f1 = global_pointer_f1_score(labels, predictions)
        return {"f1": f1}

    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

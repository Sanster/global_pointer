#!/usr/bin/env python
import logging
import json
from pathlib import Path

import datasets
import rich
import numpy as np
import torch
import typer
from datasets import load_dataset
from transformers import AutoTokenizer

from global_pointer import BertGPForTokenClassification

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

app = typer.Typer(add_completion=False)


@app.command()
def main(model_path: str, save_path: Path, device: str = 'cuda', dataset_name: str = './cluener_dataset.py'):
    device = torch.device(device)
    raw_datasets = load_dataset(dataset_name, split=datasets.Split.TEST)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = BertGPForTokenClassification.from_pretrained(model_path).to(device)

    entities = []
    for it in raw_datasets:
        text = it['text']
        tokenized_inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        offsets_mapping = tokenized_inputs.pop('offset_mapping')[0]
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized_inputs.items()}
        pred = model(**inputs)

        logits = pred.logits[0].detach().cpu().numpy()

        logits[:, [0, -1]] -= np.inf
        logits[:, :, [0, -1]] -= np.inf

        # {
        #     'text': '索尼《GT赛车》新作可能会发行PC版？',
        #     'label': {
        #         'game': {'《GT赛车》': [[2, 7]]},
        #         'company': {'索尼': [[0, 1]]}
        #     }
        # }
        labels = {}
        for tag_idx, token_start_index, token_end_index in zip(*np.where(logits > 0)):
            start = offsets_mapping[token_start_index][0].item()
            end = offsets_mapping[token_end_index][-1].item()
            tag = tags[tag_idx]
            entity = text[start:end]

            if tag not in labels:
                labels[tag] = {entity: [[start, end]]}
            else:
                if entity in labels[tag]:
                    labels[tag][entity].append([start, end])
                else:
                    labels[tag][entity] = [[start, end]]

        entities.append({
            "id": it['id'],
            "text": text,
            'label': labels
        })

        rich.print(labels)

    with open(save_path, 'w', encoding='utf-8') as f:
        for e in entities:
            f.write(f"{json.dumps(e, ensure_ascii=False)}\n")


if __name__ == "__main__":
    app()

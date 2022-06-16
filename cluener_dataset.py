# coding=utf-8
import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_URL = "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip"
_TRAINING_FILE = "train.json"
_DEV_FILE = "dev.json"
_TEST_FILE = "test.json"


class CluenerConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(CluenerConfig, self).__init__(**kwargs)


class Cluener(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        CluenerConfig(name="cluener", version=datasets.Version("1.0.0")),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "span_tags": datasets.Sequence(
                        {
                            "tag": datasets.features.ClassLabel(
                                names=[
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
                            ),
                            "start": datasets.Value("int16"),
                            "end": datasets.Value("int16"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        data_files = {
            "train": os.path.join(downloaded_file, _TRAINING_FILE),
            "dev": os.path.join(downloaded_file, _DEV_FILE),
            "test": os.path.join(downloaded_file, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for line in f:
                """
                {
                    'text': '索尼《GT赛车》新作可能会发行PC版？',
                    'label': {
                        'game': {'《GT赛车》': [[2, 7]]}, 
                        'company': {'索尼': [[0, 1]]}
                    }
                }
                """
                data = json.loads(line)
                span_tags = []

                if 'label' in data:
                    for tag, labels in data['label'].items():
                        for spans in labels.values():
                            for span in spans:
                                span_tags.append({
                                    "tag": tag,
                                    "start": span[0],
                                    "end": span[1],
                                })

                yield guid, {
                    "id": str(data.get("id", guid)),
                    "text": data['text'].lower(),
                    "span_tags": span_tags,
                }
                guid += 1

import os
from functools import partial

from datasets import load_dataset
from transformers import pipeline, BitsAndBytesConfig

from tools import predict

pipe = pipeline(
    task="zero-shot-classification",
    model="bigscience/bloomz-7b1",
    model_kwargs={
        "low_cpu_mem_usage": True,
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    },
)

root = ".."
datastore_dir = os.path.join(root, "datastore", "cola_datasets")

pipe_fn = partial(predict, pipe=pipe)

for lang in [
    "fr",
    "en",
    "it",
    "ru",
    "sv",
    "zh",
    "no",
    "ja",
]:
    with open(f"{lang}-boomz.txt", "w") as file:
        print(f"Language: {lang}", file=file)
        data_dir = os.path.join(datastore_dir, f"{lang}-cola")
        dev_dataset = load_dataset(data_dir, data_files=["dev.tsv"])
        test_dataset = load_dataset(data_dir, data_files=["test.tsv"])

        dev_dataset = dev_dataset.map(pipe_fn)
        test_dataset = test_dataset.map(pipe_fn)

        dev_accuracy = (
            sum(dev_dataset["train"]["good_prediction"])
            / len(dev_dataset["train"]["good_prediction"])
            * 100
        )
        test_accuracy = (
            sum(test_dataset["train"]["good_prediction"])
            / len(test_dataset["train"]["good_prediction"])
            * 100
        )

        print(f"Dev acc: {dev_accuracy}", file=file)
        print(f"Test acc: {test_accuracy}", file=file)

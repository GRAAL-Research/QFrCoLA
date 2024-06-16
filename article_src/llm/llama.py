import os
from functools import partial

from datasets import load_dataset
from dotenv import dotenv_values
from evaluate import load
from transformers import pipeline, BitsAndBytesConfig

from tools import predict

secrets = dotenv_values(".env")

token = secrets["huggingface_token"]

pipe = pipeline(
    task="zero-shot-classification",
    model="meta-llama/Llama-2-7b-hf",
    model_kwargs={
        "low_cpu_mem_usage": True,
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        "token": token,
    },
)

ACCURACY = load("accuracy")
MCC = load("matthews_correlation")

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
    with open(f"{lang}-llama.txt", "w") as file:
        print(f"Language: {lang}\n", file=file)
        data_dir = os.path.join(datastore_dir, f"{lang}-cola")
        dev_dataset = load_dataset(data_dir, data_files=["dev.tsv"])
        test_dataset = load_dataset(data_dir, data_files=["test.tsv"])

        dev_dataset = dev_dataset.map(pipe_fn)

        predictions = dev_dataset["train"]["prediction"]
        labels = dev_dataset["train"]["label"]
        dev_accuracy = ACCURACY.compute(predictions=predictions, references=labels)
        dev_mcc = MCC.compute(predictions=predictions, references=labels)

        print(f"Dev acc: {dev_accuracy}\n", file=file)
        print(f"Dev MCC: {dev_mcc}\n", file=file)

        test_dataset = test_dataset.map(pipe_fn)

        predictions = test_dataset["train"]["prediction"]
        labels = test_dataset["train"]["label"]
        test_accuracy = ACCURACY.compute(predictions=predictions, references=labels)
        test_mcc = MCC.compute(predictions=predictions, references=labels)

        print(f"Test acc: {test_accuracy}\n", file=file)
        print(f"Test MCC: {test_mcc}\n", file=file)

        if lang == "fr":
            ood_dataset = load_dataset(data_dir, data_files=["ood.tsv"])
            ood_dataset = ood_dataset.map(pipe_fn)

            predictions = ood_dataset["train"]["prediction"]
            labels = ood_dataset["train"]["label"]
            ood_accuracy = ACCURACY.compute(predictions=predictions, references=labels)
            ood_mcc = MCC.compute(predictions=predictions, references=labels)

            print(f"OOD acc: {ood_accuracy}\n", file=file)
            print(f"OOD MCC: {ood_mcc}\n", file=file)

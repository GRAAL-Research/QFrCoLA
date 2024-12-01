import os
from functools import partial

import numpy as np
from datasets import load_dataset
from dotenv import dotenv_values
from transformers import pipeline, BitsAndBytesConfig

from metrics_wrapper import accuracy, mcc
from tools import predict

batch_size = 1024

secrets = dotenv_values(".env")

token = secrets["huggingface_token"]

pipe = pipeline(
    task="zero-shot-classification",
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=token,
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
    data_dir = os.path.join(datastore_dir, f"{lang}-cola")
    dev_dataset = load_dataset(data_dir, data_files=["dev.tsv"])
    test_dataset = load_dataset(data_dir, data_files=["test.tsv"])

    dev_dataset = dev_dataset.map(
        pipe_fn, batched=True, batch_size=batch_size, desc="llama3.1-instruct-dev"
    )

    dev_predictions = dev_dataset["train"]["prediction"]
    dev_labels = dev_dataset["train"]["label"]
    dev_accuracy = accuracy(predictions=dev_predictions, references=dev_labels)
    dev_mcc = mcc(predictions=dev_predictions, references=dev_labels)

    test_dataset = test_dataset.map(
        pipe_fn, batched=True, batch_size=batch_size, desc="llama3.1-instruct-test"
    )

    test_predictions = test_dataset["train"]["prediction"]
    test_labels = test_dataset["train"]["label"]
    test_accuracy = accuracy(predictions=test_predictions, references=test_labels)
    test_mcc = mcc(predictions=test_predictions, references=test_labels)

    with open(os.path.join(".", "results", f"{lang}-llama-instruct.txt", "w")) as file:
        print(f"Dev acc: {dev_accuracy}\n", file=file)
        print(f"Dev MCC: {dev_mcc}\n", file=file)

        print(f"Test acc: {test_accuracy}\n", file=file)
        print(f"Test MCC: {test_mcc}\n", file=file)

        if lang == "fr":
            test_categories = test_dataset["train"]["category"]

            for category_name in ["syntax", "morphology", "semantic", "anglicism"]:
                test_cat_idxs = [
                    idx
                    for idx, category in enumerate(test_categories)
                    if category == category_name
                ]
                test_cat_labels = np.array(test_labels)[test_cat_idxs]
                test_cat_pred = np.array(test_predictions)[test_cat_idxs]

                test_accuracy_cat = accuracy(
                    predictions=test_cat_pred, references=test_cat_labels
                )
                test_mcc_cat = mcc(
                    predictions=test_cat_pred, references=test_cat_labels
                )

                print(
                    f"Test category {category_name} acc: {test_accuracy_cat}\n",
                    file=file,
                )
                print(f"Test category {category_name} MCC: {test_mcc_cat}\n", file=file)

            ood_dataset = load_dataset(data_dir, data_files=["ood.tsv"])
            ood_dataset = ood_dataset.map(
                pipe_fn,
                batched=True,
                batch_size=batch_size,
                desc="llama3.1-instruct-ood",
            )

            predictions = ood_dataset["train"]["prediction"]
            labels = ood_dataset["train"]["label"]
            ood_accuracy = accuracy(predictions=predictions, references=labels)
            ood_mcc = mcc(predictions=predictions, references=labels)

            print(f"OOD acc: {ood_accuracy}\n", file=file)
            print(f"OOD MCC: {ood_mcc}\n", file=file)

import os.path

import pandas as pd
from datasets import Dataset, DatasetDict

seed = 42


def validate_punctuation(string):
    if string[-1] not in [".", "!", "?", "»", "…"]:
        string += "."
    return string


root_dataset_path = os.path.join("datastore", "cola_datasets", "fr-cola-source")
bdl_data = pd.read_csv(
    os.path.join(root_dataset_path, "bdl.tsv"),
    sep="\t",
    encoding="utf8",
    header=1,
    names=["label", "sentence", "source"],
    dtype={"label": int, "sentence": str, "source": str},
)

bdl_data["sentence"] = bdl_data["sentence"].str.strip()
bdl_data["source"] = bdl_data["source"].str.strip()
bdl_data["sentence"] = bdl_data["sentence"].str.replace("\xa0", " ")
bdl_data["sentence"] = bdl_data["sentence"].str.replace("  ", " ")
bdl_data["sentence"].apply(validate_punctuation)

assert len(bdl_data[bdl_data.duplicated(subset="sentence")]) == 0

print("Unsplitted")
total_row = len(bdl_data["label"])
positive = bdl_data["label"].sum()
ratio = round(positive / total_row * 100, 2)
print(total_row)
print(positive)
print(ratio)

complete_dataset = Dataset.from_pandas(bdl_data, preserve_index=False)
complete_dataset = complete_dataset.class_encode_column("label")

# 60-10-30 train-test ratio
# 30% for test
trainvalid_test = complete_dataset.train_test_split(
    test_size=0.3, shuffle=True, seed=seed, stratify_by_column="label"
)
# Split the train in 60-10
train_valid = trainvalid_test["train"].train_test_split(
    test_size=0.1, shuffle=True, seed=seed, stratify_by_column="label"
)

train_test_valid_dataset = DatasetDict(
    {
        "train": train_valid["train"],
        "test": trainvalid_test["test"],
        "dev": train_valid["test"],
    }
)
print("Train")
total_row = len(train_test_valid_dataset["train"]["label"])
positive = sum(train_test_valid_dataset["train"]["label"])
ratio = round(positive / total_row * 100, 2)
print(total_row)
print(positive)
print(ratio)

print("Dev")
total_row = len(train_test_valid_dataset["dev"]["label"])
positive = sum(train_test_valid_dataset["dev"]["label"])
ratio = round(positive / total_row * 100, 2)
print(total_row)
print(positive)
print(ratio)

print("Test")
total_row = len(train_test_valid_dataset["test"]["label"])
positive = sum(train_test_valid_dataset["test"]["label"])
ratio = round(positive / total_row * 100, 2)
print(total_row)
print(positive)
print(ratio)

output_dataset_path = os.path.join("datastore", "cola_datasets", "fr-cola")
output_train_file = os.path.join(output_dataset_path, "train.tsv")
output_validation_file = os.path.join(output_dataset_path, "dev.tsv")
output_test_file = os.path.join(output_dataset_path, "test.tsv")

train_test_valid_dataset["train"].to_pandas().to_csv(
    output_train_file, index=False, sep="\t"
)
train_test_valid_dataset["dev"].to_pandas().to_csv(
    output_validation_file, index=False, sep="\t"
)
train_test_valid_dataset["test"].to_pandas().to_csv(
    output_test_file, index=False, sep="\t"
)

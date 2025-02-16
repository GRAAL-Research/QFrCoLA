import os

from datasets import concatenate_datasets, DatasetDict

from src.load_dataset_wrapper import load_dataset_tsv

seed = 42

train_file = os.path.join("data", "zh-cola", "original_splits", "train.tsv")
validation_file = os.path.join("data", "zh-cola", "original_splits", "dev.tsv")
test_file = os.path.join("data", "zh-cola", "original_splits", "test.tsv")

data_files = {
    "train": train_file,
    "dev": validation_file,
}

# Loading a dataset from local csv files
raw_datasets = load_dataset_tsv(data_files=data_files)

all_datasets = concatenate_datasets([raw_datasets["train"], raw_datasets["dev"]])

# 60-10-30 train-test ratio
# 30% for test
trainvalid_test = all_datasets.train_test_split(test_size=0.3, shuffle=True, seed=seed)
# Split the train in 60-10
train_valid = trainvalid_test["train"].train_test_split(
    test_size=0.1, shuffle=True, seed=seed
)
train_test_valid_dataset = DatasetDict(
    {
        "train": train_valid["train"],
        "test": trainvalid_test["test"],
        "dev": train_valid["test"],
    }
)

output_train_file = os.path.join("data", "zh-cola", "train.tsv")
output_validation_file = os.path.join("data", "zh-cola", "dev.tsv")
output_test_file = os.path.join("data", "zh-cola", "test.tsv")

train_test_valid_dataset["train"].to_pandas().to_csv(
    output_train_file, index=False, sep="\t"
)
train_test_valid_dataset["dev"].to_pandas().to_csv(
    output_validation_file, index=False, sep="\t"
)
train_test_valid_dataset["test"].to_pandas().to_csv(
    output_test_file, index=False, sep="\t"
)

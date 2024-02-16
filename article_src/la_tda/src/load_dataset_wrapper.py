import pandas as pd
from datasets import Dataset, DatasetDict


def load_dataset_tsv(data_files):
    raw_dataset = DatasetDict()
    for data_file_name, data_file_path in data_files.items():
        raw_dataset.update(
            {
                data_file_name: Dataset.from_pandas(
                    pd.read_csv(data_file_path, sep="\t"), split=data_file_name
                )
            }
        )

    return raw_dataset

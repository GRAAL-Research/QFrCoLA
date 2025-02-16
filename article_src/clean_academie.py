import os.path

import pandas as pd

seed = 42


def validate_punctuation(string):
    if string[-1] not in [".", "!", "?", "»", "…"]:
        string += "."
    return string


root_dataset_path = os.path.join("datastore", "cola_datasets", "fr-cola-source")
bdl_data = pd.read_csv(
    os.path.join(root_dataset_path, "academie.tsv"),
    sep="\t",
    encoding="utf8",
    header=1,
    names=["label", "sentence", "source", "category"],
    dtype={"label": int, "sentence": str, "source": str, "category": str},
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

bdl_data.to_csv(
    os.path.join("datastore", "cola_datasets", "fr-cola", "ood.tsv"),
    sep="\t",
    index=False,
    encoding="utf8",
)

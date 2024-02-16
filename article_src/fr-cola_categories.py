import os

from la_tda.src.load_dataset_wrapper import load_dataset_tsv


def add_category(row):
    source = row["source"]
    structure = source.split("/")[4:]
    if "banque" in structure[0]:
        structure = structure[1:]
    category = structure[0].replace("-", " ")

    if category in [
        "la redaction et la communication",
        "les noms propres",
        "la syntaxe",
        "la ponctuation",
        "la typographie",
        "les noms propres",
    ]:
        meta_cat = "syntax"
    elif category in [
        "lorthographe",
        "la grammaire",
        "les abreviations et les symboles",
        "la prononciation",
    ]:
        meta_cat = "morphology"
    elif category in ["le vocabulaire"]:
        meta_cat = "semantic"
    elif category in ["les emprunts a langlais"]:
        meta_cat = "anglicism"
    else:
        meta_cat = "N/A"

    return {"category": meta_cat}


lang = "fr"

root = "."
datastore_dir = os.path.join(root, "datastore", "cola_datasets")
data_dir = os.path.join(datastore_dir, f"{lang}-cola")
print(data_dir)
train_data_path = os.path.join(data_dir, "train.tsv")
dev_data_path = os.path.join(data_dir, "dev.tsv")
test_data_path = os.path.join(data_dir, "test.tsv")

data_files = {
    "train": train_data_path,
    "dev": dev_data_path,
    "test": test_data_path,
}

cola = load_dataset_tsv(data_files=data_files)

cola = cola.map(add_category, num_proc=2)

cola["train"].to_csv(train_data_path, sep="\t", index=False)
cola["dev"].to_csv(dev_data_path, sep="\t", index=False)
cola["test"].to_csv(test_data_path, sep="\t", index=False)

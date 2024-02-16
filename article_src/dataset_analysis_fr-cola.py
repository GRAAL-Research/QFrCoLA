import os
from collections import Counter

from la_tda.src.load_dataset_wrapper import load_dataset_tsv

categories_counter = Counter()
sub_categories_counter = Counter()

cat_dic_label = {"syntax": [], "morphology": [], "semantic": [], "anglicism": []}


def compute_stats(row):
    source = row["source"]
    label = row["label"]
    structure = source.split("/")[4:]
    if "banque" in structure[0]:
        structure = structure[1:]
    category = structure[0].replace("-", " ")
    sub_category = structure[1].replace("-", " ")
    categories_counter.update([category])

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
        print("a")
    cat_dic_label[meta_cat].append(label)

    sub_categories_counter.update([f"{category}/{sub_category}"])


all_data_stats = {}
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

lang_data_stats = {}
ratios = []

for split in ["train", "dev", "test"]:
    cola[split].map(compute_stats)

[(key, sum(value) / len(value)) for key, value in cat_dic_label.items()]

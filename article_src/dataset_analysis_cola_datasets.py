import json
import os
from statistics import mean

import spacy

from la_tda.src.load_dataset_wrapper import load_dataset_tsv


def tokenize(model, text):
    return [token for token in model.tokenizer(text)]


def compute_stats(row):
    text = row["sentence"]

    tokens = [
        token.text
        for token in tokenize(nlp, text)
        if not token.is_punct
        and not "\n" in token.text
        and not token.is_digit
        and not "|" in token.text
        and not "$" in token.text
        and not "<" in token.text
        and not ">" in token.text
        and not " " in token.text
    ]

    lexical_words = [
        token.text
        for token in tokenize(nlp, text)
        if not token.is_punct
        and not "\n" in token.text
        and not token.is_digit
        and not "|" in token.text
        and not "$" in token.text
        and not "<" in token.text
        and not ">" in token.text
        and not " " in token.text
        and not token.is_stop
    ]

    sentences_len.append(len(tokens))
    lexical_words_len.append(len(lexical_words))

    global_vocabulary_set.update(tokens)
    global_vocabulary_lexical_words_set.update(lexical_words)
    vocab.update(tokens)
    ratios.append(row["label"])
    split_ratio.append(row["label"])


all_data_stats = {}
for lang in [
    "en",
    "it",
    "ru",
    "sv",
    "zh",  # New dataset
    "no",  # New dataset
    "ja",  # New dataset
    "fr",  # New dataset
]:
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

    if lang == "en":
        model_name = "en_core_web_trf"
    elif lang == "it":
        model_name = "it_core_news_lg"
    elif lang == "ru":
        model_name = "ru_core_news_lg"
    elif lang == "sv":
        model_name = "sv_core_news_lg"
    elif lang == "zh":
        model_name = "zh_core_web_trf"
    elif lang == "no":
        model_name = "nb_core_news_lg"
    elif lang == "ja":
        model_name = "ja_core_news_trf"
    elif lang == "fr":
        model_name = "fr_dep_news_trf"
    else:
        raise ValueError(f"Incorrect lang '{lang}'.")

    nlp = spacy.load(model_name)

    cola = load_dataset_tsv(data_files=data_files)

    vocab = set()

    lang_data_stats = {}
    ratios = []
    split_len = []
    for split in ["train", "dev", "test"]:
        sentences_len = []
        lexical_words_len = []
        global_vocabulary_set = set()
        global_vocabulary_lexical_words_set = set()
        split_ratio = []

        cola[split].map(compute_stats)

        lang_data_stats.update(
            {
                split: {
                    "mean_sentences_len": round(mean(sentences_len), 2),
                    "mean_lexical_words_len": mean(lexical_words_len),
                    "global_vocabulary_set_len": len(global_vocabulary_set),
                    "global_vocabulary_lexical_words_set_len": len(
                        global_vocabulary_lexical_words_set
                    ),
                    "size": len(cola[split]),
                    "global_vocabulary_set_len_normalize": len(global_vocabulary_set)
                    / len(cola[split]),
                    "global_vocabulary_lexical_words_set_len_normalize": len(
                        global_vocabulary_lexical_words_set
                    )
                    / len(cola[split]),
                    "split_ratio": round(sum(split_ratio) / len(split_ratio) * 100, 2),
                }
            }
        )
        split_len.append(round(mean(sentences_len), 2))

    lang_data_stats.update({"ratio": round(sum(ratios) / len(ratios) * 100, 2)})
    lang_data_stats.update({"mean_sentences_len": round(mean(split_len), 2)})
    lang_data_stats.update({"global_vocabulary_set_len": len(vocab)})
    all_data_stats.update({lang: lang_data_stats})

with open("dataset_analysis_cola_datasets.json", "w") as file:
    json.dump(all_data_stats, file)

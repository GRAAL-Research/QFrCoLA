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


all_data_stats = {}
for lang in [
    "en",
    "ru",
    "zh",  # New dataset
    "fr",  # New dataset
]:
    root = "."
    datastore_dir = os.path.join(root, "datastore", "cola_datasets")
    data_dir = os.path.join(datastore_dir, f"{lang}-cola")
    print(data_dir)
    train_data_path = os.path.join(data_dir, "ood.tsv")

    data_files = {
        "train": train_data_path,
    }

    if lang == "en":
        model_name = "en_core_web_trf"
    elif lang == "ru":
        model_name = "ru_core_news_lg"
    elif lang == "zh":
        model_name = "zh_core_web_trf"
    elif lang == "fr":
        model_name = "fr_dep_news_trf"
    else:
        raise ValueError(f"Incorrect lang '{lang}'.")

    nlp = spacy.load(model_name)

    cola = load_dataset_tsv(data_files=data_files)

    vocab = set()

    lang_data_stats = {}
    split_len = []
    for split in ["train"]:
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
                }
            }
        )
        split_len.append(round(mean(sentences_len), 2))

    lang_data_stats.update({"mean_sentences_len": round(mean(split_len), 2)})
    lang_data_stats.update({"global_vocabulary_set_len": len(vocab)})
    all_data_stats.update({lang: lang_data_stats})

with open("dataset_analysis_cola_datasets_ood.json", "w") as file:
    json.dump(all_data_stats, file)

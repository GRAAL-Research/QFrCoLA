import os
import shutil
import subprocess
import warnings

import hydra
import numpy as np
import wandb

# from concurrent.futures import ProcessPoolExecutor # Until fixed in Python 3.12.1

os.environ["TOKENIZERS_PARALLELISM"] = "true"

warnings.filterwarnings("ignore")


@hydra.main(version_base="1.2", config_path="./", config_name="params")
def train(config):
    # Flag variable for debug
    wandb_project = config.wandb_project
    os.environ["WANDB_PROJECT"] = wandb_project

    python_executable_path = config.python_executable_path

    model_save_dir = config.model_save_dir

    epoch = config.epoch
    lr = config.lr
    decay = config.decay
    batch = config.batch

    # We do it for the 4 language (Ru is not in the LA-TDA but in a following article) and also 3 new languages.
    # Namely, ZH, NO and JA.
    for lang in [
        "fr",  # New dataset
        "en",
        "it",
        "ru",
        "sv",
        "zh",  # New dataset
        "no",  # New dataset
        "ja",  # New dataset
        "en-multi",
        "it-multi",
        "ru-multi",
        "sv-multi",
        "zh-multi",  # New dataset
        "no-multi",  # New dataset
        "ja-multi",  # New dataset
        "fr-multi",  # New dataset
    ]:
        print(f"-- Processing for language {lang} --")

        if lang == "en":
            model_name = "bert-base-cased"
        elif lang == "it":
            model_name = "dbmdz/bert-base-italian-cased"
        elif lang == "ru":
            model_name = "ai-forever/ruBert-base"
        elif lang == "sv":
            model_name = "KB/bert-base-swedish-cased"
        elif lang == "zh":
            model_name = "bert-base-chinese"
        elif lang == "no":
            model_name = "NbAiLab/nb-bert-base"
        elif lang == "ja":
            model_name = "cl-tohoku/bert-base-japanese"
        elif lang == "fr":
            model_name = "almanach/camembert-base"
        elif "multi" in lang:
            model_name = "xlm-roberta-base"
            # We keep the lang part of the "lang" in the case of the multi addon, otherwise it will fail at the data_dir
            lang = lang.split("-")[0]
        else:
            raise ValueError(f"Incorrect lang '{lang}'.")

        root = config.root
        datastore_dir = os.path.join(root, "datastore", "cola_datasets")
        data_dir = os.path.join(datastore_dir, f"{lang}-cola")
        print(data_dir)
        train_data = os.path.join(data_dir, "train.tsv")
        dev_data = os.path.join(data_dir, "dev.tsv")
        test_data = os.path.join(data_dir, "test.tsv")

        # Fine-tuning pretrained model over lang 'lang'

        run_name = f"{model_name}-{lang}-cola_{batch}_{lr}_balanced"
        output_dir = model_save_dir + run_name

        subprocess.run(
            f"{python_executable_path} src/train.py \
              --model_name_or_path {model_name} \
              --train_file {train_data} \
              --validation_file {dev_data} \
              --test_file {test_data} \
              --do_eval \
              --do_predict\
              --num_train_epochs {epoch}\
              --learning_rate {lr}\
              --weight_decay {decay}\
              --max_seq_length 64\
              --per_device_train_batch_size {batch}\
              --per_device_eval_batch_size {batch * 4}\
              --output_dir {output_dir}\
              --load_best_model_at_end True\
              --metric_for_best_model eval_loss\
              --evaluation_strategy epoch\
              --save_strategy epoch\
              --logging_strategy epoch\
              --balance_loss\
              --seed {42}\
              --data_seed {42}\
              --run_name {run_name}_transformer\
              --wandb_project {wandb_project}\
              --overwrite_output_dir",
            shell=True,
        )

        if config.clean_seed_directory:
            # We deleted the directory since everything is logged in Wandb and it takes space.
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    train()

import datetime
import gc
import gzip
import json
import os
import pickle
import shutil
import subprocess
import timeit
import warnings
from functools import partial
from math import ceil
from multiprocessing import Pool
from statistics import mean
from time import time

import hydra
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from evaluate import load
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)
from tqdm import tqdm
from transformers import AutoTokenizer

# from concurrent.futures import ProcessPoolExecutor # Until fixed in Python 3.12.1
from process_fix import ProcessPoolExecutor  # Temporary local fix
from src.barcode_feature import (
    get_only_barcodes,
    save_barcodes,
    reformat_barcodes,
    count_ripser_features,
)
from src.features_pre_process import (
    order_files,
    get_token_length,
    split_matricies_and_lengths,
    count_top_stats,
)
from src.metrics import report
from src.opt_threshold_search import print_scores
from src.read_features import read_labels, load_features

ACCURACY = load("accuracy")
MCC = load("matthews_correlation")
Pearson = load("pearsonr")
r2_metric = load("r_squared")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

warnings.filterwarnings("ignore")


@hydra.main(version_base="1.2", config_path="./", config_name="params")
def train(config):
    python_executable_path = config.python_executable_path

    wandb_project = config.wandb_project
    os.environ["WANDB_PROJECT"] = wandb_project

    seeds = config.seeds
    model_save_dir = config.model_save_dir

    epoch = config.epoch
    lr = config.lr
    decay = config.decay
    batch = config.batch

    for lang in [
        "en",  # New dataset
        "en-multi",  # New dataset
    ]:
        print(f"-- Processing for language {lang} --")

        if lang == "en":
            model_name = "bert-base-cased"
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
        ood_data = os.path.join(data_dir, "fluency.tsv")

        # Fine-tuning pretrained model over lang 'lang'

        # In total, we retrain 10 models to seed variance and average of model performance.
        processing_time = []
        for seed in seeds:
            np.random.seed(seed)

            run_name = f"{model_name}-{lang}-cola_{batch}_{lr}_balanced_{seed}"
            output_dir = model_save_dir + run_name

            # Here we retrain the BERT model.
            subprocess.run(
                f"{python_executable_path} src/train_fluency.py \
                  --model_name_or_path {model_name} \
                  --train_file {train_data} \
                  --validation_file {dev_data} \
                  --test_file {test_data} \
                  --hold_out_file {ood_data} \
                  --do_train \
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
                  --seed {seed}\
                  --data_seed {seed}\
                  --run_name {run_name}_transformer\
                  --wandb_project {wandb_project}\
                  --overwrite_output_dir",
                shell=True,
            )

            if model_name != "xlm-roberta-base":
                # We compute the processing time from the point we generate the features.
                # Thus, we also include the grab of the attentions weights. We will later do an average over the 10 runs
                # and per sentence.
                tic = timeit.default_timer()

                # Here we extract the attentions for the feature engineering pre-process (the next step).
                subprocess.run(
                    f"{python_executable_path} -m src.grab_attentions --model_dir {output_dir} "
                    f"--data_file {train_data}",
                    shell=True,
                )
                subprocess.run(
                    f"{python_executable_path} -m src.grab_attentions --model_dir {output_dir} "
                    f"--data_file {dev_data}",
                    shell=True,
                )
                subprocess.run(
                    f"{python_executable_path} -m src.grab_attentions --model_dir {output_dir} "
                    f"--data_file {test_data}",
                    shell=True,
                )
                subprocess.run(
                    f"{python_executable_path} -m src.grab_attentions --model_dir {output_dir} "
                    f"--data_file {ood_data}",
                    shell=True,
                )

                # Process the features pre-processing
                stats_name = "s_w_e_v_c_b0b1_m_k"
                thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
                thrs = len(thresholds_array)
                layers_of_interest = [i for i in range(12)]

                # We do it for each subset
                for subset in ["train", "dev", "test", "fluency"]:
                    print(
                        f"-- Processing feature for model with seed {seed} and with subset: {subset} --"
                    )
                    model_path = output_dir

                    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)

                    attn_dir = model_path + "/attentions/"
                    adj_filenames = order_files(path=attn_dir, subset=subset)

                    stats_file = (
                        model_path
                        + "/features/"
                        + subset
                        + "_"
                        + stats_name
                        + "_array_"
                        + str(thrs)
                        + ".npy"
                    )

                    subset_data_file_path = os.path.join(data_dir, subset + ".tsv")
                    data = pd.read_csv(subset_data_file_path, sep="\t")

                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    max_tokens_amount = 64
                    max_len = max_tokens_amount

                    get_token_length_partial = partial(
                        get_token_length, tokenizer=tokenizer, max_len=max_len
                    )

                    data_hf = Dataset.from_pandas(data).map(
                        get_token_length_partial,
                        batched=True,
                        batch_size=batch * 4,
                        desc="Running tokenizer on dataset",
                        num_proc=2,
                        remove_columns=[
                            column_name
                            for column_name in data.columns.tolist()
                            if column_name != "sentence"
                        ],
                    )

                    data["tokenizer_length"] = data_hf["tokenizer_length"]
                    ntokens_array = data["tokenizer_length"].values

                    # Process using 3/4 the number of CPU core plus 1 on the computer.
                    # e.g. on 16 cores, it will use 13 cores.
                    num_of_workers = int(os.cpu_count() // 4 * 3 + 1)

                    batch_size = (
                        10  # batch size for the features processing (per CPU core)
                    )
                    number_of_batches = ceil(len(data["sentence"]) / batch_size)
                    DUMP_SIZE = 100  # number of batches to be dumped
                    batched_sentences = np.array_split(
                        data["sentence"].values, number_of_batches
                    )

                    stats_cap = 500
                    stats_tuple_lists_array = []

                    pool = Pool(num_of_workers)

                    print("-- Start of the calculating of topological features --")
                    for i, filename in enumerate(
                        tqdm(adj_filenames, desc="Calculating topological features")
                    ):
                        if "gz" in filename:
                            with gzip.GzipFile(filename, "rb") as f:
                                adj_matricies = np.load(f, allow_pickle=True)
                        else:
                            with open(filename, "rb") as f:
                                adj_matricies = np.load(f, allow_pickle=True)
                        ntokens = ntokens_array[
                            i
                            * batch_size
                            * DUMP_SIZE : (i + 1)
                            * batch_size
                            * DUMP_SIZE
                        ]
                        splitted = split_matricies_and_lengths(
                            adj_matricies, ntokens, num_of_workers
                        )
                        args = [
                            (
                                m,
                                thresholds_array,
                                ntokens,
                                stats_name.split("_"),
                                stats_cap,
                            )
                            for m, ntokens in splitted
                        ]
                        stats_tuple_lists_array_part = pool.starmap(
                            count_top_stats, args
                        )
                        stats_tuple_lists_array.append(
                            np.concatenate(
                                [_ for _ in stats_tuple_lists_array_part], axis=3
                            )
                        )

                    # We close the pool after the execution
                    pool.close()
                    pool.terminate()
                    pool.join()
                    del pool
                    gc.collect()

                    print("-- Save of the stats tuple list array -- ")
                    stats_tuple_lists_array = np.concatenate(
                        stats_tuple_lists_array, axis=3
                    )
                    np.save(stats_file, stats_tuple_lists_array)

                    # Compute the template features
                    print("-- Start of the template features process --")
                    subprocess.run(
                        f"{python_executable_path} -m src.template_features --model_dir {output_dir} "
                        f"--data_file {subset_data_file_path} --num_of_workers {num_of_workers}",
                        shell=True,
                    )

                    # Compute the Barcode and Ripser features
                    print("-- Starting compute of the Barcode and Ripser++ features --")

                    r_file = attn_dir + subset
                    barcodes_dir = model_path + "/features/barcodes/"
                    os.makedirs(barcodes_dir, exist_ok=True)
                    # we want tho have file as '`subset`_...' and not '/`subset`/...
                    barcodes_file = barcodes_dir + subset

                    adj_matricies = []
                    assert number_of_batches == len(batched_sentences)  # sanity check

                    dim = 1
                    lower_bound = 1e-3

                    # Processing of the features

                    for i, filename in enumerate(
                        tqdm(adj_filenames, desc="Barcodes calculation")
                    ):
                        part = filename.split("_")[-1].split(".")[0]
                        if os.path.isfile(barcodes_file + "_" + part + ".json"):
                            print("file already exists")
                            print("passing", barcodes_file + "_" + part + ".json")
                            continue

                        if "gz" in filename:
                            with gzip.GzipFile(filename, "rb") as f:
                                adj_matricies = np.load(f, allow_pickle=True)
                        else:
                            with open(filename, "rb") as f:
                                adj_matricies = np.load(f, allow_pickle=True)
                        ntokens = ntokens_array[
                            i
                            * batch_size
                            * DUMP_SIZE : (i + 1)
                            * batch_size
                            * DUMP_SIZE
                        ]

                        # We use a PoolExecutor to properly handle Risper++ process memory release as suggested by
                        # this issue https://github.com/simonzhang00/ripser-plusplus/issues/5.
                        with ProcessPoolExecutor(max_workers=1) as executor:
                            barcodes = executor.submit(
                                get_only_barcodes,
                                adj_matricies=adj_matricies,
                                ntokens_array=ntokens,
                                dim=dim,
                                lower_bound=lower_bound,
                                verbose=True,
                            ).result()

                        save_barcodes(barcodes, barcodes_file + "_" + part + ".json")

                    print("-- Processing of Ripser++ features --")
                    # Barcodes' Ripser Features
                    ripser_features = [
                        "h0_s",
                        "h0_e",
                        "h0_t_d",
                        "h0_n_d_m_t0.75",
                        "h0_n_d_m_t0.5",
                        "h0_n_d_l_t0.25",
                        "h1_t_b",
                        "h1_n_b_m_t0.25",
                        "h1_n_b_l_t0.95",
                        "h1_n_b_l_t0.70",
                        "h1_s",
                        "h1_e",
                        "h1_v",
                        "h1_nb",
                    ]

                    json_filenames = [
                        output_dir + "/features/barcodes/" + filename
                        for filename in os.listdir(model_path + "/features/barcodes/")
                        if r_file.split("/")[-1] in filename.split("_part")[0]
                    ]
                    json_filenames = sorted(
                        json_filenames,
                        key=lambda x: int(x.split("_")[-1].split("of")[0][4:].strip()),
                    )

                    features_array = []

                    for filename in tqdm(json_filenames, desc="Computing Ripser++"):
                        barcodes = json.load(open(filename))
                        print(f"Barcodes loaded from: {filename}", flush=True)
                        features_part = []
                        for layer in barcodes:
                            features_layer = []
                            for head in barcodes[layer]:
                                ref_barcodes = reformat_barcodes(barcodes[layer][head])
                                features = count_ripser_features(
                                    ref_barcodes, ripser_features
                                )
                                features_layer.append(features)
                            features_part.append(features_layer)
                        features_array.append(np.asarray(features_part))

                    features = np.concatenate(features_array, axis=2)
                    ripser_file = f"{model_path}/features/{subset}_ripser.npy"
                    np.save(ripser_file, features)

                toc = timeit.default_timer()

                print(
                    f"Time taken to process the seed {seed} is: {toc - tic} over the three splits."
                )  # elapsed time in seconds
                processing_time.append(toc - tic)

                print("---- Start training of features classificator ----")
                file_type = ".tsv"
                train_set_name, valid_set_name, test_set_name, ood_set_name = (
                    "train",
                    "dev",
                    "test",
                    "fluency",
                )
                data_args = {"data_dir": data_dir, "file_type": file_type}
                (_, y_train), (_, y_valid), (_, y_test), (_, y_ood) = list(
                    map(
                        lambda x_: read_labels(x_, **data_args),
                        [
                            x_
                            for x_ in (
                                train_set_name,
                                valid_set_name,
                                test_set_name,
                                ood_set_name,
                            )
                        ],
                    )
                )

                topological_thr = 6
                features_dir = model_path + "/features/"

                kwargs = {
                    "features_dir": features_dir,
                    "model_path": model_path,
                    "topological_thr": topological_thr,
                    "heads": 12,
                    "layers": 12,
                }

                X_train, X_valid, X_test, X_ood = list(
                    map(
                        lambda x_: load_features(
                            x_, topological_features=stats_name, **kwargs
                        ),
                        [
                            x_
                            for x_ in (
                                train_set_name,
                                valid_set_name,
                                test_set_name,
                                ood_set_name,
                            )
                        ],
                    )
                )

                data_files = {
                    "train": train_data,
                    "dev": dev_data,
                    "test": test_data,
                }

                X_train = X_train.iloc[:, ~X_train.columns.str.startswith("w")]
                X_valid = X_valid.loc[:, X_train.columns]
                X_test = X_test.loc[:, X_train.columns]
                X_ood = X_ood.loc[:, X_train.columns]

                # Removing constant and quasi-constant features
                var_thr = VarianceThreshold(threshold=0.00001)
                var_thr.fit(X_valid)
                not_constant_f = var_thr.get_support()
                X_train = X_train.loc[:, not_constant_f]
                X_valid = X_valid.loc[:, not_constant_f]
                X_test = X_test.loc[:, not_constant_f]
                X_ood = X_ood.loc[:, not_constant_f]

                # Parameters grid
                params = {
                    "tol": 1e-6,
                    "random_state": seed,
                    "solver": "liblinear",
                    "penalty": "l1",
                }
                N_FEATURES_OPTIONS = np.arange(100, 300, 50)
                C_OPTIONS = [1e-3, 0.01, 0.1]
                CLASS_WEIGHT = [None]

                max_iter_range = [25, 100]
                params_grid = {
                    "reduce_dim__n_components": N_FEATURES_OPTIONS,
                    "clf__max_iter": max_iter_range,
                    "clf__C": C_OPTIONS,
                    "clf__class_weight": CLASS_WEIGHT,
                }
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("reduce_dim", PCA(whiten=True, random_state=seed)),
                        ("clf", LogisticRegression(**params)),
                    ]
                )
                kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

                clf_ = GridSearchCV(
                    pipeline,
                    cv=kfold,
                    verbose=4,
                    param_grid=params_grid,
                    scoring="accuracy",
                    n_jobs=num_of_workers,
                    pre_dispatch="n_jobs",  # We lower pre-dispatch since in some case we get an OOM.
                )

                YEAR = datetime.date.today().year  # the current year
                MONTH = datetime.date.today().month  # the current month
                DATE = datetime.date.today().day  # the current day
                HOUR = datetime.datetime.now().hour  # the current hour
                MINUTE = datetime.datetime.now().minute  # the current minute
                SECONDS = datetime.datetime.now().second  # the current second

                print(
                    "----",
                    "The current start of the GridSearch is:",
                    f"{YEAR}-{MONTH}-{DATE} {HOUR}:{MINUTE}:{SECONDS}",
                    "----",
                )

                start = time()

                clf_.fit(X_train, y_train)
                print(
                    "GridSearchCV took %.2f seconds for %d candidate parameter settings."
                    % (time() - start, len(clf_.cv_results_["params"]))
                )
                report(clf_.cv_results_, n_top=5)

                wandb.init(project=wandb_project, name=run_name + "_classifier")

                valid_res_metrics = print_scores(
                    clf_.best_estimator_.predict(X_valid), y_valid
                )
                wandb.log({"dev/lda-acc": valid_res_metrics[0]})
                wandb.log({"dev/lda-mcc": valid_res_metrics[1]})

                test_res_metrics = print_scores(
                    clf_.best_estimator_.predict(X_test), y_test
                )
                wandb.log({"test/lda-acc": test_res_metrics[0]})
                wandb.log({"test/lda-mcc": test_res_metrics[1]})

                # Eval on fluency dataset

                probs_preds = clf_.best_estimator_.predict_proba(X_ood)
                probs_preds = probs_preds.max(-1)
                labels = y_ood

                rmse = mean_squared_error(
                    y_true=labels, y_pred=probs_preds, squared=False
                )
                r_squared = r2_metric.compute(
                    predictions=probs_preds, references=labels
                )

                mcc_result = MCC.compute(predictions=probs_preds, references=labels)
                pearson_restults = Pearson.compute(
                    predictions=probs_preds, references=labels, return_pvalue=True
                )

                mean_score_pred = probs_preds.mean()
                st_dev_score_pred = probs_preds.std()
                mean_score_label = labels.mean()
                st_dev_score_label = labels.std()

                metrics = {
                    "fluency_rmse": float(rmse),
                    "fluency_R2": float(r_squared),
                    "fluency_mcc": float(mcc_result["matthews_correlation"]),
                    "fluency_pearson_corr": float(pearson_restults["pearsonr"]),
                    "fluency_mean_score_pred": float(mean_score_pred),
                    "fluency_st_dev_score_pred": float(st_dev_score_pred),
                    "fluency_mean_score_label": float(mean_score_label),
                    "fluency_st_dev_score_label": float(st_dev_score_label),
                }

                wandb.log(metrics)

                wandb.config.update({"tag": model_name})

                with open(
                    os.path.join(output_dir, "predict_la_tda.pickle"), "rb"
                ) as file:
                    pickle.dump(probs_preds, file)

                wandb.finish(exit_code=0)

                print(
                    f"Average time taken to process the features over the 10 runs for all three splits is: "
                    f"{mean(processing_time)}"
                )

                with open(
                    os.path.join(".", f"reproduce_results_time_{lang}.txt"), "w"
                ) as file:
                    print(
                        f"Average time taken to process the features over the 10 runs for all three splits is: "
                        f"{mean(processing_time)}",
                        file=file,
                    )
                with open(
                    os.path.join(".", f"reproduce_results_time_{lang}.pickle"), "wb"
                ) as file:
                    pickle.dump(processing_time, file)

        if config.clean_seed_directory:
            # We deleted the directory since everything is logged in Wandb and it takes space.
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    train()

import pandas as pd

import wandb

api = wandb.Api()

runs = api.runs("davebulaval/la-tda-reproduce", per_page=1000)

results_df_tf = pd.DataFrame()
results_df_la_tda = pd.DataFrame()
for run in runs:
    if run.config.get("run_name") is not None:
        # A Transformer model
        tag = run.config.get("run_name").split("-cola")[0]
        summary = run.summary

        # We cast the HTTPSummary into a dict for easier manipulation
        clean_summary = dict(summary)

        # We will remove all key/item that are related to "train", samples_per_second, runtime, _wandb, and _timestap.
        # i.e. we only key test/train mcc/accuracy data.
        keys_to_keep = [
            key
            for key in clean_summary.keys()
            if "train" not in key and ("mcc" in key or "accuracy" in key)
        ]

        run_data = {
            key: [value] for key, value in clean_summary.items() if key in keys_to_keep
        }
        run_data.update({"tag": [tag]})

        results_df_tf = pd.concat(
            [results_df_tf, pd.DataFrame.from_dict(run_data)], ignore_index=True
        )

    else:
        # A classifier (LA-TDA)
        tag = run.name.split("-cola")[0]
        summary = run.summary

        # We cast the HTTPSummary into a dict for easier manipulation
        clean_summary = dict(summary)

        # We will remove all key/item that are related to "train", samples_per_second, runtime, _wandb, and _timestap.
        # i.e. we only key test/train mcc/accuracy data.
        keys_to_keep = [
            key
            for key in clean_summary.keys()
            if "train" not in key and ("mcc" in key or "acc" in key)
        ]

        run_data = {
            key: [value] for key, value in clean_summary.items() if key in keys_to_keep
        }
        run_data.update({"tag": [tag]})

        results_df_la_tda = pd.concat(
            [results_df_la_tda, pd.DataFrame.from_dict(run_data)], ignore_index=True
        )

results_df_tf = results_df_tf[sorted(results_df_tf)]

# We compute the mean and STD of the data
results_df_tf.groupby("tag").mean().round(4), results_df_tf.groupby("tag").std().round(
    4
)

results_df_la_tda = results_df_la_tda[sorted(results_df_la_tda)]
results_df_la_tda.groupby("tag").mean().round(4), results_df_la_tda.groupby(
    "tag"
).std().round(4)

runs = api.runs("davebulaval/la-tda-reproduce-fr", per_page=1000)

fine_tuning = pd.DataFrame()
lda = pd.DataFrame()
for run in runs:
    if run.config.get("run_name") is not None:
        seed = run.config.get("seed")
        tag = run.config.get("run_name").split("-cola")[0]

        summary = run.summary

        # We cast the HTTPSummary into a dict for easier manipulation
        clean_summary = dict(summary)

        keys_to_keep = [
            key
            for key in clean_summary.keys()
            if "hold_out" in key and ("mcc" in key or "acc" in key)
        ]

        run_data = {
            key: [value] for key, value in clean_summary.items() if key in keys_to_keep
        }
        run_data.update({"seed": [seed]})
        run_data.update({"tag": [tag]})

        fine_tuning = pd.concat(
            [fine_tuning, pd.DataFrame.from_dict(run_data)], ignore_index=True
        )
    else:
        tag = run.name.split("-cola")[0]
        seed = run.name.split("_balanced_")[-1].split("_")[0]
        summary = run.summary

        # We cast the HTTPSummary into a dict for easier manipulation
        clean_summary = dict(summary)

        keys_to_keep = [
            key
            for key in clean_summary.keys()
            if "hold_out" in key and ("mcc" in key or "acc" in key)
        ]

        run_data = {
            key: [value] for key, value in clean_summary.items() if key in keys_to_keep
        }
        run_data.update({"seed": [int(seed)]})
        run_data.update({"tag": [tag]})

        lda = pd.concat([lda, pd.DataFrame.from_dict(run_data)], ignore_index=True)

# Finetuning
fine_tuning.drop_duplicates(subset=["seed", "tag"], inplace=True)
fine_tuning.groupby("tag").mean().round(4), fine_tuning.groupby("tag").std().round(4)


lda.drop_duplicates(subset=["seed", "tag"], inplace=True)
lda["tag"] = lda["tag"].astype(str)
lda.groupby("tag").mean().round(4), lda.groupby("tag").std().round(4)


runs = api.runs("davebulaval/la-tda-reproduce-fr-cat", per_page=1000)

fine_tuning = pd.DataFrame()
lda = pd.DataFrame()
for run in runs:
    if run.config.get("run_name") is not None:
        seed = run.config.get("seed")
        tag = run.config.get("run_name").split("-cola")[0]

        summary = run.summary

        # We cast the HTTPSummary into a dict for easier manipulation
        clean_summary = dict(summary)

        keys_to_keep = [
            key
            for key in clean_summary.keys()
            if "test" in key and ("_mcc" in key or "_acc" in key)
        ]

        run_data = {
            key: [value] for key, value in clean_summary.items() if key in keys_to_keep
        }
        run_data.update({"seed": [seed]})
        run_data.update({"tag": [tag]})

        fine_tuning = pd.concat(
            [fine_tuning, pd.DataFrame.from_dict(run_data)], ignore_index=True
        )
    else:
        tag = run.name.split("-cola")[0]
        seed = run.name.split("_balanced_")[-1].split("_")[0]
        summary = run.summary

        # We cast the HTTPSummary into a dict for easier manipulation
        clean_summary = dict(summary)

        keys_to_keep = [
            key
            for key in clean_summary.keys()
            if "test" in key and ("_mcc" in key or "_acc" in key)
        ]

        run_data = {
            key: [value] for key, value in clean_summary.items() if key in keys_to_keep
        }
        run_data.update({"seed": [int(seed)]})
        run_data.update({"tag": [tag]})

        lda = pd.concat([lda, pd.DataFrame.from_dict(run_data)], ignore_index=True)

# Finetuning
fine_tuning.groupby("tag").mean().round(4), fine_tuning.groupby("tag").std().round(4)


lda.drop_duplicates(subset=["seed", "tag"], inplace=True)
lda["tag"] = lda["tag"].astype(str)
lda.groupby("tag").mean().round(4), lda.groupby("tag").std().round(4)

print("a")

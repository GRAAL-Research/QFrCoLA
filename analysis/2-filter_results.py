"""
Fetch LLM and fine-tuned model results from wandb and produce a unified CSV.

Sources:
  - doctorate/COLE-final       → qfrcola.accuracy  (in-domain)
  - doctorate/QFrCoLA-OOD      → qfrcola_ood.accuracy  (OOD Académie française)
  - davebulaval/la-tda-reproduce-fr → fine-tuned camembert / xlm-roberta,
        test/accuracy (QFrCoLA) and train/hold_out/_accuracy (OOD),
        averaged over 10 seeds

Output: results/qfrcola_filtered_accuracies.csv
        columns: name, qfrcola.accuracy, academie_francaise.accuracy
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb


def extract_accuracy(summary_value):
    """Extract float accuracy from a wandb summary value.

    The value may be a plain float, or a dict-like object (wandb SummarySubDict)
    with an "accuracy" key.
    """
    if summary_value is None:
        return None
    if hasattr(summary_value, "get"):
        # Handles dict and wandb.old.summary.SummarySubDict
        val = summary_value.get("accuracy")
        if val is not None:
            return float(val)
    if isinstance(summary_value, (int, float)):
        return float(summary_value)
    return None


# ── 1. Fetch LLM results from COLE-final (QFrCoLA in-domain) ────────────

def fetch_cole_final(api):
    """Fetch qfrcola.accuracy per model from doctorate/COLE-final."""
    runs = api.runs("doctorate/COLE-final")

    records = []
    for run in runs:
        name = run.name
        val = run.summary.get("qfrcola.accuracy")
        acc = extract_accuracy(val)
        if acc is not None:
            records.append({"name": name, "qfrcola.accuracy": acc})

    df = pd.DataFrame(records)
    print(f"[COLE-final] Fetched {len(df)} runs with qfrcola.accuracy")
    return df


# ── 2. Fetch LLM results from QFrCoLA-OOD (Académie française) ──────────

def fetch_qfrcola_ood(api):
    """Fetch qfrcola_ood.accuracy per model from doctorate/QFrCoLA-OOD."""
    runs = api.runs("doctorate/QFrCoLA-OOD")

    records = []
    for run in runs:
        name = run.name
        val = run.summary.get("qfrcola_ood.accuracy")
        acc = extract_accuracy(val)
        if acc is not None:
            records.append({"name": name, "academie_francaise.accuracy": acc})

    df = pd.DataFrame(records)
    print(f"[QFrCoLA-OOD] Fetched {len(df)} runs with qfrcola_ood.accuracy")
    return df


# ── 3. Fetch fine-tuned models from la-tda-reproduce-fr ──────────────────

def extract_base_model_name(run_name):
    """Extract base model name from fine-tuned run name.

    Run names follow the pattern:
        almanach/camembert-base-fr-cola_32_3e-05_balanced_42_transformer
        xlm-roberta-base-fr-cola_32_3e-05_balanced_42_transformer
    Returns the part before '-fr-cola_'.
    """
    marker = "-fr-cola_"
    if marker in run_name:
        return run_name.split(marker)[0]
    return run_name


def fetch_finetuned(api):
    """Fetch camembert and xlm-roberta results, averaged over 10 seeds."""
    runs = api.runs("davebulaval/la-tda-reproduce-fr")

    # Deduplicate: keep only the latest run per run name
    latest_by_name = {}  # run_name -> run (keep most recent)
    for run in runs:
        name = run.name
        if name not in latest_by_name:
            latest_by_name[name] = run
        else:
            # Keep the one with the later created_at timestamp
            if run.created_at > latest_by_name[name].created_at:
                latest_by_name[name] = run

    # Collect per-model seed results
    model_results = defaultdict(list)  # base_model_name -> list of (qfrcola_acc, ood_acc)

    for run in latest_by_name.values():
        qfrcola_acc = run.summary.get("test/accuracy")
        ood_acc = run.summary.get("train/hold_out/_accuracy")

        if qfrcola_acc is None or ood_acc is None:
            continue

        base_model = extract_base_model_name(run.name)
        model_results[base_model].append((float(qfrcola_acc), float(ood_acc)))

    records = []
    for model_name, seed_results in model_results.items():
        qfrcola_accs = [r[0] for r in seed_results]
        ood_accs = [r[1] for r in seed_results]
        records.append(
            {
                "name": model_name,
                "qfrcola.accuracy": np.mean(qfrcola_accs),
                "academie_francaise.accuracy": np.mean(ood_accs),
                "_n_seeds": len(seed_results),
            }
        )
        print(
            f"  {model_name}: {len(seed_results)} seeds, "
            f"qfrcola={np.mean(qfrcola_accs):.4f}, ood={np.mean(ood_accs):.4f}"
        )

    df = pd.DataFrame(records)
    print(f"[la-tda-reproduce-fr] Fetched {len(df)} fine-tuned models")
    return df


# ── Main pipeline ────────────────────────────────────────────────────────

def main():
    print("Fetching results from wandb...\n")

    api = wandb.Api()

    df_cole = fetch_cole_final(api)
    print()
    df_ood = fetch_qfrcola_ood(api)
    print()
    df_ft = fetch_finetuned(api)
    print()

    # Join LLM results: inner join on name (only keep models with both scores)
    df_llm = pd.merge(df_cole, df_ood, on="name", how="inner")
    print(f"LLM models with both scores: {len(df_llm)}")

    # --- Specific manipulations ---

    # Transfer qfrcola accuracy from unsloth/gpt-oss-20b to openai/gpt-oss-20b
    source_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    target_name = "openai/gpt-oss-20b"
    source_mask = df_llm["name"] == source_name
    target_mask = df_llm["name"] == target_name

    if source_mask.any() and target_mask.any():
        source_value = df_llm.loc[source_mask, "qfrcola.accuracy"].values[0]
        df_llm.loc[target_mask, "qfrcola.accuracy"] = source_value
        df_llm = df_llm[~source_mask]
        print(f"Transferred qfrcola from '{source_name}' to '{target_name}', deleted source row")

    # Remove CohereForAI/aya-expanse-8b
    row_to_remove = "CohereForAI/aya-expanse-8b"
    if (df_llm["name"] == row_to_remove).any():
        df_llm = df_llm[df_llm["name"] != row_to_remove]
        print(f"Removed '{row_to_remove}'")

    # Combine LLMs and fine-tuned models
    df_ft_clean = df_ft[["name", "qfrcola.accuracy", "academie_francaise.accuracy"]]
    combined = pd.concat([df_llm, df_ft_clean], ignore_index=True)

    # Validate: check for missing values
    check_cols = ["qfrcola.accuracy", "academie_francaise.accuracy"]
    missing_mask = combined[check_cols].isna().any(axis=1)
    missing_rows = combined[missing_mask]

    if not missing_rows.empty:
        print(f"\nWARNING: {len(missing_rows)} rows with missing values:")
        for _, row in missing_rows.iterrows():
            missing_cols = [c for c in check_cols if pd.isna(row[c])]
            print(f"  '{row['name']}' missing: {', '.join(missing_cols)}")
    else:
        print("\nAll rows have complete data.")

    print(f"\nTotal models: {len(combined)}")
    print(combined.head(10))

    # Save
    output_path = os.path.join("results", "qfrcola_filtered_accuracies.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to '{output_path}'")


if __name__ == "__main__":
    main()

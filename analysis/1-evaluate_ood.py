"""
Evaluate LLMs on the Académie française OOD dataset.

This script reuses the COLE evaluation infrastructure to evaluate models
on the out-of-distribution (OOD) binary grammatical judgement task from
the Académie française corpus (ood/ood.tsv).

Usage:
    python 1-evaluate_ood.py --models_name RandomBaselineModel
    python 1-evaluate_ood.py --models_name "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    python 1-evaluate_ood.py --llm_split 1 --batch_size 16
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import pandas as pd
import wandb
from tqdm import tqdm

# Add COLE to sys.path so we can import its modules
sys.path.insert(0, "/home/david/Github/COLE")

from predictions.all_llms import llms, private_llm, small_llm, small_llm_2
from src.dataset.dataset import Dataset
from src.dataset.prompt_builder import PromptBuilder
from src.evaluation.llm_evaluator import ModelEvaluator
from src.evaluation.tools import split_llm_list
from src.task.task import Task, TaskType

# ── OOD Dataset (local TSV) ─────────────────────────────────────────────

OOD_TSV_PATH = os.path.join(os.path.dirname(__file__), "..", "ood", "ood.tsv")
WANDB_PROJECT = "QFrCoLA-OOD"
WANDB_ENTITY = "doctorate"
TASK_NAME = "qfrcola_ood"


def load_ood_data():
    """Load the OOD Académie française TSV as a list of dicts."""
    tsv_path = os.path.abspath(OOD_TSV_PATH)
    df = pd.read_csv(tsv_path, sep="\t")
    return df.to_dict(orient="records")


def build_ood_dataset():
    """Build the OOD Académie française dataset with the same prompt as QFrCoLA."""
    return Dataset(
        name=TASK_NAME,
        description=(
            "Binary grammatical judgement (OOD): "
            "Predicts whether a sentence is grammatically correct (1) or not (0). "
            "Sentences from the Académie française."
        ),
        possible_ground_truths=["0", "1"],
        line_to_truth_fn=lambda line: str(line["label"]),
        line_to_prompt_fn=lambda line: PromptBuilder()
        .add_premise("Juge si cette phrase est grammaticalement correcte :")
        .add_data(line["sentence"])
        .add_end(
            "Réponds avec seulement 1 si la phrase est grammaticalement correcte, 0 sinon. La réponse est :"
        )
        .build(),
        line_to_data_fn=lambda line: line["sentence"],
        data=load_ood_data(),
    )


class OODTask(Task):
    """Task subclass that uses a local dataset instead of the datasets registry."""

    def __init__(self, dataset_obj):
        super().__init__(
            task_name=TASK_NAME,
            metric="accuracy",
            task_type=TaskType.INFERENCE,
            dataset=dataset_obj,
        )


def main():
    # ── CLI ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Evaluate LLMs on OOD Académie française")
    parser.add_argument(
        "--models_name",
        "-mn",
        help="Model name(s), comma-separated, or a group key from all_llms.py.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for HF models.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--llm_split",
        help="Split of the LLMs list (1, 2, or 3).",
        type=int,
        default=None,
        choices=[1, 2, 3],
    )
    parser.add_argument(
        "--skip_first_n",
        help="Number of LLMs to skip in the list.",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # ── Resolve model list ───────────────────────────────────────────────
    all_models = llms["all"] + small_llm["all"] + small_llm_2["all"] + private_llm["all"]
    all_model_groups = {**llms, **small_llm, **small_llm_2, **private_llm}

    if args.models_name is not None:
        if args.models_name in all_model_groups:
            models = all_model_groups[args.models_name]
        else:
            models = args.models_name.split(",")
    else:
        models = all_models

    models = split_llm_list(models=models, llm_split=args.llm_split)

    if args.skip_first_n is not None:
        models = models[args.skip_first_n :]

    # ── Build task ───────────────────────────────────────────────────────
    ood_dataset = build_ood_dataset()
    ood_task = OODTask(dataset_obj=ood_dataset)
    tasks = [ood_task]

    print(f"OOD dataset loaded: {len(ood_dataset)} examples")
    print(f"Models to evaluate: {len(models)}")

    # ── Evaluation loop ─────────────────────────────────────────────────
    logging.info("Starting OOD Evaluation")
    time_start = datetime.now()

    for model_name in tqdm(models, total=len(models), desc="Evaluating on OOD"):
        try:
            # Lazy import: avoid pulling in unsloth/torch when only using API models
            if model_name in private_llm["all"]:
                from src.language_model.private_lm import RemoteLLMModel
                model = RemoteLLMModel(model_name=model_name)
            elif model_name == "RandomBaselineModel":
                from src.language_model.baseline import RandomBaselineModel
                model = RandomBaselineModel(model_name="random_baseline")
            else:
                from src.evaluation.llm_factory import model_factory
                model = model_factory(model_name, batch_size=args.batch_size)
            evaluator = ModelEvaluator()

            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config={
                    "model_name": model_name,
                    "tasks": TASK_NAME,
                    "batch_size": args.batch_size,
                },
                name=model_name,
            )

            predictions_payload = evaluator.evaluate(model, tasks)
            wandb.log(predictions_payload)

            # Compute metrics manually (can't use evaluator.compute_metrics()
            # because it calls tasks_factory which doesn't know qfrcola_ood)
            preds = predictions_payload["tasks"][0][TASK_NAME]
            metric_score, warning = ood_task.compute(preds)
            wandb.log({f"{TASK_NAME}.accuracy": {**metric_score}})
            if warning:
                logging.warning(warning)

        except Exception as e:
            error_message = f"Evaluation failed for model {model_name}: {e}"
            logging.error(error_message)
            wandb.finish(exit_code=1)
            continue
        finally:
            if "model" in locals():
                del model
            if "evaluator" in locals():
                del evaluator
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            wandb.finish(exit_code=0)

    time_end = datetime.now()
    print(f"Done. Elapsed time: {time_end - time_start}")


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn.functional as F
from evaluate import load
from sklearn.metrics import mean_squared_error
from transformers import (
    EvalPrediction,
)

ACCURACY = load("accuracy")
MCC = load("matthews_correlation")
Pearson = load("pearsonr")
r2_metric = load("r_squared")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    preds = np.argmax(preds, axis=1)

    acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
    mcc_result = MCC.compute(predictions=preds, references=p.label_ids)
    pearson_restults = Pearson.compute(
        predictions=preds, references=p.label_ids, return_pvalue=True
    )

    result = {
        "accuracy": acc_result["accuracy"],
        "mcc": mcc_result["matthews_correlation"],
        "pearson": pearson_restults["pearsonr"],
    }

    return result


def compute_metrics_probs(p: EvalPrediction):
    # We get the highest probability amongst the two predicted by the model.
    # We also multiply it by 100 since fluency score are in [0, 100].
    probs_preds = (
        F.softmax(torch.tensor(p.predictions, dtype=torch.long), dim=-1).max(-1).values
        * 100
    )

    labels = p.label_ids
    rmse = mean_squared_error(y_true=labels, y_pred=probs_preds, squared=False)
    r_squared = r2_metric.compute(predictions=probs_preds, references=labels)

    mcc_result = MCC.compute(predictions=probs_preds, references=labels)
    pearson_restults = Pearson.compute(
        predictions=probs_preds, references=labels, return_pvalue=True
    )

    mean_score_pred = probs_preds.numpy().mean()
    st_dev_score_pred = probs_preds.numpy().std()
    mean_score_label = labels.mean()
    st_dev_score_label = labels.std()

    results = {
        "fluency_rmse": float(rmse),
        "fluency_R2": float(r_squared),
        "fluency_mcc": float(mcc_result["matthews_correlation"]),
        "fluency_pearson_corr": float(pearson_restults["pearsonr"]),
        "fluency_mean_score_pred": float(mean_score_pred),
        "fluency_st_dev_score_pred": float(st_dev_score_pred),
        "fluency_mean_score_label": float(mean_score_label),
        "fluency_st_dev_score_label": float(st_dev_score_label),
    }
    return results

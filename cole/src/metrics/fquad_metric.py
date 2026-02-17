import re
import string
from collections import Counter
from typing import Dict, List

from src.metrics.metrics_wrapper import Metric


def normalize_answer(answer: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Based on the SQUAD official metric: https://huggingface.co/spaces/evaluate-metric/squad
    """

    def remove_articles(text):
        return re.sub(r"\b(le|la|l'|du|des|aux|un|une)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    answer = str(answer)
    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_score(predictions: List, references: List) -> Dict:
    f1 = exact_match = total = 0
    for prediction, reference in zip(predictions, references):
        total += 1

        ground_truths = reference["text"]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


class FQuAD(Metric):
    def compute(self, predictions: List, references: List) -> Dict:
        score = compute_score(predictions=predictions, references=references)
        return score

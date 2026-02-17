# pylint: disable=unused-argument

import abc
import logging
from abc import ABC
from typing import List, Dict

from evaluate import load

from src import NA_VALUE


class Metric(ABC):
    @abc.abstractmethod
    def compute(self, predictions, references) -> Dict:
        pass


class AccuracyWrapper(Metric):
    def __init__(self):
        self._metric = load("accuracy")

    def compute(self, predictions: List, references: List, **kwargs) -> Dict:
        clean_predictions = apply_int_casting(predictions_to_clean=predictions)
        return self._metric.compute(
            predictions=clean_predictions, references=references
        )


class PearsonCorrelation(Metric):
    def __init__(self):
        self._metric = load("pearsonr")

    def compute(self, predictions: List, references: List) -> Dict:
        clean_predictions = apply_int_casting(predictions_to_clean=predictions)
        return self._metric.compute(
            predictions=clean_predictions, references=references, return_pvalue=False
        )


class F1Score(Metric):
    def __init__(self):
        self._metric = load("f1")

    def compute(self, predictions: List, references: List) -> Dict:
        clean_predictions = apply_int_casting(predictions_to_clean=predictions)
        return self._metric.compute(
            predictions=clean_predictions, references=references
        )


class ExactMatch(Metric):
    def compute(self, predictions: List, references: List, **kwargs) -> Dict:
        score = [
            reference.strip() == prediction.strip()
            for reference, prediction in zip(references, predictions)
        ]
        return {"exact_match": sum(score) / len(score)}


def apply_int_casting(predictions_to_clean: List) -> List:
    na_value = 0
    none_value = 0
    undetected_value = 0
    for idx, prediction in enumerate(predictions_to_clean):
        if isinstance(prediction, int):
            # Case where the prediction is already an int.
            # We use this branch since we want an else statement to capture undetected type.
            pass
        elif isinstance(prediction, float):
            predictions_to_clean[idx] = int(prediction)
        elif isinstance(prediction, str):
            if prediction.strip().isdigit():
                predictions_to_clean[idx] = int(prediction)
            else:
                na_value += 1
                predictions_to_clean[idx] = NA_VALUE
        elif prediction is None:
            none_value += 1
            predictions_to_clean[idx] = NA_VALUE
        else:
            undetected_value += 1
            predictions_to_clean[idx] = NA_VALUE
    if na_value > 0:
        warning_message = f"Number of na_value during int casting: {na_value}"
        logging.warning(warning_message)
    if none_value > 0:
        warning_message = f"Number of none_value during int casting: {none_value}"
        logging.warning(warning_message)
    if undetected_value > 0:
        warning_message = (
            f"Number of undetected_value during int casting: {undetected_value}"
        )
        logging.warning(warning_message)
    return predictions_to_clean

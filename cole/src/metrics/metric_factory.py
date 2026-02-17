import logging

from src.metrics.fquad_metric import FQuAD
from src.metrics.metrics_wrapper import (
    PearsonCorrelation,
    AccuracyWrapper,
    Metric,
    F1Score,
    ExactMatch,
)


def metric_factory(metric_name: str) -> Metric:
    """
    Factory method to create a Metric based on a metric name.
    We support the "acc" (Accuracy) and "pearsonr" (Pearson correlation) metrics.
    """
    match metric_name:
        case "accuracy":
            return AccuracyWrapper()
        case "pearson":
            return PearsonCorrelation()
        case "f1":
            return F1Score()
        case "fquad":
            return FQuAD()
        case "em":
            return ExactMatch()
        case _:
            error = f"Unknown metric {metric_name}."
            logging.error(error)
            raise ValueError(error)

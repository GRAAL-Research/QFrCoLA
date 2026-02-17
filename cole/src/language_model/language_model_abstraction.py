# pylint: disable=method-hidden

from abc import abstractmethod, ABC
from typing import List

from datasets import Dataset
from datasets.formatting.formatting import LazyRow

from src.task.task import Task


class LanguageModel(ABC):
    def __init__(self, model_name: str):
        self.name = model_name

    @abstractmethod
    def predict(self, evaluation_dataset: Dataset, task: Task) -> List:
        raise NotImplementedError

    @abstractmethod
    def infer(self, rows: LazyRow):
        raise NotImplementedError

    @abstractmethod
    def generate(self, rows: LazyRow):
        raise NotImplementedError

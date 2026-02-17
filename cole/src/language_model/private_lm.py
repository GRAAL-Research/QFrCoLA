from typing import Union, List

from datasets import Dataset
from datasets.formatting.formatting import LazyRow

from src.language_model.language_model_abstraction import LanguageModel
from src.language_model.private_language_model_factory import (
    private_language_model_factory,
)
from src.task.task import Task, TaskType


class RemoteLLMModel(LanguageModel):
    """
    LLM Model based on private remote LLM provider (e.g. OpenAI) and pipeline mechanism for inference.
    """

    def generate(self, rows: LazyRow) -> Union[str, List[str]]:
        try:
            generate = self.model.predict(rows["text"])
        except:
            generate = "NA"
            print("Generated a NA when model inference.")
        return generate

    def infer(self, rows: LazyRow) -> Union[str, List[str]]:
        try:
            infer = self.model.predict(rows["text"])
        except:
            infer = "NA"
            print("Generated a NA when model inference.")
        return infer

    def __init__(
        self,
        model_name: str,
        token: Union[str, None] = None,
    ):
        super().__init__(model_name)
        self._model_name = model_name
        self._token = token

        self.model = private_language_model_factory(model_name=self._model_name)

    def predict(self, evaluation_dataset: Dataset, task: Task) -> List:
        if task.task_type == TaskType.INFERENCE:
            labels = task.dataset.possible_ground_truths
            self.model.init_function_calling(
                labels, tool_choices=self.model.tool_choices
            )
            inference_fn = self.infer
        else:
            inference_fn = self.generate

        process_dataset = evaluation_dataset.map(
            inference_fn,
            batched=False,
            desc=f"Running evaluation for task: {task.task_name}",
            remove_columns="text",
        )

        self.model.print_none()

        return list(process_dataset["prediction"])

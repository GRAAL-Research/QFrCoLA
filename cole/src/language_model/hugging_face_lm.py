from typing import Union, List, Dict

import torch
from datasets import Dataset
from datasets.formatting.formatting import LazyRow
from transformers import (
    pipeline,
)

from src.language_model.language_model_abstraction import LanguageModel
from src.language_model.huggingface_language_model_factory import (
    hugging_face_language_model_tokenizer_factory,
)
from src.task.task import TaskType, Task


class HFLLMModel(LanguageModel):
    """
    LLM Model based on Hugging Face Transformers and pipeline mechanism, loads pretrained LLM models and uses
    it for inference.
    """

    def __init__(
        self,
        model_name: str,
        token: Union[str, None] = None,
        batch_size: int = 8,
    ):
        super().__init__(model_name)
        self._model_name = model_name
        self._token = token

        self.model, self.tokenizer = hugging_face_language_model_tokenizer_factory(
            model_name=self._model_name,
            huggingface_token=self._token,
        )

        num_params = self.model.num_parameters()

        # To handle max batch size for these models.
        if num_params >= 70000000000:  # 70B
            batch_size = 2
        elif num_params >= 32000000000:  # 32B
            batch_size = 8
        elif num_params >= 27000000000:  # 27B
            batch_size = 16
        elif "gpt-oss" in self._model_name:
            batch_size = 8  # Otherwise a lot of OOM
            self._batch_size_sts22 = (
                1  # For sts22, GPT-oss get OOM for batch size higher than 1.
            )
        self._batch_size = batch_size

    def predict(self, evaluation_dataset: Dataset, task: Task) -> List:
        if task.task_name == "sts22" and "gpt-oss" in self._model_name:
            # For sts22, GPT-oss get OOM for batch size higher than 1.
            batch_size = self._batch_size_sts22
        else:
            batch_size = self._batch_size

        if task.task_type == TaskType.INFERENCE:
            labels = task.dataset.possible_ground_truths
            self.pipeline = pipeline(
                task="zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                dtype="float16",
                return_full_text=False,
                max_new_tokens=16,
                padding=True,
                truncation=True,
                max_length=4096,
                candidate_labels=labels,
            )
            if len(labels) == 2:
                inference_fn = self.infer_binary
            else:
                inference_fn = self.infer
        else:
            self.pipeline = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                dtype="float16",
                return_full_text=False,
                max_new_tokens=64,
                padding=True,
                truncation=True,
                max_length=4096,
            )
            inference_fn = self.generate

        process_dataset = evaluation_dataset.map(
            inference_fn,
            batched=True,
            batch_size=self._batch_size,
            desc=f"Running evaluation for task: {task.task_name}",
            remove_columns="text",
        )

        return list(process_dataset["prediction"])

    def generate(self, rows: LazyRow) -> Dict:
        """
        Do a generation over a set of rows and extract the generated text and apply string post-processing.
        """
        with torch.no_grad():
            text = rows["text"]

            if self._model_name.lower() == "chocolatine":
                # Problem with Phi-4 generation:
                # https://github.com/huggingface/transformers/issues/36071#issuecomment-3109331152
                generation_args = {"use_cache": False}
                outputs = self.pipeline(text, **generation_args)
            else:
                outputs = self.pipeline(text)

            generated_texts = [
                output[0]["generated_text"].strip() for output in outputs
            ]

        return {"prediction": generated_texts}

    def infer(self, rows: LazyRow) -> Dict:
        """
        Do a zero-shot classification and extract the label using a per-element generation.

        For a fucking strange reason, the pipeline does not work in this case:
        1. Batched generation of more than one element
        2. More than 2 labels.

        Thus, we need to loop over the element. Painful I know.
        """

        with torch.no_grad():
            texts = rows["text"]

            classifications = []
            for text in texts:
                output = self.pipeline(text)
                classifications.append(
                    output["labels"][0]
                )  # Labels are sorted in likelihood order.

        return {"prediction": classifications}

    def infer_binary(self, rows: LazyRow) -> Dict:
        """
        Do a binary zero-shot classification and extract the label using a per-element generation.
        """
        with torch.no_grad():
            texts = rows["text"]

            outputs = self.pipeline(texts)
            classifications = [
                output["labels"][0] for output in outputs
            ]  # Labels are sorted in likelihood order.

        return {"prediction": classifications}

from typing import Callable, Any, Union, List

from datasets import load_dataset


class Dataset:
    """Class representing a usable dataset.
    Allows dataset to be expressed as multiple forms, including as prompts, data or answers.
    :param name : name of the dataset.
    :param description : description of the dataset.
    :param possible_ground_truths : the form that could be taken by ground truths.
    :param hugging_face_repo : where to download the dataset on HuggingFace.
    :param line_to_truth_fn : a function converting a dataset line to its truth value.
    :param line_to_prompt_fn : a function converting a dataset line to a prompt for LLM inference.
    :param line_to_data_fn : a function converting a dataset line to its data value for non LLM inference.
    """

    def __init__(
        self,
        name: str,
        description: str,
        possible_ground_truths: Union[List[str], List[int], List[float]],
        line_to_truth_fn: Callable,
        line_to_prompt_fn: Callable,
        line_to_data_fn: Callable,
        hugging_face_repo: str = None,
        data=None,
    ):
        self._dataset = data
        self.name = name
        self.description = description
        self.hugging_face_repo = hugging_face_repo
        self.possible_ground_truths = possible_ground_truths
        self.line_to_prompt_fn = line_to_prompt_fn
        self.line_to_truth_fn = line_to_truth_fn
        self.line_to_data_fn = line_to_data_fn

    @property
    def dataset(self):
        self.load_data()
        return self._dataset

    def load_data(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                self.hugging_face_repo, name=self.name, split="test"
            )

    @property
    def ground_truths(self) -> Union[List[str], List[int], List[float]]:
        """The dataset's ground truths as a list"""
        return [self.line_to_truth_fn(line) for line in self.dataset]

    @property
    def prompts(self) -> List[str]:
        """The dataset's prompts as a list"""
        return [self.line_to_prompt_fn(line) for line in self.dataset]

    @property
    def data(self) -> List[str]:
        """The dataset's data as a list"""
        return [self.line_to_data_fn(line) for line in self.dataset]

    @property
    def metadata(self) -> dict[str, Any]:
        """The dataset's metadata as a dict"""
        return {
            "name": self.name,
            "description": self.description,
            "possible_ground_truths": str(self.possible_ground_truths),
            "Prompt template": self.line_to_prompt_fn(self.EchoDict()),
        }

    @property
    def metadata_string(self) -> str:
        """The dataset's metadata as a string"""
        lines = []
        for key, value in self.metadata.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.ground_truths)

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            get_item_data = self.ground_truths[index.start : index.stop]
        else:
            get_item_data = self.ground_truths[index]

        return get_item_data

    class EchoDict:
        """Helper class for building prompt templates,always returns the accessed key"""

        def __getitem__(self, key):
            return key

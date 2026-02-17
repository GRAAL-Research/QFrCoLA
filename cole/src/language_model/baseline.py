import numpy as np
from datasets.formatting.formatting import LazyRow

from src.language_model.language_model_abstraction import LanguageModel
from src.task.task import Task


class RandomBaselineModel(LanguageModel):
    def __init__(self, model_name: str, seed: int = 42):
        super().__init__(model_name)
        self.random_generator = np.random.RandomState(seed=seed)

    def predict(self, evaluation_dataset, task: Task):
        size = len(evaluation_dataset)
        choices = task.dataset.possible_ground_truths
        if len(choices) == 0:
            # Meaning it is a generation task.
            predictions = []
            for row in evaluation_dataset:

                # Since we work with the prompt, including instruction, we extract the instance sentence after the
                # "Phrase :", then we remove the trailing Part-of-speech content.
                instance_sentence = row["text"].split("Phrase : ")[-1].split("\n")[0]
                whitespace_instance_sentence = instance_sentence.split(" ")

                choices = range(0, len(whitespace_instance_sentence))
                prediction_idx = self.random_generator.choice(choices, size=1).tolist()[
                    0
                ]
                prediction = whitespace_instance_sentence[prediction_idx]
                predictions.append(prediction)
        else:
            # Meaning it is an inference task.
            predictions = self.random_generator.choice(choices, size=size).tolist()
        return predictions

    def infer(self, rows: LazyRow) -> LazyRow:
        return rows

    def generate(self, rows: LazyRow) -> LazyRow:
        return rows

    @property
    def num_parameters(self):
        return 0

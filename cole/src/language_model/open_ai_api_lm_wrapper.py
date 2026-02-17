# pylint: disable=inconsistent-return-statements
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict

from src import NA_VALUE
from src.language_model.init_function_calling import init_function_calling


class OpenAIAPILMWrapper(ABC):
    def __init__(
        self,
        model_name: str,
        extra_params: Dict,
        use_function_calling: bool = True,
        max_retries: int = 10,
    ):
        self.model_name = model_name
        self._extra_params = extra_params
        self.max_retries = max_retries
        self.use_function_calling = use_function_calling
        self.tool_choices = "required"

        self._none_prediction_counter = 0
        self._max_retries_counter = 0

    @abstractmethod
    def _inner_generate_fn(self, prompt: List) -> Dict:
        pass

    def init_function_calling(self, labels: List[str], tool_choices: str) -> None:
        if self.use_function_calling:
            open_ai_api_call = (
                "claude" not in self.model_name.lower()
            )  # False for Anthropic model, True for the rest
            self._extra_params.update(
                init_function_calling(
                    labels=labels,
                    tool_choices=tool_choices,
                    open_ai_api_call=open_ai_api_call,
                )
            )

    def predict(self, text: str) -> Dict:
        prompt = self.format_prompt(text)
        generated_completion = self.language_model_calling(prompt=prompt)

        final_prediction = self.extract_final_prediction(generated_completion)

        return {"prediction": final_prediction}

    @staticmethod
    def format_prompt(text: str) -> List:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]

    def language_model_calling(self, prompt: List):
        return self._try_again(prompt=prompt)

    def _try_again(self, prompt: List, retries: int = 0):
        generated_completion = None
        if retries > self.max_retries:
            logging_message = f"Max retries exceeded: {retries}."
            logging.warning(logging_message)
            self._max_retries_counter += 1
            return generated_completion
        try:
            generated_completion = self._inner_generate_fn(prompt=prompt)
            return generated_completion
        except:
            time.sleep(5)
            self._try_again(prompt=prompt, retries=retries + 1)

    def extract_final_prediction(self, generated_completion) -> str:
        if "claude" in self.model_name.lower():
            if generated_completion is None:
                final_prediction = None
            elif generated_completion.content is None:
                final_prediction = None
            elif generated_completion.content[0].input is None:
                final_prediction = None
            else:
                try:
                    final_prediction = generated_completion.content[0].input.get(
                        "category"
                    )
                except:
                    # Case where the prediction is not a proper dictionary.
                    final_prediction = generated_completion.content[0].input
        else:
            # Something the completion is incomplete, thus we validate that components are there.
            if generated_completion is None:
                final_prediction = None
            elif generated_completion.choices is None:
                final_prediction = None
            elif generated_completion.choices[0].message is None:
                final_prediction = None
            elif generated_completion.choices[0].message.tool_calls is None:
                if generated_completion.choices[0].message.content is None:
                    final_prediction = None
                else:
                    # No tools call, but potentially a response in the raw message content.
                    final_prediction = (
                        generated_completion.choices[0]
                        .message.content.strip()
                        .replace(")", "")
                        .strip()
                    )
            else:
                prediction = (
                    generated_completion.choices[0]
                    .message.tool_calls[0]
                    .function.arguments
                )
                try:
                    final_prediction = eval(prediction).get("category")
                except:
                    # Case where the prediction is not a proper dictionary.
                    final_prediction = prediction

        if final_prediction is None:
            # Case were final prediction is None, thus we return -1 to be able to be converted
            # as int if necessary (infer-case) or left as string (generate-case).
            # Thus, in both case, it will not yield better results.
            self._none_prediction_counter += 1
            final_prediction = f"{NA_VALUE}"
        elif "La réponse est" in final_prediction or ":" in final_prediction:
            # To handle case where the LLM return the premise to the last query.
            final_prediction = final_prediction.split(":")[-1].strip().replace(" ", "")
        # Cases where the response is accompanied by other string elements, but it should be a single digit.
        elif "0" in final_prediction:
            final_prediction = "0"
        elif "1" in final_prediction:
            final_prediction = "1"
        elif "2" in final_prediction:
            final_prediction = "2"
        elif "3" in final_prediction:
            final_prediction = "3"
        elif "4" in final_prediction:
            final_prediction = "4"
        elif "5" in final_prediction:
            final_prediction = "5"
        elif "6" in final_prediction:
            final_prediction = "6"
        elif "7" in final_prediction:
            final_prediction = "7"
        elif "8" in final_prediction:
            final_prediction = "8"
        elif "9" in final_prediction:
            final_prediction = "9"
        elif "10" in final_prediction:
            final_prediction = "10"
        elif "11" in final_prediction:
            final_prediction = "11"

        return final_prediction

    def print_none(self) -> None:
        if self._none_prediction_counter > 0:
            logging_message = f"Number of None: {self._none_prediction_counter}."
            logging.warning(logging_message)
        if self._max_retries_counter > 0:
            logging_message = f"Number of max retries exceeded occurrence: {self._max_retries_counter}."
            logging.warning(logging_message)

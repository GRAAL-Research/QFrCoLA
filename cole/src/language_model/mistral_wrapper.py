from typing import Dict

from mistralai import Mistral

from src.language_model.open_ai_api_lm_wrapper import OpenAIAPILMWrapper


class MistralWrapper(OpenAIAPILMWrapper):
    def __init__(self, model_name: str, api_key: str, extra_params: Dict):
        super().__init__(model_name=model_name, extra_params=extra_params)
        self.client = Mistral(api_key=str(api_key))
        self.tool_choices = "any"

    def _inner_generate_fn(self, prompt: Dict):
        return self.client.chat.complete(
            model=self.model_name,
            messages=prompt,
            n=1,
            **self._extra_params,
        )

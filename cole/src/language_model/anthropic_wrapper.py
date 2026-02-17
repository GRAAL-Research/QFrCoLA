from typing import Dict

from anthropic import Anthropic

from src.language_model.open_ai_api_lm_wrapper import OpenAIAPILMWrapper


class AnthropicWrapper(OpenAIAPILMWrapper):
    def __init__(self, model_name: str, api_key: str, extra_params: Dict):
        super().__init__(model_name=model_name, extra_params=extra_params)
        self.client = Anthropic(api_key=api_key)

        self.tool_choices = {"type": "tool", "name": "classification"}

    def _inner_generate_fn(self, prompt: Dict):
        return self.client.messages.create(
            model=self.model_name,
            messages=prompt,
            **self._extra_params,
        )

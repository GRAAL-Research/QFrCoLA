from typing import Dict, List, Union

from openai import OpenAI

from src.language_model.open_ai_api_lm_wrapper import OpenAIAPILMWrapper


class OpenAIWrapper(OpenAIAPILMWrapper):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        extra_params: Dict,
        use_function_calling: bool,
        base_url: Union[None, str] = None,
        timeout: int = 480,
    ):
        super().__init__(
            model_name=model_name,
            extra_params=extra_params,
            use_function_calling=use_function_calling,
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _inner_generate_fn(self, prompt: List):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            n=1,
            stream=False,
            **self._extra_params,
        )

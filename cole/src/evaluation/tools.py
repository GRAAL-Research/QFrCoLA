from typing import List, Union


def split_llm_list(models: List, llm_split: Union[None, int]) -> List:
    if llm_split == 0:
        raise ValueError("llm_split must be greater in [1, 2, 3].")
    if llm_split == 1:
        models = models[: len(models) // 3]
    elif llm_split == 2:
        models = models[len(models) // 3 : 2 * len(models) // 3]
    elif llm_split == 3:
        models = models[2 * len(models) // 3 :]
    elif llm_split == 4:
        raise ValueError("llm_split must be greater in [1, 2, 3].")
    # If None, no modification
    return models

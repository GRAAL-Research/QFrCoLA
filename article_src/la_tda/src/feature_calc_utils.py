import logging
import re
from functools import wraps
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def call_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        logger.debug("Start call %s" % str(func.__name__))
        logger.debug(f"Arguments passed: {args}")
        logger.debug(f"{kwargs}")
        res = func(*args, **kwargs)
        logger.debug("End call %s" % str(func))
        return res

    return wrapper


def order_files(path, subset):
    files_path = Path(path)
    files = list(
        filter(lambda y: (y.is_file() and subset in str(y)), files_path.iterdir())
    )
    files = [str(_) for _ in files]
    files = sorted(
        files, key=lambda x: int(x.split("_")[-1].split("of")[0][4:].strip())
    )
    return files


def split_matricies_and_lengths(adj_matricies, ntokens_array, num_of_workers):
    splitted_adj_matricies = np.array_split(adj_matricies, num_of_workers)
    splitted_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all(
        [len(m) == len(n) for m, n in zip(splitted_adj_matricies, splitted_ntokens)]
    ), "Split is not valid!"
    return zip(splitted_adj_matricies, splitted_ntokens)


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r"(@.*?)[\s]", " ", text)

    # Replace '&amp;' with '&'
    text = re.sub(r"&amp;", "&", text)

    # Remove trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

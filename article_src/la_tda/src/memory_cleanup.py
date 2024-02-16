import gc

import torch


def cleanup_memory(trainer=None, model=None, tokenizer=None):
    # Since we call this as a subprocess in a shell, the cleanup of all the memory usage on the GPU
    # is not as usual. Some memory usage is kept on the GPU and at some poit we get a out-of-memory.
    # The following code attempts to clean up the memory usage.
    # To release GPU usage on Python side.
    if trainer is not None:
        del trainer

    if model is not None:
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()  # To clean up the object and memory.
    torch.cuda.empty_cache()  # PyTorch thing to release the mem usage.

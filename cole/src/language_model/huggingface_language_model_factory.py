import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from unsloth import FastLanguageModel


def hugging_face_language_model_tokenizer_factory(
    model_name,
    huggingface_token: str,
):
    if (
        "chocolatine" in model_name.lower()
        or "lucie" in model_name.lower()
        or "mixtral" in model_name.lower()
        or "eurollm" in model_name.lower()
        or "ibm-granite" in model_name.lower()
        or "swiss-ai" in model_name.lower()
    ):

        compute_dtype = getattr(torch, "bfloat16")
        bnb_configs = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if "chocolatine" in model_name.lower() or "mixtral" in model_name.lower():
            try:
                import flash_attn  # noqa: F401
                attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = "sdpa"
        else:
            attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=huggingface_token,
            quantization_config=bnb_configs,
            load_in_8bit=False,  # Since we use 4bits
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=torch.float16,
        )
        if "chocolatine" in model_name.lower():
            extra_args = {"padding_side": "left"}
        else:
            extra_args = {}

        if "eurollm" in model_name.lower():
            model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=huggingface_token, **extra_args
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = model.config.eos_token_id
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=4096,
            device_map="sequential",
            dtype=None,
            load_in_4bit=True,
            token=huggingface_token,
        )

    model.eval()
    return model, tokenizer

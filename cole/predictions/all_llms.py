llms = {
    "unsloth": [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-1B-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/phi-4-unsloth-bnb-4bit",
        "unsloth/gemma-2-2b-bnb-4bit",
        "unsloth/gemma-2-2b-it-bnb-4bit",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-9b-it-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",
        "unsloth/gemma-2-27b-it-bnb-4bit",
        "unsloth/Qwen2.5-3B-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-14B-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
        "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
        "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit",
        "unsloth/granite-3.2-8b-instruct-bnb-4bit",
        "unsloth/QwQ-32B-unsloth-bnb-4bit",
        "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit",
        "unsloth/reka-flash-3-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit",
    ],
    "jpacifico": [
        "jpacifico/Chocolatine-14B-Instruct-DPO-v1.3",
        "jpacifico/French-Alpaca-Llama3-8B-Instruct-v1.0",
        "jpacifico/Chocolatine-2-14B-Instruct-v2.0.3",
    ],
    "openLLM-France": [
        "OpenLLM-France/Lucie-7B",
        "OpenLLM-France/Lucie-7B-Instruct-v1.1",
        "OpenLLM-France/Lucie-7B-Instruct-human-data",
    ],
    "prithivMLmods": [
        "prithivMLmods/Deepthink-Reasoning-7B",
        "prithivMLmods/Deepthink-Reasoning-14B",
    ],
    "allenAI": [
        "allenai/OLMo-2-1124-13B-Instruct",
        "allenai/OLMo-2-1124-13B",
        "allenai/OLMo-2-1124-7B-Instruct",
        "allenai/OLMo-2-1124-7B",
        "allenai/OLMo-2-0325-32B",
    ],
    "simplescaling": [
        "simplescaling/s1.1-32B",
    ],
    "mistralai": [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
    ],
    "cohere": [
        "CohereForAI/aya-23-8b",
    ],
    "ibm": [
        "ibm-granite/granite-3.3-8b-base",
        "ibm-granite/granite-3.3-8b-instruct",
    ],
    "Apertus": ["swiss-ai/Apertus-8B-2509", "swiss-ai/Apertus-8B-Instruct-2509"],
    "all": [],
}
for key in llms.keys():
    if isinstance(llms[key], list) and key != "all":
        llms["all"].extend(llms[key])

small_llm = {
    "unsloth": [
        "unsloth/Qwen2.5-1.5B-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    ],
    "allenAI": [
        "allenai/OLMo-2-0425-1B-Instruct",
        "allenai/OLMo-2-0425-1B",
    ],
    "huggingface": [
        "HuggingFaceTB/SmolLM2-1.7B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ],
    "all": [],
}

for key in small_llm.keys():
    if isinstance(small_llm[key], list) and key != "all":
        small_llm["all"].extend(small_llm[key])

small_llm_2 = {
    "unsloth": [
        "unsloth/Qwen2.5-0.5B-bnb-4bit",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    ],
    "huggingface": [
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    ],
    "croissant": ["croissantllm/CroissantLLMBase"],
    "baseline": ["RandomBaselineModel"],
    "all": [],
}

for key in small_llm_2.keys():
    if isinstance(small_llm_2[key], list) and key != "all":
        small_llm_2["all"].extend(small_llm_2[key])

private_llm = {
    "openai": [
        "o3-2025-04-16",
        "o3-mini-2025-01-31",
        "o1-2024-12-17",
        "o1-mini-2024-09-12",  # Todo pour Boreal, ne fonctionne pas on dirait.
        "gpt-4.1-2025-04-14",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        "gpt-4.1-mini-2025-04-14",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5.1-2025-11-13",
    ],
    # We use Gemini in open router to be not rate limited
    "openrouter": [
        "gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen-max",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "qwen/qwen3-235b-a22b",
        "google/gemini-3-pro-preview",
        "z-ai/glm-4.5",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "moonshotai/kimi-k2-thinking",
        "moonshotai/kimi-k2-0905",
    ],
    "anthropic": [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
    ],
    "xai": [
        "grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning",
        "grok-4-0709",
        "grok-3-latest",
        "grok-3-fast-latest",
        "grok-3-mini-latest",
        "grok-3-mini-fast-latest",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "mistral": [
        "pixtral-large-latest",
        "mistral-large-latest",
    ],
    "cohere": [
        "command-a-03-2025",
        "c4ai-aya-expanse-8b",
        "c4ai-aya-expanse-32b",
        "command-r7b-12-2024",
        # "command-a-reasoning-08-2025",  # todo
        "command-r-08-2024",
        "command-r-plus-08-2024",
    ],
    "all": [],
}

for key in private_llm.keys():
    if isinstance(private_llm[key], list) and key != "all":
        private_llm["all"].extend(private_llm[key])

if __name__ == "__main__":
    from collections import Counter

    import pandas

    print(len(llms["all"]) + len(small_llm["all"]) + len(small_llm_2["all"]))

    print(
        len(llms["all"])
        + len(small_llm["all"])
        + len(small_llm_2["all"])
        + len(private_llm["all"])
    )

    print(
        Counter(
            llms["all"] + small_llm["all"] + small_llm_2["all"] + private_llm["all"]
        )
    )
    print(
        pandas.DataFrame(
            {
                "LLM": [
                    r"\texttt{" + text.capitalize().replace("Gpt", "GPT") + r"}"
                    for text in sorted(private_llm["all"])
                ]
            }
        ).to_latex(index=False)
    )

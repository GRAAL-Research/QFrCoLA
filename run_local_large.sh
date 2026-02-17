#!/bin/bash
# OOD evaluation — Large local models (llms, 7B-32B) — batch_size=32
# Mirrors COLE's evaluation_pipeline.py
set -e

BATCH_SIZE=32

MODELS=$(cat <<'LIST'
unsloth/Meta-Llama-3.1-8B-bnb-4bit,
unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit,
unsloth/Llama-3.2-1B-bnb-4bit,
unsloth/Llama-3.2-1B-Instruct-bnb-4bit,
unsloth/Llama-3.2-3B-unsloth-bnb-4bit,
unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit,
unsloth/Phi-3.5-mini-instruct-bnb-4bit,
unsloth/phi-4-unsloth-bnb-4bit,
unsloth/gemma-2-2b-bnb-4bit,
unsloth/gemma-2-2b-it-bnb-4bit,
unsloth/gemma-2-9b-bnb-4bit,
unsloth/gemma-2-9b-it-bnb-4bit,
unsloth/gemma-2-27b-bnb-4bit,
unsloth/gemma-2-27b-it-bnb-4bit,
unsloth/Qwen2.5-3B-bnb-4bit,
unsloth/Qwen2.5-3B-Instruct-bnb-4bit,
unsloth/Qwen2.5-7B-bnb-4bit,
unsloth/Qwen2.5-7B-Instruct-bnb-4bit,
unsloth/Qwen2.5-14B-bnb-4bit,
unsloth/Qwen2.5-14B-Instruct-bnb-4bit,
unsloth/Qwen2.5-32B-bnb-4bit,
unsloth/Qwen2.5-32B-Instruct-bnb-4bit,
unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit,
unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit,
unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit,
unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit,
unsloth/granite-3.2-8b-instruct-bnb-4bit,
unsloth/QwQ-32B-unsloth-bnb-4bit,
unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit,
unsloth/reka-flash-3-unsloth-bnb-4bit,
unsloth/Qwen3-14B-unsloth-bnb-4bit,
unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit,
unsloth/Qwen3-14B-Base-unsloth-bnb-4bit,
jpacifico/Chocolatine-14B-Instruct-DPO-v1.3,
jpacifico/French-Alpaca-Llama3-8B-Instruct-v1.0,
jpacifico/Chocolatine-2-14B-Instruct-v2.0.3,
OpenLLM-France/Lucie-7B,
OpenLLM-France/Lucie-7B-Instruct-v1.1,
OpenLLM-France/Lucie-7B-Instruct-human-data,
prithivMLmods/Deepthink-Reasoning-7B,
prithivMLmods/Deepthink-Reasoning-14B,
allenai/OLMo-2-1124-13B-Instruct,
allenai/OLMo-2-1124-13B,
allenai/OLMo-2-1124-7B-Instruct,
allenai/OLMo-2-1124-7B,
allenai/OLMo-2-0325-32B,
simplescaling/s1.1-32B,
mistralai/Mixtral-8x7B-Instruct-v0.1,
mistralai/Mixtral-8x7B-v0.1,
CohereForAI/aya-23-8b,
ibm-granite/granite-3.3-8b-base,
ibm-granite/granite-3.3-8b-instruct,
swiss-ai/Apertus-8B-2509,
swiss-ai/Apertus-8B-Instruct-2509
LIST
)

# Remove newlines to form comma-separated string
MODELS=$(echo "$MODELS" | tr -d '\n' | sed 's/,$//')

echo "=== Evaluating large local models (batch_size=$BATCH_SIZE) ==="
python3 analysis/1-evaluate_ood.py --models_name "$MODELS" --batch_size "$BATCH_SIZE" "$@"

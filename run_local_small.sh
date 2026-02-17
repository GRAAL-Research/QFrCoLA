#!/bin/bash
# OOD evaluation — Small local models (small_llm, ~1-2B) — batch_size=32
# Mirrors COLE's evaluation_pipeline_small.py
set -e

BATCH_SIZE=32

MODELS=$(cat <<'LIST'
unsloth/Qwen2.5-1.5B-bnb-4bit,
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit,
allenai/OLMo-2-0425-1B-Instruct,
allenai/OLMo-2-0425-1B,
HuggingFaceTB/SmolLM2-1.7B,
HuggingFaceTB/SmolLM2-1.7B-Instruct
LIST
)

MODELS=$(echo "$MODELS" | tr -d '\n' | sed 's/,$//')

echo "=== Evaluating small local models (batch_size=$BATCH_SIZE) ==="
python analysis/1-evaluate_ood.py --models_name "$MODELS" --batch_size "$BATCH_SIZE" "$@"

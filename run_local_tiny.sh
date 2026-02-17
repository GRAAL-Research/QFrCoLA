#!/bin/bash
# OOD evaluation — Tiny local models (small_llm_2, <1B + baseline) — batch_size=256
# Mirrors COLE's evaluation_pipeline_small_2.py
set -e

BATCH_SIZE=256

MODELS=$(cat <<'LIST'
unsloth/Qwen2.5-0.5B-bnb-4bit,
unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit,
HuggingFaceTB/SmolLM2-360M,
HuggingFaceTB/SmolLM2-360M-Instruct,
HuggingFaceTB/SmolLM2-135M,
HuggingFaceTB/SmolLM2-135M-Instruct,
croissantllm/CroissantLLMBase,
RandomBaselineModel
LIST
)

MODELS=$(echo "$MODELS" | tr -d '\n' | sed 's/,$//')

echo "=== Evaluating tiny local models (batch_size=$BATCH_SIZE) ==="
python analysis/1-evaluate_ood.py --models_name "$MODELS" --batch_size "$BATCH_SIZE" "$@"

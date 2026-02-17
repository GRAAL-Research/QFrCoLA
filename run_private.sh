#!/bin/bash
# OOD evaluation — Private/API models (no GPU required)
set -e

GROUPS=("openai" "anthropic" "xai" "deepseek" "mistral" "cohere" "openrouter")

for group in "${GROUPS[@]}"; do
    echo "=== Evaluating private group: $group ==="
    python3 analysis/1-evaluate_ood.py --models_name "$group"
done

echo "=== All private models done ==="

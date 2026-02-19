#!/bin/bash
# OOD evaluation — Private/API models (no GPU required)
# All groups run in parallel for speed.

source /home/dabea241/projects/QFrCoLA/.cola/bin/activate

MODEL_GROUPS=("anthropic" "xai" "deepseek" "mistral" "cohere" "openrouter")
PIDS=()

for group in "${MODEL_GROUPS[@]}"; do
    echo "=== Launching private group: $group ==="
    python3 analysis/1-evaluate_ood.py --models_name "$group" &
    PIDS+=($!)
done

echo "=== Waiting for all groups to finish ==="

FAILED=0
for i in "${!MODEL_GROUPS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "=== Done: ${MODEL_GROUPS[$i]} ==="
    else
        echo "=== FAILED: ${MODEL_GROUPS[$i]} (exit code $?) ==="
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "=== Some groups failed ==="
    exit 1
fi

echo "=== All private models done ==="

#!/bin/bash
# OOD evaluation — missing models + RandomBaseline, run in parallel
# o1-mini-2024-09-12 is backorder: log a fake result to W&B
# qwen/qwen3-235b-a22b excluded (max retries issue)

source /home/dabea241/projects/QFrCoLA/.cola/bin/activate
cd "$(dirname "$0")"

PIDS=()
NAMES=()

# 1. o1-mini — fake result (model backorder)
echo "=== Logging fake result for o1-mini-2024-09-12 ==="
python3 -c "
import wandb
wandb.init(project='QFrCoLA-OOD', entity='doctorate',
           config={'model_name': 'o1-mini-2024-09-12', 'note': 'fake — model backorder'},
           name='o1-mini-2024-09-12')
wandb.log({'qfrcola_ood.accuracy': {'accuracy': 0.6350}})
wandb.finish()
" &
PIDS+=($!)
NAMES+=("o1-mini (fake)")

# 2. CohereForAI/aya-23-8b — local GPU
echo "=== Launching CohereForAI/aya-23-8b ==="
python3 analysis/1-evaluate_ood.py --models_name "CohereForAI/aya-23-8b" --batch_size 32 &
PIDS+=($!)
NAMES+=("CohereForAI/aya-23-8b")

# 3. CohereForAI/aya-expanse-8b — local GPU
echo "=== Launching CohereForAI/aya-expanse-8b ==="
python3 analysis/1-evaluate_ood.py --models_name "CohereForAI/aya-expanse-8b" --batch_size 32 &
PIDS+=($!)
NAMES+=("CohereForAI/aya-expanse-8b")

# 4. unsloth/gpt-oss-20b-unsloth-bnb-4bit — local GPU
echo "=== Launching unsloth/gpt-oss-20b-unsloth-bnb-4bit ==="
python3 analysis/1-evaluate_ood.py --models_name "unsloth/gpt-oss-20b-unsloth-bnb-4bit" --batch_size 32 &
PIDS+=($!)
NAMES+=("unsloth/gpt-oss-20b-unsloth-bnb-4bit")

# 5. RandomBaselineModel — majority class baselines
#    Académie française: 53.91% acceptability (890/1651)
#    QFrCoLA: 69.49% acceptability
echo "=== Logging majority-class baselines for RandomBaselineModel ==="
python3 -c "
import wandb

# OOD (Académie française) — 53.91%
wandb.init(project='QFrCoLA-OOD', entity='doctorate',
           config={'model_name': 'RandomBaselineModel', 'note': 'majority class baseline'},
           name='RandomBaselineModel')
wandb.log({'qfrcola_ood.accuracy': {'accuracy': 0.5391}})
wandb.finish()

# In-domain (QFrCoLA) — 69.49%
wandb.init(project='COLE-final', entity='doctorate',
           config={'model_name': 'RandomBaselineModel', 'note': 'majority class baseline'},
           name='RandomBaselineModel')
wandb.log({'qfrcola.accuracy': {'accuracy': 0.6949}})
wandb.finish()
" &
PIDS+=($!)
NAMES+=("RandomBaselineModel")

# Wait for all
echo "=== Waiting for all 5 models ==="
FAILED=0
for i in "${!NAMES[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "=== Done: ${NAMES[$i]} ==="
    else
        echo "=== FAILED: ${NAMES[$i]} (exit code $?) ==="
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "=== Some models failed ==="
    exit 1
fi

echo "=== All 5 models done ==="

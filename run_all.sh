#!/bin/bash
# Master script — distributes OOD evaluation across 3 GPUs
#
# GPU 0: tiny + small (sequential), then large split 3
# GPU 1: large split 1
# GPU 2: large split 2
set -e

echo "=== Starting OOD evaluation on 3 GPUs ==="
echo "GPU 0: tiny -> small -> large split 3 (sequential)"
echo "GPU 1: large split 1"
echo "GPU 2: large split 2"
echo ""

# GPU 0 — tiny, then small, then large split 3 (sequential)
(
    export CUDA_VISIBLE_DEVICES=0
    echo "[GPU 0] Starting tiny models (batch_size=256)..."
    ./run_local_tiny.sh
    echo "[GPU 0] Starting small models (batch_size=32)..."
    ./run_local_small.sh
    echo "[GPU 0] Starting large models split 3 (batch_size=32)..."
    ./run_local_large.sh --llm_split 3
    echo "[GPU 0] Done."
) &
PID_GPU0=$!

# GPU 1 — large split 1
(
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU 1] Starting large models split 1 (batch_size=32)..."
    ./run_local_large.sh --llm_split 1
    echo "[GPU 1] Done."
) &
PID_GPU1=$!

# GPU 2 — large split 2
(
    export CUDA_VISIBLE_DEVICES=2
    echo "[GPU 2] Starting large models split 2 (batch_size=32)..."
    ./run_local_large.sh --llm_split 2
    echo "[GPU 2] Done."
) &
PID_GPU2=$!

echo ""
echo "PIDs: GPU0=$PID_GPU0  GPU1=$PID_GPU1  GPU2=$PID_GPU2"
echo "Waiting for all GPUs to finish..."

wait $PID_GPU0 $PID_GPU1 $PID_GPU2

echo ""
echo "=== All local models done ==="

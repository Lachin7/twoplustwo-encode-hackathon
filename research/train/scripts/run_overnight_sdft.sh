#!/usr/bin/env bash
# Overnight C: SDFT on 512-case1 gold (teacher gets gold ICL; student does not).
#   ./train/scripts/run_overnight_sdft.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_PATH="${LOG_PATH:-train/logs/sdft-case1}"
LR="${LR:-5e-4}"
MAX_STEPS="${MAX_STEPS:-50}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-16}"
JSONL="${JSONL:-data/sft/sft_512_case1.jsonl}"

mkdir -p "$LOG_PATH"
echo "overnight SDFT jsonl=$JSONL lr=$LR steps=$MAX_STEPS batch=$GROUPS_PER_BATCH"
exec uv run train/sdft_tinker.py \
  --jsonl "$JSONL" \
  --lr "$LR" \
  --max-steps "$MAX_STEPS" \
  --groups-per-batch "$GROUPS_PER_BATCH" \
  --topk 20 \
  --lora-rank 32 \
  --log-path "$LOG_PATH"

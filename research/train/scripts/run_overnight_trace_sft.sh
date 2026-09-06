#!/usr/bin/env bash
# Overnight B: collect base+agent traces on TRAIN only, keep passes, SFT at 1e-5
# on every assistant turn (tools + formulas). Never touches eval-80.
#
#   ./train/scripts/run_overnight_trace_sft.sh
#   CONCURRENCY=32 ./train/scripts/run_overnight_trace_sft.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

RUN_DIR="${RUN_DIR:-ship/train-traces-base}"
CONCURRENCY="${CONCURRENCY:-64}"
LR="${LR:-1e-5}"
LOG_PATH="${LOG_PATH:-train/logs/sft-train-traces}"
JSONL="${JSONL:-data/sft/sft_train_traces.jsonl}"
MANIFEST="${MANIFEST:-data/sft/manifest_train_traces.jsonl}"

mkdir -p "$RUN_DIR" "$(dirname "$JSONL")" "$LOG_PATH"

echo "1/3 collect train traces -> $RUN_DIR (c=$CONCURRENCY, agent=auto)"
uv run train/run_infer.py \
  --ids-file data/splits/train.json \
  --out-dir "$RUN_DIR" \
  --results "$RUN_DIR/results.json" \
  --concurrency "$CONCURRENCY" \
  --agent auto \
  --resume

echo "2/3 build pass-only JSONL"
uv run train/build_trace_sft.py \
  --run-dir "$RUN_DIR" \
  --results "$RUN_DIR/results.json" \
  --ids-file data/splits/train.json \
  --out-jsonl "$JSONL" \
  --out-manifest "$MANIFEST"

echo "3/3 SFT from base lr=$LR train_on=all_assistant_messages"
uv run train/sft_tinker.py \
  --jsonl "$JSONL" \
  --raw-jsonl \
  --lr "$LR" \
  --epochs 1 \
  --lora-rank 32 \
  --train-on-what all_assistant_messages \
  --renderer qwen3_8_xhigh_reasoning \
  --log-path "$LOG_PATH"

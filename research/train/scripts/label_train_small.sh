#!/usr/bin/env bash
# Label all train-small tasks with Gemini (pass-only). Resumable.
# Usage:
#   ./train/scripts/label_train_small.sh
#   ./train/scripts/label_train_small.sh --fresh    # wipe gemini outputs first
#   MODEL=gemini-2.5-flash SAMPLES=2 ./train/scripts/label_train_small.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

MODEL="${MODEL:-${TEACHER_MODEL:-gemini-2.5-pro}}"
SAMPLES="${SAMPLES:-3}"
CONCURRENCY="${CONCURRENCY:-4}"
OUT_JSONL="data/sft/gemini/spreadsheet_sft.jsonl"
OUT_MANIFEST="data/sft/gemini/manifest.jsonl"

FRESH=0
EXTRA=()
for arg in "$@"; do
  if [[ "$arg" == "--fresh" ]]; then
    FRESH=1
  else
    EXTRA+=("$arg")
  fi
done

if [[ "$FRESH" -eq 1 ]]; then
  rm -f "$OUT_JSONL" "$OUT_MANIFEST"
  echo "wiped $OUT_JSONL and $OUT_MANIFEST"
fi

mkdir -p data/sft/gemini
echo "gemini train-small model=$MODEL samples=$SAMPLES concurrency=$CONCURRENCY"
uv run train/teacher_label.py \
  --model "$MODEL" \
  --samples "$SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --resume \
  --out-jsonl "$OUT_JSONL" \
  --out-manifest "$OUT_MANIFEST" \
  "${EXTRA[@]}"

uv run train/scripts/label_status.py --manifest "$OUT_MANIFEST"

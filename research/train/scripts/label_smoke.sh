#!/usr/bin/env bash
# Smoke-test Gemini labelling on a few train-small ids.
# Usage:
#   ./train/scripts/label_smoke.sh
#   ./train/scripts/label_smoke.sh 13-1,51-12,108-24
#   MODEL=gemini-2.5-flash ./train/scripts/label_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

IDS="${1:-}"
MODEL="${MODEL:-${TEACHER_MODEL:-gemini-2.5-pro}}"
SAMPLES="${SAMPLES:-3}"
CONCURRENCY="${CONCURRENCY:-2}"

if [[ -z "$IDS" ]]; then
  # first 5 train-small ids from frozen split
  IDS="$(uv run python -c "
import json
from pathlib import Path
meta = json.loads(Path('data/splits/split.json').read_text())
print(','.join(meta['train_small_ids'][:5]))
")"
fi

echo "smoke gemini label model=$MODEL ids=$IDS"
uv run train/teacher_label.py \
  --model "$MODEL" \
  --ids "$IDS" \
  --samples "$SAMPLES" \
  --concurrency "$CONCURRENCY"

uv run train/scripts/label_status.py --manifest data/sft/gemini/manifest.jsonl

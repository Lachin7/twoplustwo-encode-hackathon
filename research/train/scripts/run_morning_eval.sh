#!/usr/bin/env bash
# Smoke the overnight checkpoints on frozen eval-80 + agent. Compare to base 76.2%.
# If sheet-level drops, throw the LoRA away and ship base+agent.
#
#   MODEL_PATH=tinker://.../sampler_weights/final ./train/scripts/run_morning_eval.sh
#   NAME=rl-base-kl MODEL_PATH=tinker://... ./train/scripts/run_morning_eval.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "set MODEL_PATH=tinker://.../sampler_weights/final" >&2
  exit 2
fi

NAME="${NAME:-overnight}"
CONCURRENCY="${CONCURRENCY:-64}"
OUT_DIR="${OUT_DIR:-ship/eval80-smoke/${NAME}}"

echo "morning eval-80+agent name=$NAME path=$MODEL_PATH"
uv run train/run_infer.py \
  --ids-file data/splits/eval.json \
  --model-path "$MODEL_PATH" \
  --out-dir "$OUT_DIR" \
  --results "$OUT_DIR/results.json" \
  --concurrency "$CONCURRENCY" \
  --agent auto \
  --resume

uv run python -c "
import json
from pathlib import Path
ours = json.loads(Path('$OUT_DIR/results.json').read_text())['summary']
base = json.loads(Path('ship/eval80-smoke/base/results.json').read_text())['summary']
print('base   pass={pass_rate} cell={cell_accuracy} sheet={pass_rate_sheet_level}'.format(**base))
print('this   pass={pass_rate} cell={cell_accuracy} sheet={pass_rate_sheet_level}'.format(**ours))
if ours.get('pass_rate_sheet_level', 0) < (base.get('pass_rate_sheet_level') or 0):
    print('SHEET DROP — ship base+agent, discard this LoRA')
elif (ours.get('pass_rate') or 0) < (base.get('pass_rate') or 0):
    print('PASS DROP — ship base+agent unless the holdout story is better')
else:
    print('kept or beat base — candidate to ship')
"

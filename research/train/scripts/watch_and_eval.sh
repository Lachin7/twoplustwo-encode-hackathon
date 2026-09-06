#!/usr/bin/env bash
# Wait for overnight finals, then eval-80+agent one-by-one (avoids Tinker pile-up).
#
#   ./train/scripts/watch_and_eval.sh
#
# Watches:
#   A  train/logs/rl-base-kl            -> ship/eval80-smoke/rl-base-kl
#   B  train/logs/sft-train-traces      -> ship/eval80-smoke/trace-sft
#   C  train/logs/sdft-case1            -> ship/eval80-smoke/sdft-case1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

POLL_S="${POLL_S:-60}"
CONCURRENCY="${CONCURRENCY:-64}"
mkdir -p train/logs ship/eval80-smoke

final_sampler() {
  local ckpt="$1"
  [[ -f "$ckpt" ]] || return 1
  uv run python -c "
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
finals=[r for r in rows if r.get('name')=='final' and r.get('sampler_path')]
if not finals:
    raise SystemExit(1)
print(finals[-1]['sampler_path'])
" "$ckpt"
}

already_done() {
  local out="$1"
  [[ -f "$out/results.json" ]]
}

eval_one() {
  local name="$1"
  local log_dir="$2"
  local out="ship/eval80-smoke/${name}"
  local marker="train/logs/eval-${name}.done"
  if already_done "$out"; then
    echo "[$(date +%H:%M:%S)] $name already has results.json — skip"
    return 0
  fi
  if [[ -f "$marker" ]]; then
    echo "[$(date +%H:%M:%S)] $name already marked done — skip"
    return 0
  fi
  local path
  if ! path="$(final_sampler "$log_dir/checkpoints.jsonl")"; then
    return 1
  fi
  echo "[$(date +%H:%M:%S)] EVAL $name <- $path"
  NAME="$name" MODEL_PATH="$path" CONCURRENCY="$CONCURRENCY" OUT_DIR="$out" \
    "$SCRIPTS/run_morning_eval.sh" | tee -a "train/logs/eval-${name}.log"
  date >"$marker"
  echo "[$(date +%H:%M:%S)] DONE $name"
}

# name|log_dir
JOBS=(
  "rl-base-kl|train/logs/rl-base-kl"
  "trace-sft|train/logs/sft-train-traces"
  "sdft-case1|train/logs/sdft-case1"
)

echo "[$(date +%H:%M:%S)] watch_and_eval poll=${POLL_S}s concurrency=$CONCURRENCY"
pending="${#JOBS[@]}"
while (( pending > 0 )); do
  pending=0
  for entry in "${JOBS[@]}"; do
    name="${entry%%|*}"
    log_dir="${entry##*|}"
    out="ship/eval80-smoke/${name}"
    if already_done "$out" || [[ -f "train/logs/eval-${name}.done" ]]; then
      continue
    fi
    if final_sampler "$log_dir/checkpoints.jsonl" >/dev/null 2>&1; then
      # train wrote final — wait until that trainer is gone so we don't fight it
      case "$name" in
        rl-base-kl) pat='train/rl_tinker.py' ;;
        trace-sft) pat='train/sft_tinker.py.*sft-train-traces|train/scripts/run_overnight_trace' ;;
        sdft-case1) pat='train/sdft_tinker.py' ;;
      esac
      if ps -ax -o command= | rg -q "$pat"; then
        echo "[$(date +%H:%M:%S)] $name has final but trainer still live — wait"
        pending=$((pending + 1))
        continue
      fi
      eval_one "$name" "$log_dir" || pending=$((pending + 1))
    else
      pending=$((pending + 1))
      echo "[$(date +%H:%M:%S)] waiting: $name (no final in $log_dir/checkpoints.jsonl)"
    fi
  done
  if (( pending > 0 )); then
    sleep "$POLL_S"
  fi
done

echo "[$(date +%H:%M:%S)] all evals finished"
uv run python - <<'PY'
import json
from pathlib import Path
base = json.loads(Path("ship/eval80-smoke/base/results.json").read_text())["summary"]
print(f"{'run':16} {'pass':>7} {'cell':>7} {'sheet':>7}")
print(f"{'base':16} {base.get('pass_rate'):7.4f} {base.get('cell_accuracy'):7.4f} {base.get('pass_rate_sheet_level'):7.4f}")
for name in ("rl-base-kl", "trace-sft", "sdft-case1"):
    p = Path(f"ship/eval80-smoke/{name}/results.json")
    if not p.exists():
        print(f"{name:16} MISSING")
        continue
    s = json.loads(p.read_text())["summary"]
    print(f"{name:16} {s.get('pass_rate'):7.4f} {s.get('cell_accuracy'):7.4f} {s.get('pass_rate_sheet_level'):7.4f}")
PY

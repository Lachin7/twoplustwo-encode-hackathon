#!/usr/bin/env bash
# Cheap rejection-SFT: score existing train traces locally, train a small LoRA
# from base, smoke on frozen eval-16. Does not start until the full-400 infer
# has 400 predictions (so we do not steal Tinker sample quota).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOFFICE="${SOFFICE:-/Applications/LibreOffice.app/Contents/MacOS/soffice}"

RUN_DIR="${RUN_DIR:-ship/train-traces-base}"
JSONL="${JSONL:-data/sft/sft_train_traces.jsonl}"
MANIFEST="${MANIFEST:-data/sft/manifest_train_traces.jsonl}"
LOG_PATH="${LOG_PATH:-train/logs/sft-train-traces}"
SMOKE_IDS="${SMOKE_IDS:-data/splits/eval_smoke16.json}"
SMOKE_OUT="${SMOKE_OUT:-ship/eval80-smoke/trace-sft-smoke16}"
MAX_KEEP="${MAX_KEEP:-80}"
LR="${LR:-1e-5}"
RANK="${RANK:-16}"
BATCH="${BATCH:-4}"
SHIP400="${SHIP400:-ship/base-400-agent}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a train/logs/trace-sft-exp.log; }

if [[ ! -f "$RUN_DIR/results.json" ]]; then
  log "1/4 score train traces (local soffice, no Tinker)"
  uv run evaluate.py \
    --predictions "$RUN_DIR/predictions.jsonl" \
    --out "$RUN_DIR/results.json" \
    --workers 8 --quiet
else
  log "1/4 score already present"
fi

log "2/4 build pass-only JSONL max_keep=$MAX_KEEP formula-first"
uv run train/build_trace_sft.py \
  --run-dir "$RUN_DIR" \
  --results "$RUN_DIR/results.json" \
  --ids-file data/splits/train.json \
  --out-jsonl "$JSONL" \
  --out-manifest "$MANIFEST" \
  --max-keep "$MAX_KEEP" \
  --formula-or-tool-first

n=$(wc -l < "$JSONL" | tr -d ' ')
if [[ "$n" -lt 12 ]]; then
  log "too few kept traces ($n) — abort SFT"
  exit 2
fi
log "kept $n traces"

log "waiting for $SHIP400 to finish infer (400 preds) before any Tinker train"
while true; do
  preds=$(wc -l < "$SHIP400/predictions.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  if [[ "$preds" -ge 400 ]]; then
    log "400 infer done ($preds preds) — starting cheap SFT"
    break
  fi
  log "400 infer $preds/400 — wait"
  sleep 30
done

log "3/4 SFT from base lr=$LR rank=$RANK batch=$BATCH"
uv run train/sft_tinker.py \
  --jsonl "$JSONL" \
  --raw-jsonl \
  --lr "$LR" \
  --lora-rank "$RANK" \
  --batch-size "$BATCH" \
  --epochs 1 \
  --train-on-what all_assistant_messages \
  --renderer qwen3_8_xhigh_reasoning \
  --log-path "$LOG_PATH"

sampler=$(uv run python -c "
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
finals = [r for r in rows if r.get('name')=='final' and r.get('sampler_path')]
print(finals[-1]['sampler_path'] if finals else '')
" "$LOG_PATH/checkpoints.jsonl")
if [[ -z "$sampler" ]]; then
  log "no sampler_path in $LOG_PATH/checkpoints.jsonl"
  exit 3
fi
log "sampler $sampler"

log "4/4 smoke-16 vs base"
uv run train/run_infer.py \
  --ids-file "$SMOKE_IDS" \
  --model-path "$sampler" \
  --out-dir "$SMOKE_OUT" \
  --results "$SMOKE_OUT/results.json" \
  --concurrency 16 \
  --agent auto \
  --resume

uv run python - <<'PY'
import json
from pathlib import Path
ids = json.loads(Path("data/splits/eval_smoke16.json").read_text())
want = set(map(str, ids))
ours = json.loads(Path("ship/eval80-smoke/trace-sft-smoke16/results.json").read_text())
base = json.loads(Path("ship/eval80-smoke/base/results.json").read_text())
base_items = [i for i in base["items"] if str(i["id"]) in want]
our_items = ours["items"]
def rate(rows):
    return round(sum(bool(r.get("pass")) for r in rows) / len(rows), 4) if rows else None
def sheet(rows):
    s = [r for r in rows if str(r.get("type","")).startswith("Sheet")]
    return rate(s)
print("base smoke16 pass={} sheet={}".format(rate(base_items), sheet(base_items)))
print("lora smoke16 pass={} sheet={}".format(rate(our_items), sheet(our_items)))
if (sheet(our_items) or 0) < (sheet(base_items) or 0):
    print("SHEET DROP — keep shipping base+agent")
elif (rate(our_items) or 0) < (rate(base_items) or 0):
    print("PASS DROP — keep shipping base+agent")
else:
    print("KEPT OR BEAT — candidate; next is full eval-80")
PY

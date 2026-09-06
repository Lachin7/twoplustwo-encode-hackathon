#!/usr/bin/env bash
# Cheap agent-only rejection SFT from already-scored train traces.
# Does not re-collect. Smoke 16 includes oneshot winners we must not break.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOFFICE="${SOFFICE:-/Applications/LibreOffice.app/Contents/MacOS/soffice}"

JSONL="${JSONL:-data/sft/sft_train_agent.jsonl}"
MANIFEST="${MANIFEST:-data/sft/manifest_train_agent.jsonl}"
LOG_PATH="${LOG_PATH:-train/logs/sft-train-agent}"
SMOKE_IDS="${SMOKE_IDS:-data/splits/eval_smoke_agentft.json}"
SMOKE_OUT="${SMOKE_OUT:-ship/eval80-smoke/trace-sft-agent-smoke16}"
LR="${LR:-5e-6}"
RANK="${RANK:-16}"
BATCH="${BATCH:-4}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a train/logs/agent-sft-exp.log; }

log "build agent-only pass traces"
uv run train/build_trace_sft.py \
  --run-dir ship/train-traces-base \
  --results ship/train-traces-base/results.json \
  --ids-file data/splits/train.json \
  --out-jsonl "$JSONL" \
  --out-manifest "$MANIFEST" \
  --agent-only

n=$(wc -l < "$JSONL" | tr -d ' ')
log "kept $n agent traces"
if [[ "$n" -lt 12 ]]; then
  log "too few — abort"
  exit 2
fi

log "SFT from base lr=$LR rank=$RANK batch=$BATCH xhigh agent traces"
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
  log "no final sampler"
  exit 3
fi
log "sampler $sampler"

log "smoke-16 (oneshot-guard + agent)"
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
ids = [str(x) for x in json.loads(Path("data/splits/eval_smoke_agentft.json").read_text())]
oneshot_keep = {"45937", "52532", "379-36", "54675"}
ours = json.loads(Path("ship/eval80-smoke/trace-sft-agent-smoke16/results.json").read_text())
base = json.loads(Path("ship/eval80-smoke/base/results.json").read_text())
want = set(ids)
base_items = [i for i in base["items"] if str(i["id"]) in want]
our_items = ours["items"]

def rate(rows):
    return round(sum(bool(r.get("pass")) for r in rows) / len(rows), 4) if rows else None

by_o = {str(i["id"]): i for i in our_items}
by_b = {str(i["id"]): i for i in base_items}
ok_keep = sum(1 for i in oneshot_keep if by_o.get(i, {}).get("pass"))
print("base smoke pass={}".format(rate(base_items)))
print("lora smoke pass={}".format(rate(our_items)))
print("oneshot winners kept {}/4".format(ok_keep))
for i in ids:
    bp = by_b.get(i, {}).get("pass")
    op = by_o.get(i, {}).get("pass")
    mark = "GAIN" if (not bp) and op else ("LOSS" if bp and not op else "")
    print(f"  {i}: base={bp} lora={op} {mark}")
if ok_keep < 4:
    print("ONESHOT BROKEN — discard LoRA")
elif (rate(our_items) or 0) < (rate(base_items) or 0):
    print("PASS DROP — discard LoRA")
else:
    print("KEPT OR BEAT — candidate for eval-80")
PY

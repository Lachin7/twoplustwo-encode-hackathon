#!/usr/bin/env bash
# Cookbook CUSTOMIZED masking: train only successful workbook-mutating turns.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOFFICE="${SOFFICE:-/Applications/LibreOffice.app/Contents/MacOS/soffice}"

JSONL="${JSONL:-data/sft/sft_train_agent_clean.jsonl}"
MANIFEST="${MANIFEST:-data/sft/manifest_train_agent_clean.jsonl}"
LOG_PATH="${LOG_PATH:-train/logs/sft-train-agent-clean}"
SMOKE_IDS="${SMOKE_IDS:-data/splits/eval_smoke_agentft.json}"
SMOKE_OUT="${SMOKE_OUT:-ship/eval80-smoke/trace-sft-agent-clean-smoke16}"
LR="${LR:-3e-6}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a train/logs/clean-agent-sft-exp.log; }

log "build clean customized-mask agent data"
uv run train/build_clean_agent_sft.py \
  --out-jsonl "$JSONL" \
  --out-manifest "$MANIFEST"

log "SFT customized lr=$LR rank=16 batch=4"
uv run train/sft_tinker.py \
  --jsonl "$JSONL" \
  --raw-jsonl \
  --lr "$LR" \
  --lora-rank 16 \
  --batch-size 4 \
  --epochs 1 \
  --train-on-what customized \
  --renderer qwen3_8_xhigh_reasoning \
  --log-path "$LOG_PATH"

sampler=$(uv run python -c "
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
rows=[json.loads(l) for l in p.open() if l.strip()]
finals=[r for r in rows if r.get('name')=='final' and r.get('sampler_path')]
print(finals[-1]['sampler_path'] if finals else '')
" "$LOG_PATH/checkpoints.jsonl")
[[ -n "$sampler" ]] || { log "no final sampler"; exit 3; }
log "sampler $sampler"

log "smoke-16"
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
ids=set(map(str,json.loads(Path("data/splits/eval_smoke_agentft.json").read_text())))
base=json.loads(Path("ship/eval80-smoke/base/results.json").read_text())
ours=json.loads(Path("ship/eval80-smoke/trace-sft-agent-clean-smoke16/results.json").read_text())
byb={str(i["id"]):i for i in base["items"] if str(i["id"]) in ids}
byo={str(i["id"]):i for i in ours["items"]}
gains=[]; losses=[]
for tid,o in byo.items():
    b=byb[tid]
    if b.get("pass") and not o.get("pass"): losses.append(tid)
    elif not b.get("pass") and o.get("pass"): gains.append(tid)
print("summary",ours["summary"])
print("gains",gains)
print("losses",losses)
if losses:
    print("REGRESSION — discard")
elif not gains:
    print("NO GAIN — discard")
else:
    print("STRICT IMPROVEMENT — run eval-80")
PY

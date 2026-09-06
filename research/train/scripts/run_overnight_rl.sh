#!/usr/bin/env bash
# Overnight A: RL from untuned base + KL vs base + LibreOffice recalc.
# Do not pass an oracle SFT checkpoint.
#
#   ./train/scripts/run_overnight_rl.sh
#   MAX_STEPS=40 LR=1e-5 ./train/scripts/run_overnight_rl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

LOG_PATH="${LOG_PATH:-train/logs/rl-base-kl}"
LR="${LR:-2e-5}"
# Cost-lean defaults: 20 * 8 * 4 = 640 gens vs old 50 * 16 * 8 = 6400 (~10× cheaper)
MAX_STEPS="${MAX_STEPS:-20}"
GROUP_SIZE="${GROUP_SIZE:-4}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-8}"
KL="${KL:-0.05}"
LORA_RANK="${LORA_RANK:-32}"

mkdir -p "$LOG_PATH"
echo "overnight RL from=base pool=recommended kl=$KL lr=$LR steps=$MAX_STEPS g=${GROUP_SIZE}x${GROUPS_PER_BATCH}"
exec uv run train/rl_tinker.py \
  --from-base \
  --recalc \
  --kl-penalty-coef "$KL" \
  --lr "$LR" \
  --max-steps "$MAX_STEPS" \
  --group-size "$GROUP_SIZE" \
  --groups-per-batch "$GROUPS_PER_BATCH" \
  --lora-rank "$LORA_RANK" \
  --pool recommended \
  --log-path "$LOG_PATH"

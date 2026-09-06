#!/usr/bin/env bash
# Launch A (RL), B (trace SFT), C (SDFT) in parallel via nohup.
#
#   ./train/scripts/run_overnight.sh          # all three
#   ./train/scripts/run_overnight.sh rl
#   ./train/scripts/run_overnight.sh traces
#   ./train/scripts/run_overnight.sh sdft
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${SOFFICE:-}" && -x /Applications/LibreOffice.app/Contents/MacOS/soffice ]]; then
  export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice
fi

MODE="${1:-all}"
mkdir -p train/logs

launch() {
  local name="$1"
  local script="$2"
  local log="train/logs/${name}.log"
  echo "starting $name -> $log"
  nohup "$script" >"$log" 2>&1 &
  echo $! >"train/logs/${name}.pid"
  echo "  pid $(cat "train/logs/${name}.pid")"
}

case "$MODE" in
  rl)
    launch overnight-rl "$SCRIPTS/run_overnight_rl.sh"
    ;;
  traces)
    launch overnight-traces "$SCRIPTS/run_overnight_trace_sft.sh"
    ;;
  sdft)
    launch overnight-sdft "$SCRIPTS/run_overnight_sdft.sh"
    ;;
  all|both)
    launch overnight-rl "$SCRIPTS/run_overnight_rl.sh"
    launch overnight-traces "$SCRIPTS/run_overnight_trace_sft.sh"
    launch overnight-sdft "$SCRIPTS/run_overnight_sdft.sh"
    ;;
  *)
    echo "usage: $0 [all|rl|traces|sdft]" >&2
    exit 2
    ;;
esac

echo "tail -f train/logs/overnight-rl.log train/logs/overnight-traces.log train/logs/overnight-sdft.log"
echo "morning: MODEL_PATH=tinker://... ./train/scripts/run_morning_eval.sh"

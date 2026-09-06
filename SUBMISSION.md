# Submission: spreadsheet-tinker

## Team

- Team name: spreadsheet-tinker
- Members, one GitHub handle per line:
  - Lachin7
- Repo URL: https://github.com/Lachin7/twoplustwo-encode-hackathon

## What we built and why

We stayed on the official model `Qwen/Qwen3.8-27B` via Tinker (`qwen3_8_xhigh_reasoning`) and spent the time on the harness, not another oracle LoRA. SpreadsheetBench grades only the answer cells after LibreOffice recalc, so the pipeline writes those cells and leaves the rest of the workbook alone.

Small tasks (`≤20` graded cells) stay one-shot JSON: literals or a short formula. A hardened parser strips thinking, fences, and trailing commas. Writes accept Excel formulas (`_xlfn` / FILTER), dates, merged cells, and repaired ranges. If the model gives one relative formula on a tall column or a wide row, the harness fills the rest. One-shot prompts include a 120×30 preview; agent prompts use 40 rows plus focus windows because the agent can inspect the live workbook.

Tasks with more than 20 graded cells use a six-turn Python tool on a live openpyxl workbook (`{"tool":"python","code":"..."}` / `{"tool":"done"}` / a `cells` JSON). Exec is restricted, time-capped, and exposes an allowlist of common builtins and modules. Exact worksheet names are separated from preview labels, and completion tokens are capped to Qwen's remaining context. An early `done` on a barely-filled range is sent back for another turn. The Python loop is why we ship Docker.

## Models

- Inference only: `Qwen/Qwen3.8-27B` on Tinker, renderer `qwen3_8_xhigh_reasoning`. No `tinker://` checkpoint.

## Scores on the 400

```sh
cd research
uv run evaluate.py --predictions ship/base-400-agent/predictions.jsonl --all --out ship/base-400-agent/results.json
```

```json
{"items": 400, "graded": 400, "missing": 0, "errors": 0, "pass_rate": 0.65, "cell_accuracy": 0.7692, "pass_rate_cell_level": 0.6691, "pass_rate_sheet_level": 0.608}
```

Holdout check (frozen eval-80, same pipeline, no LoRA): pass_rate **0.7625**, sheet **0.75**.

Masked agent-trace LoRA research candidate on the same frozen eval-80:
pass_rate **0.775**, sheet **0.7917**. See `research/FAILURE_ANALYSIS.md`.

## Your run on the 400

- `predictions.jsonl`: `research/ship/base-400-agent/predictions.jsonl`
- `outputs/`: `research/ship/base-400-agent/outputs/`
- `traces/`: `research/ship/base-400-agent/traces/`
- `run.log`: `research/ship/base-400-agent/run.log`

## Code

```sh
docker build -t ylookup .
docker run --rm \
  -e TINKER_API_KEY \
  -e TINKER_PROJECT_ID \
  -v /path/to/spreadsheetbench_verified_400:/data:ro \
  -v /path/to/out:/out \
  ylookup
```

Same command without Docker:

```sh
cd research
uv sync --extra tinker
uv run baseline/tinker_predict.py --dataset-dir /data --out-dir /out \
  --base-model Qwen/Qwen3.8-27B --renderer qwen3_8_xhigh_reasoning --agent auto
```

Env: `TINKER_API_KEY` (required), `TINKER_PROJECT_ID` (if your Tinker project needs it). The image execs model-written Python against the live workbook; that is why it is sandboxed. It does not run LibreOffice — judges grade the workbooks with the shipped evaluator.

## Things to look at

- `research/baseline/agent.py` — Python tool loop, `n>20` routing
- `research/baseline/common.py` — prompt, JSON parse, formula fill, thin-on-done
- `research/sb.py` — workbook serialize + focus windows
- `research/ship/verify-agent/` — train-slice one-shot vs auto (no eval leak)
- `research/baseline/test_harness_changes.py` — harness unit tests
- `research/FAILURE_ANALYSIS.md` — quantified 400 failure clusters and FT ablations
- `research/ship/eval80-clean-agent-v2/results.json` — first LoRA to beat base+agent on eval-80
- `research/data/splits/` — frozen 320/80 used only for research, not for the shipped 400

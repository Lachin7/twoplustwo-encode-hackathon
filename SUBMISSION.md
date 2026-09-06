# Submission: spreadsheet-tinker

## Team

- Team name: spreadsheet-tinker
- Members, one GitHub handle per line:
  - Lachin7
- Repo URL: https://github.com/Lachin7/twoplustwo-encode-hackathon

## What we built and why

We stayed on the official model `Qwen/Qwen3.8-27B` via Tinker (`qwen3_8_xhigh_reasoning`) and spent the time on the harness, not another oracle LoRA. SpreadsheetBench grades only the answer cells after LibreOffice recalc, so the pipeline writes those cells and leaves the rest of the workbook alone.

Small tasks (`≤20` graded cells) stay one-shot JSON: literals or a short formula. A hardened parser strips thinking, fences, and trailing commas. Writes accept Excel formulas (`_xlfn` / FILTER), dates, merged cells, and repaired ranges. If the model gives one relative formula on a tall column or a wide row, the harness fills the rest. The prompt includes a 120×30 overview plus a focus window when the answer or data sit outside that crop.

Tasks with more than 20 graded cells use a six-turn Python tool on a live openpyxl workbook (`{"tool":"python","code":"..."}` / `{"tool":"done"}` / a `cells` JSON). Exec is restricted and time-capped. An early `done` on a barely-filled range is sent back for another turn. Oracle SFT on the 400 goldens helped eval-80 and hurt large sheet tasks; we do not ship that checkpoint. The Python loop is why we ship Docker.

## Models

- Inference only: `Qwen/Qwen3.8-27B` on Tinker, renderer `qwen3_8_xhigh_reasoning`. No `tinker://` checkpoint.

## Scores on the 400

```sh
cd research
uv run evaluate.py --predictions ship/base-400-agent/predictions.jsonl --all --out ship/base-400-agent/results.json
```

`items` must be 400. Summary is filled after the shipped run finishes:

```json
{"items": 400, "graded": "...", "missing": "...", "errors": "...", "pass_rate": "...", "cell_accuracy": "...", "pass_rate_cell_level": "...", "pass_rate_sheet_level": "..."}
```

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
- `research/data/splits/` — frozen 320/80 used only for research, not for the shipped 400

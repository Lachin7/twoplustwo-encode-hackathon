# Submission: spreadsheet-tinker

## Team

- Team name: spreadsheet-tinker
- Members, one GitHub handle per line:
  - Lachin7
  - Flavio
  - Royantha
  - Kris
- Repo URL: https://github.com/Lachin7/twoplustwo-encode-hackathon

## What we built and why

SpreadsheetBench only grades answer cells after LibreOffice recalc, so we stayed on the official model `Qwen/Qwen3.8-27B` via Tinker (`qwen3_8_xhigh_reasoning`) and spent the time on a reliable write path plus one cheap LoRA. Tasks with ≤20 graded cells stay one-shot JSON (literals or a short formula). The parser strips thinking, fences, and trailing commas; writes accept Excel formulas (`_xlfn` / FILTER), dates, merged cells, and repaired ranges. One relative formula on a tall column or wide row is enough — the harness fills the rest. Tasks with more than 20 graded cells use a six-turn Python tool on a live openpyxl workbook (`python` / `done` / cells JSON). Exec is restricted, time-capped, and allowlists ordinary builtins and modules. Exact worksheet names are kept separate from preview labels, completion tokens are clipped to Qwen's remaining 65,536-token context, and agent previews are shortened so large sheets do not overflow. An early `done` on a barely-filled range is sent back. That loop is why we ship Docker. Oracle SFT, mixed-trace SFT, SDFT, and RL dumped gold values or trained on inspect/error turns and lost holdout tasks. The LoRA we ship is masked agent SFT: 66 passing train-320 traces, loss only on successful mutations, cell answers, and `done` after a mutation. No eval-80 ids and no golden cell values were in the training JSONL.

## Models

- Base: `Qwen/Qwen3.8-27B` on Tinker, renderer `qwen3_8_xhigh_reasoning`.
- Fine-tune: `tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final`
  - Data: passing agent traces from the 320-task train split (`research/train/build_clean_agent_sft.py`). No eval-80 tasks. No golden workbook values in the conversations.
  - Objective: `TrainOnWhat.CUSTOMIZED` — inspect-only, error, and thin-fill turns are context only.
  - Rank 16, lr `3e-6`, batch 4, 1 epoch, 16 optimizer steps, about two minutes wall time.

## Scores on the 400

```sh
cd research
uv run evaluate.py --predictions ship/full400-masked-16k/predictions.jsonl --all --out ship/full400-masked-16k/results.json
```

```json
{"items": 400, "graded": 400, "missing": 0, "errors": 0, "pass_rate": 0.715, "cell_accuracy": 0.7979, "pass_rate_cell_level": 0.7345, "pass_rate_sheet_level": 0.672}
```

## Your run on the 400

- `predictions.jsonl`: `research/ship/full400-masked-16k/predictions.jsonl`
- `outputs/`: `research/ship/full400-masked-16k/outputs/`
- `traces/`: `research/ship/full400-masked-16k/traces/`
- `run.log`: `research/ship/full400-masked-16k/run.log`
- `results.json`: `research/ship/full400-masked-16k/results.json`

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

Without Docker:

```sh
cd research
uv sync --extra tinker
uv run baseline/tinker_predict.py --dataset-dir /data --out-dir /out \
  --base-model Qwen/Qwen3.8-27B --renderer qwen3_8_xhigh_reasoning --agent auto \
  --model-path tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final
```

Env: `TINKER_API_KEY` (required), `TINKER_PROJECT_ID` if the Tinker project needs it. The image runs model-written Python on the live workbook. It does not run LibreOffice; judges grade the workbooks with the shipped evaluator.

## Things to look at

- `research/baseline/agent.py` — Python tool loop, `n>20` routing, sandbox allowlist
- `research/baseline/common.py` — prompt, JSON parse, formula fill, thin-on-done, context cap
- `research/sb.py` — workbook serialize; sheet title vs overview/focus
- `research/baseline/test_harness_changes.py` — harness unit tests
- `research/train/build_clean_agent_sft.py` — masked SFT builder
- `research/train/sft_tinker.py` — LoRA train
- `research/FAILURE_ANALYSIS.md` — 400 failure clusters and why other FT recipes were dropped
- `research/data/splits/` — frozen 320/80 used only for research, not for the shipped 400

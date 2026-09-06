# Submission: spreadsheet-tinker

## Team

- Team name: spreadsheet-tinker
- Members, one GitHub handle per line:
  - Lachin7
- Repo URL: https://github.com/Lachin7/twoplustwo-encode-hackathon

## What we built and why

SpreadsheetBench grades only the answer cells after LibreOffice recalc, so we kept `Qwen/Qwen3.8-27B` on Tinker (`qwen3_8_xhigh_reasoning`) and spent the weekend on the write path plus a small, clean LoRA.

Small tasks (≤20 graded cells) stay one-shot JSON: a literal or a short formula. The parser strips thinking, fences, and trailing commas; writes accept Excel formulas (`_xlfn` / FILTER), dates, merged cells, and repaired ranges. One relative formula on a tall column or wide row is enough — the harness fills the rest. One-shot prompts use a 120×30 preview.

Tasks with more than 20 graded cells use a six-turn Python tool on a live openpyxl workbook. Exec is restricted and time-capped, with an allowlist of ordinary builtins and modules. Worksheet names are kept exact and separate from preview labels (`overview` / `focus`). Completion tokens are clipped to Qwen's remaining 65,536-token context, and agent previews are shortened so large sheets do not overflow. An early `done` on a barely-filled range is sent back. That loop is why we ship Docker.

Oracle SFT, mixed-trace SFT, SDFT, and RL either dumped gold values or trained on inspect/error turns and lost eval-80. The LoRA we ship is masked agent SFT: 66 passing train-320 traces, loss only on successful mutations, cell answers, and `done` after a mutation. Eval-80 ids and golden cell values were not in the training JSONL. On the frozen eval-80 it was 62/80 (77.5%) against 61/80 (76.25%) for the same harness without LoRA. The full-400 numbers below are that checkpoint plus this harness.

## Models

- Base: `Qwen/Qwen3.8-27B` on Tinker, renderer `qwen3_8_xhigh_reasoning`.
- Fine-tune: `tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final`
  - Data: passing agent traces from the 320-task train split (`research/train/build_clean_agent_sft.py`). No eval-80 tasks. No golden workbook values in the conversations.
  - Objective: `TrainOnWhat.CUSTOMIZED` — inspect-only, error, and thin-fill turns are context only.
  - Rank 16, lr `3e-6`, batch 4, 1 epoch, 16 steps, about two minutes.

## Scores on the 400

```sh
cd research
uv run evaluate.py --predictions ship/full400-masked-16k/predictions.jsonl --all --out ship/full400-masked-16k/results.json
```

Pending the live full-400 evaluate. Replace this block with the `summary` from `results.json` (`items` must be 400):

```json
{"items": 400, "graded": "TBD", "missing": "TBD", "errors": "TBD", "pass_rate": "TBD", "cell_accuracy": "TBD", "pass_rate_cell_level": "TBD", "pass_rate_sheet_level": "TBD"}
```

Holdout (frozen eval-80, same LoRA + harness): pass_rate **0.775**, sheet **0.7917**. Previous base+agent eval-80: **0.7625**. Previous base+agent full 400 (older harness): **0.65**.

## Your run on the 400

- `predictions.jsonl`: `research/ship/full400-masked-16k/predictions.jsonl`
- `outputs/`: `research/ship/full400-masked-16k/outputs/`
- `traces/`: `research/ship/full400-masked-16k/traces/`
- `run.log`: `research/ship/full400-masked-16k/run.log`
- `results.json`: `research/ship/full400-masked-16k/results.json` (after scoring)

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
  --base-model Qwen/Qwen3.8-27B --renderer qwen3_8_xhigh_reasoning --agent auto \
  --model-path tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final
```

Env: `TINKER_API_KEY` (required), `TINKER_PROJECT_ID` (if your Tinker project needs it). The image execs model-written Python against the live workbook. It does not run LibreOffice — judges grade the workbooks with the shipped evaluator.

## Things to look at

- `research/baseline/agent.py` — Python tool loop, `n>20` routing, sandbox allowlist
- `research/baseline/common.py` — prompt, JSON parse, formula fill, thin-on-done, context cap
- `research/sb.py` — workbook serialize; sheet title vs overview/focus
- `research/baseline/test_harness_changes.py` — harness unit tests
- `research/train/build_clean_agent_sft.py` — masked SFT builder
- `research/train/sft_tinker.py` — LoRA train
- `research/FAILURE_ANALYSIS.md` — 400 failure clusters and why other FT recipes were dropped
- `research/ship/eval80-clean-agent-v2/results.json` — LoRA eval-80 that beat base+agent
- `research/data/splits/` — frozen 320/80 used only for research, not for the shipped 400

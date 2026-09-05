# Submission: spreadsheet-tinker

Research track. Primary local metric is the frozen **eval-80** holdout. Full-400 base answers are also shipped under `ship/base-400/`.

## Team

- Team name: spreadsheet-tinker
- Members, one GitHub handle per line:
  - Lachin7
- Repo URL: https://github.com/Lachin7/twoplustwo-encode-hackathon

## What we built and why

We treat SpreadsheetBench as fill-the-graded-cells, not rebuild-the-workbook. The one-shot baseline already tells the model the answer range and asks for JSON values. We kept that contract and hardened the harness (coerce, align to `answer_position`, retry on bad JSON, skip huge ranges, pin `qwen3_8_disable_thinking`).

Training uses a frozen 320/80 split (`data/splits/`, seed 42). Train-small (≤200 graded cells) is the SFT set; huge tasks stay on harness fallback. Primary SFT is **oracle** (golden JSON as assistant; **no goldens in the user prompt**). Optional Gemini teacher via Google AI Studio (`GOOGLE_API_KEY`) can replace/filter traces with pass-only keeps (`train/scripts/`).

Inference: `baseline/tinker_predict.py` / `train/run_infer.py`. Env: `TINKER_API_KEY`, `TINKER_PROJECT_ID`.

## Models

- Base / student: `Qwen/Qwen3.8-27B` on Tinker, renderer `qwen3_8_disable_thinking`
- **Shipped SFT (use this):** `tinker://6fddcac0-b2a6-545a-b7e4-b3886d94d9d6:train:0/sampler_weights/final`  
  Oracle LoRA, rank 32, lr 4e-4, batch 8, max_length 16384, 1 epoch (`train/sft_tinker.py` defaults)
- Do **not** use `train/logs/sft2` / windowed oracle — that run is **worse than base** (~27.5% on eval-80)
- Teacher (optional data only): `gemini-2.5-pro` via Google AI Studio, or `--oracle`

## Scores on the holdout 80

Frozen ids: `data/splits/eval.json` (56 cell-level + 24 sheet-level).

| Run | Artifact | pass_rate |
|-----|----------|-----------|
| Base (no LoRA) | `ship/eval80-base/` | **0.325** |
| **SFT oracle (shipped)** | `ship/sft-eval80/` | **0.450** |
| Base full 400 | `ship/base-400/` | **0.4375** |

```json
{"items": 80, "graded": 80, "missing": 0, "errors": 0, "pass_rate": 0.45, "cell_accuracy": 0.3078, "pass_rate_cell_level": 0.4643, "pass_rate_sheet_level": 0.4167}
```

Base for comparison:

```json
{"items": 80, "graded": 80, "missing": 0, "errors": 0, "pass_rate": 0.325, "cell_accuracy": 0.3135, "pass_rate_cell_level": 0.2143, "pass_rate_sheet_level": 0.5833}
```

Full-400 base (`Qwen/Qwen3.8-27B`, no LoRA):

```json
{"items": 400, "graded": 400, "missing": 0, "errors": 0, "pass_rate": 0.4375, "cell_accuracy": 0.3636, "pass_rate_cell_level": 0.4727, "pass_rate_sheet_level": 0.36}
```

## Your run on the holdout 80 (shipped)

Judges can score without re-calling Tinker:

- `ship/sft-eval80/predictions.jsonl`
- `ship/sft-eval80/outputs/`
- `ship/sft-eval80/traces/`
- `ship/sft-eval80/run.log`
- `ship/sft-eval80/results.json`

```sh
cd research
uv run evaluate.py --predictions ship/sft-eval80/predictions.jsonl --out ship/sft-eval80/results.json
```

## Reproduce SFT + eval (optional)

Needs Tinker credentials. Checkpoint path is on our Tinker project; retrain if you cannot load it:

```sh
cd research
uv sync --extra tinker
export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice   # or Linux path

# 1) oracle chats for train-small
uv run train/teacher_label.py --oracle

# 2) LoRA SFT (same hyperparams as the 45% run)
uv run train/sft_tinker.py --jsonl data/sft/spreadsheet_sft.jsonl --lora-rank 32 --lr 4e-4 --epochs 1

# 3) eval-80 with the new sampler_weights/final from the log dir
uv run train/run_infer.py --ids-file data/splits/eval.json \
  --out-dir ship/sft-eval80-repro --results ship/sft-eval80-repro/results.json \
  --model-path tinker://<new-run>/sampler_weights/final
```

Or load the shipped checkpoint (if still readable on the project):

```sh
uv run train/run_infer.py --ids-file data/splits/eval.json \
  --out-dir ship/sft-eval80 --results ship/sft-eval80/results.json \
  --model-path tinker://6fddcac0-b2a6-545a-b7e4-b3886d94d9d6:train:0/sampler_weights/final
```

Full 400 (costly): `uv run train/run_infer.py --all --resume --out-dir ship --model-path tinker://6fddcac0-b2a6-545a-b7e4-b3886d94d9d6:train:0/sampler_weights/final`

## Code (one-shot values, no model-written code execution)

```sh
uv run baseline/tinker_predict.py --dataset-dir /data --out-dir /out \
  --base-model Qwen/Qwen3.8-27B --renderer qwen3_8_disable_thinking \
  --model-path tinker://6fddcac0-b2a6-545a-b7e4-b3886d94d9d6:train:0/sampler_weights/final
```

Without `--model-path` this is the base model (~32.5% on our eval-80), not the shipped SFT.

## Things to look at

- `train/SHIPPED_CHECKPOINT.txt` — which LoRA is submission
- `data/splits/` — frozen 320/80
- `baseline/common.py` — harness
- `train/teacher_label.py` + `train/scripts/` — oracle / Gemini labelling
- `train/sft_tinker.py` — LoRA SFT defaults (= 45% recipe)
- `ship/sft-eval80/` — graded holdout run
- `ship/eval80-base/` — base comparison on holdout 80
- `ship/base-400/` — base full-400 predictions + `results.json` (pass_rate 0.4375)
- `train/sft_tinker.py` filters to train-small by default (`--all-train` for all 320)

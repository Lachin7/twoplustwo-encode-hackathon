# Research track: SpreadsheetBench

Given a workbook and a plain-English instruction from a real Excel forum post, produce the workbook with the answer filled in. Fine-tune a model, build a harness around one, or both. The score is the share of tasks where every graded cell matches the golden workbook.

Dataset: [SpreadsheetBench Verified](https://huggingface.co/datasets/KAKA22/SpreadsheetBench), 400 human-validated tasks. Licence CC-BY-SA-4.0. Paper: [arxiv 2406.14991](https://arxiv.org/abs/2406.14991).

## Setup

```sh
uv sync
uv run data/download.py
```

Then install LibreOffice, which the scorer uses to recalculate formulas:

| OS | Install | Notes |
|---|---|---|
| macOS | `brew install --cask libreoffice` | tested |
| Linux | `sudo apt install libreoffice-calc` | tested on Debian 12 |
| Windows | use WSL2 with Ubuntu, then follow the Linux row | native Windows is not supported |

If `soffice` is somewhere else, set `SOFFICE=/path/to/soffice`. Run every script with `uv run`, for example `uv run evaluate.py --oracle`.

`download.py` pulls a 15 MB tarball from Hugging Face, checks its SHA-256, and unpacks it to `data/spreadsheetbench_verified_400/`. Direct link if you prefer:

```
https://huggingface.co/datasets/KAKA22/SpreadsheetBench/resolve/main/spreadsheetbench_verified_400.tar.gz?download=true
```

LibreOffice is only needed to score. The evaluator recalculates your output workbooks headlessly so formulas you write with openpyxl get values.

If you write formulas with openpyxl, newer Excel functions need the prefix Excel itself stores in the file, or both Excel and the LibreOffice recalculation return `#NAME?`: `_xlfn.XLOOKUP(...)`, `_xlfn.UNIQUE(...)`, `_xlfn.LET(...)`, `_xlfn.CHOOSECOLS(...)`, and `_xlfn._xlws.FILTER(...)`. Classic functions such as `SUM`, `SUMIFS`, `INDEX`, `MATCH`, `VLOOKUP` need no prefix. Dates must be written as real dates, not text.

## Data

```
data/spreadsheetbench_verified_400/
  dataset.json                 400 tasks
  spreadsheet/<id>/
    1_<id>_init.xlsx           the workbook you start from
    1_<id>_golden.xlsx         the expected result
    prompt.txt                 the instruction
```

One entry in `dataset.json`:

```json
{"id": "51-12", "instruction": "How can I create a VBA code that will count ...", "spreadsheet_path": "spreadsheet/51-12",
 "instruction_type": "Sheet-Level Manipulation", "answer_position": "B6", "answer_sheet": "Sheet1",
 "data_position": "A1:M7"}
```

| Type | Tasks |
|---|---|
| Cell-Level Manipulation | 275 |
| Sheet-Level Manipulation | 125 |

## Score

Only the cells in `answer_position` on `answer_sheet` are compared, after recalculation, with the official SpreadsheetBench normalisation: numbers rounded to 2 decimals, dates as Excel serials, empty string equals empty cell.

| Metric | Meaning | Use |
|---|---|---|
| `pass_rate` | tasks where every graded cell matches | ranking |
| `cell_accuracy` | graded cells that match, over all tasks | tie-break |
| `pass_rate_cell_level`, `pass_rate_sheet_level` | pass rate per instruction type | judges |

```sh
uv run evaluate.py --predictions submissions/<team>/predictions.jsonl
uv run evaluate.py --predictions submissions/<team>/predictions.jsonl --all   # every task counts, missing = fail
uv run evaluate.py --oracle            # golden vs golden, must be 1.0
uv run evaluate.py --predictions ... --out results.json                       # per-cell results as JSON
```

Without `--all` only the tasks present in `predictions.jsonl` are scored, which is what you want while iterating on a few ids. Judges always score with `--all`.

## Quick start

```sh
uv run baseline/llm_predict.py --out-dir submissions/my-llm --model deepseek/deepseek-v3.2 --ids 13-1,51-12
uv run evaluate.py --predictions submissions/my-llm/predictions.jsonl --no-recalc
```

`llm_predict.py` is a one-shot LLM baseline via [Pydantic AI](https://ai.pydantic.dev) + OpenRouter. Put `OPENROUTER_API_KEY` in `.env`. It writes `predictions.jsonl`, `outputs/<id>.xlsx`, `traces/<id>.jsonl` with one line per model call, and `run.log` into the out dir. `--concurrency 8` runs the 400 in minutes.

`tinker_predict.py` is the same baseline through [Tinker](https://tinker-docs.thinkingmachines.ai), for a base model or your fine-tuned checkpoint. `uv sync --extra tinker`, put `TINKER_API_KEY` in `.env`, then `--base-model Qwen/Qwen3-8B` and optionally `--model-path tinker://<run-id>/sampler_weights/final`.

Reference numbers for one-shot prompting, values not formulas, on all 400: DeepSeek-V3.2 55.8% pass, Qwen3.8-27B 59.0% pass, Gemini 3.7 Flash 68.3% pass.

## Docker

If your pipeline is an agent or executes model-written code, it runs inside a Docker container you write. The judges mount a dataset dir read-only at `/data` and take `/out`. See [SUBMISSION.md](SUBMISSION.md).

## Submit

See [SUBMISSION.md](SUBMISSION.md).

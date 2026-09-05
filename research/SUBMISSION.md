# Research track submission

Your work lives in **your team's own repo**. By Sunday 12:00 you send us its URL through a form we share then. Everything below must be in it.

```
<your-repo>/
  SUBMISSION.md          filled copy of SUBMISSION_TEMPLATE.md: team, approach, models, scores
  Dockerfile             if your pipeline is an agent or executes model-written code
  predictions.jsonl      your run on the 400
  outputs/<id>.xlsx      your run on the 400
  traces/<id>.jsonl      your run on the 400, one line per model call
  run.log                your run on the 400
  results.json           shipped evaluator on that run
```

## Docker first

The judges run your pipeline themselves on tasks you have not seen, in the same layout as `data/spreadsheetbench_verified_400`: a `dataset.json`, and per task an init workbook and a `prompt.txt`. Your container reads them from `/data`, mounted read-only, and writes everything to `/out`.

If your pipeline runs model-written code, and most will, that code runs inside your container and nowhere else. The judge's laptop only sees `/out`.

```sh
docker build -t <team> .
docker run --rm -e <YOUR_API_KEYS> -v <dataset dir>:/data:ro -v <empty dir>:/out <team>
```

- Keys come from environment variables. Name them in `SUBMISSION.md`. Never in the repo.
- Model ids fixed in code, temperature 0 where the API allows it.
- Nothing else mounted, no `--privileged`, no host paths.
- A pipeline that is one model call per task and executes no code may skip Docker. Then your repo has the script and a `pyproject.toml` with `uv.lock`, and `SUBMISSION.md` names the one `uv run` command that takes `--dataset-dir` and `--out-dir`.

Sampling is not deterministic, so two runs differ by a few tasks. That is expected.

## What `/out` must contain

**`predictions.jsonl`**, one line per task. Whatever your pipeline is, one model call or an agent that runs code, the deliverable is one workbook per task. The evaluator reads this file to find each workbook and grades only the answer cells in it.

```json
{"id": "51-12", "output": "outputs/51-12.xlsx", "status": "ok"}
```

`status` is `ok` or the error text. A task without a line, or whose file is missing or unreadable, scores zero. If your pipeline fails on a task, still write the line and copy the init workbook as the output.

**`traces/<id>.jsonl`**, one line per model call, in order. Judges read these for the top teams. A trace with the golden value and no reasoning, a prompt containing golden values, or a lookup step is a disqualification.

```json
{"step": 1, "model": "openrouter:deepseek/deepseek-v3.2", "prompt": "...", "response": "...", "input_tokens": 3210, "output_tokens": 412, "latency_ms": 2380, "error": null}
```

Agents add `tool`, `tool_input`, `tool_output` per step. Truncate a workbook serialisation to 20k characters if you must and say so. Keep failed calls in the file, with `error` set.

**`run.log`**, the stdout and stderr of the run, unedited.

The baselines in `baseline/` write all four. Copy how they do it.

## Scores you report

Run the shipped evaluator on your own outputs for the 400 and put the file in the repo:

```sh
uv run evaluate.py --predictions <your predictions.jsonl> --all --out results.json
```

Paste the `summary` block into `SUBMISSION.md`. `--all` makes `items` 400. Numbers from any other scorer, or without the file, are not accepted.

| Field | Meaning |
|---|---|
| `pass_rate` | tasks where every graded cell matches. The ranking metric. |
| `cell_accuracy` | graded cells that match, over all tasks. Tie-break. |
| `pass_rate_cell_level`, `pass_rate_sheet_level` | pass rate per instruction type |
| `items`, `graded`, `missing`, `errors` | tasks scored, skipped, crashed |

Reported scores decide which teams judges run first. They do not rank you. The judges' run does.

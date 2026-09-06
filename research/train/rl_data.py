"""Recommended one-shot RL pool for SpreadsheetBench (cookbook ProblemEnv).

Standard RL here is GRPO-style sampling against the Excel grader — not the
Python agent loop, and not another oracle-JSON clone. The pool is built so
the policy still has contrast after SFT:

  include  912 IDs that are not in verified-400 (all cases, ≤ max_cells)
  include  verified train-320 cases 2–3 (different numbers than SFT case-1)
  exclude  eval-80 entirely (judged leak)
  exclude  verified train case-1 (already in the 286 oracle SFT; reward saturates)
  hold out 10% of unseen bases as cookbook-style RL-dev (not the judged eval)

    uv run train/build_rl_dataset.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal

from openpyxl.utils.cell import range_boundaries

from train.paths import (
    RL_DEV_BASE_FRACTION,
    RL_DEV_BASE_MAX,
    RL_DEV_BASE_MIN,
    ROOT,
    SMALL_CELL_LIMIT,
    SPLIT_SEED,
)

BUCKET_UNSEEN = "unseen_912"
BUCKET_TRAIN_VARIANT = "train_variant"
Split = Literal["train", "dev"]

TASK_FIELDS = (
    "id",
    "base_id",
    "case",
    "instruction",
    "instruction_type",
    "answer_position",
    "answer_sheet",
    "data_position",
    "spreadsheet_path",
    "init_xlsx",
    "golden_xlsx",
)


def repair_range(rng: str) -> str:
    rng = "".join(rng.split())
    if ":" not in rng:
        return rng
    start, end = rng.split(":", 1)
    if end.isdigit():
        col = "".join(ch for ch in start if ch.isalpha())
        return f"{start}:{col}{end}"
    return rng


def graded_span(answer_position: str) -> int | None:
    """Answer-range size without opening the workbook. Whole-column → None."""
    cleaned = answer_position.replace("'", "").replace('"', "")
    tokens = [cleaned] if cleaned.count("!") == 1 else cleaned.split(",")
    total = 0
    for tok in tokens:
        rng = repair_range(tok.strip().rsplit("!", 1)[-1])
        try:
            min_col, min_row, max_col, last_row = range_boundaries(rng)
        except Exception:
            return None
        if None in (min_col, min_row, max_col, last_row):
            return None
        total += (last_row - min_row + 1) * (max_col - min_col + 1)
    return total


def candidate_bucket(
    base_id: str,
    case: int,
    *,
    verified: set[str],
    eval_ids: set[str],
    train_ids: set[str],
) -> str | None:
    """Return the RL bucket, or None if the task must not be used."""
    base_id = str(base_id)
    if base_id in eval_ids:
        return None
    if base_id not in verified:
        return BUCKET_UNSEEN
    if base_id in train_ids and int(case) >= 2:
        return BUCKET_TRAIN_VARIANT
    return None


def rel_to_root(path: str | Path, root: Path = ROOT) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def task_record(task: dict, *, bucket: str, n_cells: int, split: Split, root: Path = ROOT) -> dict:
    row = {
        "split": split,
        "bucket": bucket,
        "n_cells": n_cells,
        "source": "rl-recommended",
    }
    for key in TASK_FIELDS:
        if key not in task:
            continue
        value = task[key]
        if key in ("init_xlsx", "golden_xlsx") and value:
            value = rel_to_root(value, root)
        row[key] = value
    return row


def hydrate_task(row: dict, root: Path = ROOT) -> dict:
    task = {key: row[key] for key in TASK_FIELDS if key in row}
    for key in ("init_xlsx", "golden_xlsx"):
        value = task.get(key)
        if not value:
            continue
        path = Path(value)
        task[key] = str(path if path.is_absolute() else root / path)
    task["id"] = str(task["id"])
    return task


def pick_dev_bases(unseen_bases: Iterable[str], *, seed: int = SPLIT_SEED) -> set[str]:
    bases = sorted(set(unseen_bases))
    if not bases:
        return set()
    n = min(RL_DEV_BASE_MAX, max(RL_DEV_BASE_MIN, int(round(len(bases) * RL_DEV_BASE_FRACTION))))
    n = min(n, max(1, len(bases) // 5))  # never hold out more than 20%
    rng = random.Random(seed)
    rng.shuffle(bases)
    return set(bases[:n])


def select_rl_tasks(
    cases: list[dict],
    *,
    verified: set[str],
    eval_ids: set[str],
    train_ids: set[str],
    max_cells: int = SMALL_CELL_LIMIT,
    seed: int = SPLIT_SEED,
    root: Path = ROOT,
) -> tuple[list[dict], list[dict], dict]:
    """Filter 912-style case dicts into recommended train/dev records."""
    tagged: list[tuple[str, int, dict]] = []
    skipped = Counter()
    for task in cases:
        base = str(task.get("base_id") or str(task["id"]).split("__c")[0])
        case = int(task.get("case") or 1)
        task = dict(task)
        task["base_id"] = base
        bucket = candidate_bucket(
            base, case, verified=verified, eval_ids=eval_ids, train_ids=train_ids
        )
        if bucket is None:
            skipped["excluded_split"] += 1
            continue
        if not task.get("golden_xlsx") or not task.get("init_xlsx"):
            skipped["missing_xlsx"] += 1
            continue
        span = graded_span(task.get("answer_position") or "")
        if span is None:
            skipped["unbounded_range"] += 1
            continue
        if span > max_cells:
            skipped["too_many_cells"] += 1
            continue
        if span < 1:
            skipped["empty_range"] += 1
            continue
        tagged.append((bucket, span, task))

    unseen_bases = {t["base_id"] for bucket, _span, t in tagged if bucket == BUCKET_UNSEEN}
    dev_bases = pick_dev_bases(unseen_bases, seed=seed)

    train: list[dict] = []
    dev: list[dict] = []
    for bucket, span, task in tagged:
        split: Split = "dev" if task["base_id"] in dev_bases else "train"
        record = task_record(task, bucket=bucket, n_cells=span, split=split, root=root)
        (dev if split == "dev" else train).append(record)

    train.sort(key=lambda r: (r["base_id"], int(r["case"])))
    dev.sort(key=lambda r: (r["base_id"], int(r["case"])))
    summary = {
        "seed": seed,
        "max_cells": max_cells,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_dev_bases": len(dev_bases),
        "train_by_bucket": dict(Counter(r["bucket"] for r in train)),
        "dev_by_bucket": dict(Counter(r["bucket"] for r in dev)),
        "train_by_case": dict(sorted(Counter(int(r["case"]) for r in train).items())),
        "dev_by_case": dict(sorted(Counter(int(r["case"]) for r in dev).items())),
        "skipped": dict(skipped),
        "policy": {
            "include": [
                "912 ids not in verified-400, all cases, ≤ max_cells",
                "verified train cases 2–3, ≤ max_cells",
            ],
            "exclude": [
                "eval-80 any case",
                "verified train case-1 (already SFT'd)",
                "unbounded / missing answer ranges",
            ],
            "dev": "held-out unseen-912 bases only; never judged eval-80",
        },
    }
    return train, dev, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rl_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run uv run train/build_rl_dataset.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rl_tasks(path: Path, root: Path = ROOT) -> list[dict]:
    return [hydrate_task(row, root) for row in load_rl_records(path)]

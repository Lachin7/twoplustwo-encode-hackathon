"""Stratified 320/80 split of SpreadsheetBench Verified. Freeze eval ids.

    uv run train/split_400.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sb import DEFAULT_DATASET, answer_cells, load_dataset
from train.paths import (
    EVAL_CELL,
    EVAL_IDS,
    EVAL_SHEET,
    SMALL_CELL_LIMIT,
    SPLIT_META,
    SPLIT_SEED,
    SPLITS,
    TRAIN_IDS,
)


def cell_count(task: dict) -> int:
    wb = openpyxl.load_workbook(task["init_xlsx"], data_only=True)
    try:
        return len(answer_cells(task, wb))
    finally:
        wb.close()


def kind(task: dict) -> str:
    t = task.get("instruction_type") or ""
    return "sheet" if t.startswith("Sheet") else "cell"


def main() -> None:
    tasks = load_dataset(DEFAULT_DATASET)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, int] = {}
    for task in tasks:
        n = cell_count(task)
        counts[task["id"]] = n
        task["_n_cells"] = n
        by_kind[kind(task)].append(task)

    rng = random.Random(SPLIT_SEED)
    eval_ids: list[str] = []
    for group, n_eval in (("cell", EVAL_CELL), ("sheet", EVAL_SHEET)):
        ids = [t["id"] for t in by_kind[group]]
        rng.shuffle(ids)
        eval_ids.extend(ids[:n_eval])

    eval_set = set(eval_ids)
    train_ids = [t["id"] for t in tasks if t["id"] not in eval_set]
    train_small = [i for i in train_ids if counts[i] <= SMALL_CELL_LIMIT]
    train_huge = [i for i in train_ids if counts[i] > SMALL_CELL_LIMIT]
    eval_small = [i for i in eval_ids if counts[i] <= SMALL_CELL_LIMIT]

    SPLITS.mkdir(parents=True, exist_ok=True)
    TRAIN_IDS.write_text(json.dumps(train_ids, indent=2) + "\n")
    EVAL_IDS.write_text(json.dumps(eval_ids, indent=2) + "\n")
    meta = {
        "seed": SPLIT_SEED,
        "small_cell_limit": SMALL_CELL_LIMIT,
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "n_train_cell": sum(1 for t in tasks if t["id"] in set(train_ids) and kind(t) == "cell"),
        "n_train_sheet": sum(1 for t in tasks if t["id"] in set(train_ids) and kind(t) == "sheet"),
        "n_eval_cell": sum(1 for t in tasks if t["id"] in eval_set and kind(t) == "cell"),
        "n_eval_sheet": sum(1 for t in tasks if t["id"] in eval_set and kind(t) == "sheet"),
        "n_train_small": len(train_small),
        "n_train_huge": len(train_huge),
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "train_small_ids": train_small,
        "train_huge_ids": train_huge,
        "eval_small_ids": eval_small,
        "n_cells": counts,
    }
    SPLIT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"train {len(train_ids)} (small {len(train_small)}, huge {len(train_huge)})  "
        f"eval {len(eval_ids)} (small {len(eval_small)})  -> {SPLIT_META}"
    )


if __name__ == "__main__":
    main()

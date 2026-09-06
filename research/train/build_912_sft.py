"""Build oracle SFT JSONL from SpreadsheetBench 912 (no verified-400 leak).

    # Case 1 only of the 512 unseen ids (≤200 cells)
    uv run train/build_912_sft.py --mode case1

    # Full useful pool: 512 all cases + train-320 cases 2–3 (≤200). No eval-80.
    uv run train/build_912_sft.py --mode full
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

import openpyxl
from openpyxl.utils.cell import range_boundaries

from common import FORMAT_HINT, SYSTEM_PROMPT, build_prompt, load_env
from sb import load_answer_values, load_dataset_cases, transform_value
from train.paths import (
    DATA,
    DATA_912,
    SFT_512_CASE1_JSONL,
    SFT_512_CASE1_MANIFEST,
    SFT_512_FULL_JSONL,
    SFT_512_FULL_MANIFEST,
    SMALL_CELL_LIMIT,
    SPLIT_META,
)

TARBALL = DATA / "spreadsheetbench_912_v0.1.tar.gz"
VERIFIED = DATA / "spreadsheetbench_verified_400"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("case1", "full"), required=True)
    p.add_argument("--max-cells", type=int, default=SMALL_CELL_LIMIT)
    p.add_argument("--extract", action="store_true", help="extract 912 tarball if missing")
    return p.parse_args()


def ensure_912(extract: bool) -> Path:
    if (DATA_912 / "dataset.json").exists() and (DATA_912 / "spreadsheet").is_dir():
        # spreadsheet may be incomplete if only dataset.json was extracted earlier
        n_folders = sum(1 for _ in (DATA_912 / "spreadsheet").iterdir())
        if n_folders > 100:
            return DATA_912
    if not extract and not TARBALL.exists():
        raise SystemExit(f"missing {TARBALL}; download it or pass --extract after download")
    if not TARBALL.exists():
        raise SystemExit(f"missing {TARBALL}")
    print(f"extracting {TARBALL.name} -> {DATA}")
    with tarfile.open(TARBALL) as tar:
        tar.extractall(DATA, filter="data")
    return DATA_912


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
    cleaned = answer_position.replace("'", "").replace('"', "")
    tokens = [cleaned] if cleaned.count("!") == 1 else cleaned.split(",")
    total = 0
    for tok in tokens:
        rng = repair_range(tok.strip().rsplit("!", 1)[-1])
        try:
            min_col, min_row, max_col, last_row = range_boundaries(rng)
        except Exception:
            return None
        min_row = min_row or 1
        last_row = last_row or min_row
        if max_col is None or last_row is None:
            return None
        total += (last_row - min_row + 1) * (max_col - min_col + 1)
    return total


def jsonable(value):
    return transform_value(value)


def conversation(task: dict, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task) + FORMAT_HINT},
            {"role": "assistant", "content": assistant},
        ]
    }


def golden_assistant(task: dict) -> str:
    gold = load_answer_values(task["golden_xlsx"], task)
    cells = [{"cell": coord, "value": jsonable(value)} for (_sheet, coord), value in gold.items()]
    return json.dumps({"cells": cells}, ensure_ascii=False)


def write_pool(tasks: list[dict], jsonl: Path, manifest: Path) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with jsonl.open("w", encoding="utf-8") as jf, manifest.open("w", encoding="utf-8") as mf:
        for task in tasks:
            try:
                assistant = golden_assistant(task)
                n = len(json.loads(assistant)["cells"])
                # reopen init to count if answer empty? still write
                jf.write(json.dumps(conversation(task, assistant), ensure_ascii=False) + "\n")
                mf.write(
                    json.dumps(
                        {
                            "id": task["id"],
                            "base_id": task.get("base_id", task["id"]),
                            "case": task.get("case", 1),
                            "source": "oracle-912",
                            "kept": True,
                            "pass": True,
                            "n_cells": n,
                            "type": task.get("instruction_type"),
                        }
                    )
                    + "\n"
                )
                kept += 1
            except Exception as e:
                skipped += 1
                mf.write(
                    json.dumps(
                        {
                            "id": task["id"],
                            "base_id": task.get("base_id"),
                            "case": task.get("case"),
                            "kept": False,
                            "error": f"{type(e).__name__}: {e}"[:300],
                        }
                    )
                    + "\n"
                )
    print(f"wrote {kept} kept, {skipped} skipped -> {jsonl}")


def select_tasks(mode: str, max_cells: int) -> list[dict]:
    meta = json.loads(SPLIT_META.read_text())
    verified = {str(t["id"]) for t in json.loads((VERIFIED / "dataset.json").read_text())}
    eval_ids = set(map(str, meta["eval_ids"]))
    train_ids = set(map(str, meta["train_ids"]))

    all_cases = load_dataset_cases(DATA_912, cases=None)
    # Prefer 912 folders; filter by base id
    by_key: dict[tuple[str, int], dict] = {}
    for t in all_cases:
        base = t.get("base_id") or t["id"].split("__c")[0]
        t["base_id"] = base
        by_key[(base, int(t.get("case") or 1))] = t

    # Also pull train cases 2–3 from 912 for verified train ids (full mode)
    selected: list[dict] = []
    seen = set()

    def add(task: dict) -> None:
        key = (task["base_id"], int(task.get("case") or 1), task["init_xlsx"])
        if key in seen:
            return
        span = graded_span(task["answer_position"])
        if span is None or span > max_cells:
            return
        # skip whole-column monsters that inflate via max_row — already handled if span None
        seen.add(key)
        selected.append(task)

    if mode == "case1":
        for (base, case), t in by_key.items():
            if base in verified:
                continue  # unseen only
            if case != 1:
                continue
            add(t)
    else:
        # 512 all cases
        for (base, case), t in by_key.items():
            if base in verified:
                continue
            add(t)
        # train-320 cases 2–3 only (not case 1 — already in the 286 LoRA; not eval)
        for (base, case), t in by_key.items():
            if base not in train_ids or base in eval_ids:
                continue
            if case < 2:
                continue
            add(t)

    selected.sort(key=lambda t: (t["base_id"], int(t.get("case") or 1)))
    print(
        f"mode={mode} selected={len(selected)} "
        f"unique_ids={len({t['base_id'] for t in selected})} max_cells={max_cells}"
    )
    return selected


def main() -> None:
    load_env()
    args = parse_args()
    ensure_912(extract=True)
    tasks = select_tasks(args.mode, args.max_cells)
    if args.mode == "case1":
        write_pool(tasks, SFT_512_CASE1_JSONL, SFT_512_CASE1_MANIFEST)
    else:
        write_pool(tasks, SFT_512_FULL_JSONL, SFT_512_FULL_MANIFEST)


if __name__ == "__main__":
    main()

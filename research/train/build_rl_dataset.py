"""Write the recommended standard-RL train/dev JSONL.

    uv run train/build_rl_dataset.py
    uv run train/build_rl_dataset.py --max-cells 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from sb import load_dataset_cases
from train.build_912_sft import ensure_912
from train.paths import (
    DATA_912,
    RL_DEV_JSONL,
    RL_SUMMARY,
    RL_TRAIN_JSONL,
    SMALL_CELL_LIMIT,
    SPLIT_META,
    SPLIT_SEED,
    VERIFIED_400,
)
from train.rl_data import select_rl_tasks, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-cells", type=int, default=SMALL_CELL_LIMIT)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_912(extract=True)
    meta = json.loads(SPLIT_META.read_text())
    verified = {str(t["id"]) for t in json.loads((VERIFIED_400 / "dataset.json").read_text())}
    cases = load_dataset_cases(DATA_912, cases=None)
    train, dev, summary = select_rl_tasks(
        cases,
        verified=verified,
        eval_ids=set(map(str, meta["eval_ids"])),
        train_ids=set(map(str, meta["train_ids"])),
        max_cells=args.max_cells,
        seed=args.seed,
    )
    write_jsonl(RL_TRAIN_JSONL, train)
    write_jsonl(RL_DEV_JSONL, dev)
    RL_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"rl train={summary['n_train']} dev={summary['n_dev']} "
        f"dev_bases={summary['n_dev_bases']} max_cells={args.max_cells}"
    )
    print(f"  train buckets {summary['train_by_bucket']}")
    print(f"  skipped {summary['skipped']}")
    print(f"  -> {RL_TRAIN_JSONL}")
    print(f"  -> {RL_DEV_JSONL}")


if __name__ == "__main__":
    main()

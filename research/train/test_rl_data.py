"""Unit checks for the recommended RL pool (no 912 download required).

    uv run train/test_rl_data.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.rl_data import (
    BUCKET_TRAIN_VARIANT,
    BUCKET_UNSEEN,
    candidate_bucket,
    graded_span,
    hydrate_task,
    select_rl_tasks,
    task_record,
)

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _task(base: str, case: int, rng: str = "A1:A10", **extra) -> dict:
    row = {
        "id": f"{base}__c{case}",
        "base_id": base,
        "case": case,
        "instruction": "fill",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": rng,
        "answer_sheet": "Sheet1",
        "spreadsheet_path": f"spreadsheet/{base}",
        "init_xlsx": f"/tmp/data/{base}/init.xlsx",
        "golden_xlsx": f"/tmp/data/{base}/gold.xlsx",
    }
    row.update(extra)
    return row


def test_span() -> None:
    print("\n[1] graded_span")
    check("A1:A10 is 10", graded_span("A1:A10") == 10)
    check("A1:B3 is 6", graded_span("A1:B3") == 6)
    check("whole column is None", graded_span("A:A") is None)
    check("broken C1:7 repairs to 7", graded_span("C1:7") == 7)


def test_buckets() -> None:
    print("\n[2] candidate_bucket")
    verified = {"t1", "e1"}
    eval_ids = {"e1"}
    train_ids = {"t1"}
    kw = dict(verified=verified, eval_ids=eval_ids, train_ids=train_ids)
    check("eval dropped", candidate_bucket("e1", 2, **kw) is None)
    check("train c1 dropped", candidate_bucket("t1", 1, **kw) is None)
    check("train c2 kept", candidate_bucket("t1", 2, **kw) == BUCKET_TRAIN_VARIANT)
    check("unseen kept", candidate_bucket("u1", 1, **kw) == BUCKET_UNSEEN)


def test_select() -> None:
    print("\n[3] select_rl_tasks")
    verified = {"t1", "e1"}
    eval_ids = {"e1"}
    train_ids = {"t1"}
    cases = [
        _task("e1", 1),
        _task("e1", 2),
        _task("t1", 1),
        _task("t1", 2),
        _task("t1", 3),
        _task("u1", 1),
        _task("u2", 1),
        _task("u3", 1, rng="A1:A500"),
        _task("u4", 1, rng="A:A"),
        _task("u5", 1, golden_xlsx=""),
    ]
    # many unseen bases so the 10% holdout can fire without eating train
    for i in range(20):
        cases.append(_task(f"ux{i}", 1))
    train, dev, summary = select_rl_tasks(
        cases,
        verified=verified,
        eval_ids=eval_ids,
        train_ids=train_ids,
        max_cells=200,
        seed=42,
        root=Path("/tmp"),
    )
    train_bases = {r["base_id"] for r in train}
    dev_bases = {r["base_id"] for r in dev}
    all_ids = {r["id"] for r in train + dev}
    check("eval never present", not any(r["base_id"] == "e1" for r in train + dev))
    check("train case-1 never present", "t1__c1" not in all_ids)
    check("train variants kept", {"t1__c2", "t1__c3"} <= all_ids)
    check("huge dropped", "u3__c1" not in all_ids)
    check("unbounded dropped", "u4__c1" not in all_ids)
    check("missing golden dropped", "u5__c1" not in all_ids)
    check("dev is unseen-only", all(r["bucket"] == BUCKET_UNSEEN for r in dev))
    check("train variants stay in train", "t1" in train_bases and "t1" not in dev_bases)
    check("summary counts match", summary["n_train"] == len(train) and summary["n_dev"] == len(dev))
    check("dev/train bases disjoint", not (train_bases & dev_bases))


def test_hydrate() -> None:
    print("\n[4] path hydrate")
    root = Path(tempfile.mkdtemp())
    rec = task_record(
        _task("u1", 2, init_xlsx=str(root / "a.xlsx"), golden_xlsx=str(root / "b.xlsx")),
        bucket=BUCKET_UNSEEN,
        n_cells=10,
        split="train",
        root=root,
    )
    check("stores relative init", rec["init_xlsx"] == "a.xlsx")
    task = hydrate_task(rec, root)
    check("hydrates absolute init", Path(task["init_xlsx"]) == root / "a.xlsx")


def main() -> None:
    test_span()
    test_buckets()
    test_select()
    test_hydrate()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

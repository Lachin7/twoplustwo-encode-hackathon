"""Cookbook ProblemEnv scoring against the current writer.

    uv run train/test_spreadsheet_env.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from train.spreadsheet_env import reward_from_scores, score_sample

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


def _pair(tmp: Path, n: int) -> dict:
    init = tmp / "init.xlsx"
    gold = tmp / "golden.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for i in range(1, n + 1):
        ws[f"A{i}"] = i
        ws[f"B{i}"] = i * 2
    wb.save(gold)
    for i in range(1, n + 1):
        ws[f"B{i}"] = None
    wb.save(init)
    return {
        "id": f"n{n}",
        "instruction": "double A into B",
        "init_xlsx": str(init),
        "golden_xlsx": str(gold),
        "answer_position": f"B1:B{n}",
        "answer_sheet": "Sheet1",
        "instruction_type": "Cell-Level Manipulation",
    }


def main() -> None:
    print("\n[1] reward + format")
    check(
        "cookbook format term + acc + pass bonus",
        abs(reward_from_scores({"format": 1.0, "cell_accuracy": 0.8, "pass": 1.0}) - 1.05) < 1e-9,
    )
    check(
        "bad JSON is format 0",
        score_sample({"id": "x"}, "not json", Path(tempfile.mkdtemp()), False)["format"] == 0.0,
    )

    print("\n[2] current writer: one formula fills the column")
    tmp = Path(tempfile.mkdtemp())
    task = _pair(tmp, 5)
    literals = score_sample(
        task,
        '{"cells":[{"cell":"B1","value":2},{"cell":"B2","value":4},{"cell":"B3","value":6},{"cell":"B4","value":8},{"cell":"B5","value":10}]}',
        tmp / "work",
        False,
    )
    check("literal dump passes", literals["pass"] == 1.0 and literals["format"] == 1.0, str(literals))
    formula = score_sample(task, '{"cells":[{"cell":"B1","value":"=A1*2"}]}', tmp / "work", False)
    check("formula JSON is valid format", formula["format"] == 1.0, str(formula))
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

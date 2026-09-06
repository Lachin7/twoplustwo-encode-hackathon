"""Checks for residual SFT explode + fail ranking (no Tinker).

    uv run train/test_residual_sft.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.build_residual_sft import explode_last_assistant, fail_priority

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


def test_explode() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": '{"tool":"python","code":"print(1)"}'},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": '{"tool":"done"}'},
    ]
    exploded = explode_last_assistant(messages)
    check("two prefixes", len(exploded) == 2)
    check("first ends on python", "python" in exploded[0][-1]["content"])
    check("second ends on done", "done" in exploded[1][-1]["content"])
    check("second keeps history", exploded[1][2]["role"] == "assistant")


def test_priority() -> None:
    near = {"cells": 10, "correct": 9, "mismatches": [{"actual": 1}]}
    empty = {"cells": 10, "correct": 0, "mismatches": [{"actual": None}]}
    xl = {"cells": 10, "correct": 2, "mismatches": [{"actual": "#VALUE!"}]}
    other = {"cells": 10, "correct": 3, "mismatches": [{"actual": 99}]}
    ranked = sorted([other, xl, empty, near], key=fail_priority)
    check("near first", ranked[0] is near)
    check("none second", ranked[1] is empty)
    check("xl third", ranked[2] is xl)


def main() -> None:
    print("test_residual_sft")
    test_explode()
    test_priority()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

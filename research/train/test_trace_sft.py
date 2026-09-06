"""Unit checks for rejection-SFT conversation rebuild (no Tinker).

    uv run train/test_trace_sft.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from agent import AGENT_SYSTEM
from common import FORMAT_HINT, SYSTEM_PROMPT
from train.build_trace_sft import (
    build,
    conversation_from_rows,
    passing_ids,
    uses_formula_or_tool,
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


def test_oneshot_adds_format_hint() -> None:
    rows = [
        {
            "step": 1,
            "prompt": "## Instruction\nadd A1+B1 into C1",
            "response": '{"cells":[{"cell":"C1","value":"=A1+B1"}]}',
        }
    ]
    messages = conversation_from_rows(rows)
    check("oneshot has 3 turns", messages is not None and len(messages) == 3)
    check("oneshot system", messages[0]["content"] == SYSTEM_PROMPT)
    check("oneshot user gets FORMAT_HINT", messages[1]["content"].endswith(FORMAT_HINT.strip()) or FORMAT_HINT in messages[1]["content"])
    check("oneshot assistant kept", "A1+B1" in messages[2]["content"])
    check("oneshot formula detected", uses_formula_or_tool(rows))


def test_oneshot_does_not_double_hint() -> None:
    prompt = "## Instruction\nfoo" + FORMAT_HINT
    rows = [{"prompt": prompt, "response": '{"cells":[]}'}]
    messages = conversation_from_rows(rows)
    check("no double FORMAT_HINT", messages[1]["content"].count("Reply with JSON only") == 1)


def test_agent_keeps_tool_loop() -> None:
    rows = [
        {
            "step": 1,
            "mode": "agent",
            "prompt": "## Instruction\nfill G2:G100",
            "response": '{"tool":"python","code":"print(wb.sheetnames)"}',
            "tool": "python",
        },
        {
            "step": 2,
            "mode": "agent",
            "prompt": "## Python result\n```\n['Sheet1']\n```",
            "response": '{"tool":"done"}',
            "tool": "done",
        },
    ]
    messages = conversation_from_rows(rows)
    check("agent 5 messages", messages is not None and len(messages) == 5)
    check("agent system", messages[0]["content"] == AGENT_SYSTEM)
    check("agent first user is instruction", "fill G2:G100" in messages[1]["content"])
    check("agent python turn", "python" in messages[2]["content"])
    check("agent tool result user", "Python result" in messages[3]["content"])
    check("agent done turn", "done" in messages[4]["content"])
    check("agent formula_or_tool", uses_formula_or_tool(rows))


def test_skip_empty_response() -> None:
    rows = [
        {"prompt": "hello", "response": None},
        {"prompt": "retry", "response": '{"cells":[]}'},
    ]
    messages = conversation_from_rows(rows)
    check("dropped empty first turn", messages is not None and len(messages) == 3)
    check("kept retry user", messages[1]["content"].startswith("retry"))


def test_empty_trace() -> None:
    check("empty rows -> None", conversation_from_rows([]) is None)
    check("no responses -> None", conversation_from_rows([{"prompt": "x", "response": ""}]) is None)


def test_passing_filter() -> None:
    results = {
        "items": [
            {"id": "1", "status": "graded", "pass": True, "cells": 2, "correct": 2},
            {"id": "2", "status": "graded", "pass": False, "cells": 10, "correct": 9},
            {"id": "3", "status": "error"},
        ]
    }
    kept = passing_ids(results, 1.0)
    check("pass only keeps winner", set(kept) == {"1"})
    kept_loose = passing_ids(results, 0.9)
    check("min-cell-acc keeps 90%", set(kept_loose) == {"1", "2"})


def test_build_drops_eval_and_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        traces = run / "traces"
        traces.mkdir()
        (traces / "keep-me.jsonl").write_text(
            json.dumps(
                {
                    "prompt": "task",
                    "response": '{"cells":[{"cell":"B1","value":"=A1*2"}]}',
                }
            )
            + "\n"
        )
        (traces / "eval-leak.jsonl").write_text(
            json.dumps({"prompt": "leak", "response": '{"cells":[]}'}) + "\n"
        )
        (traces / "fail.jsonl").write_text(
            json.dumps({"prompt": "fail", "response": '{"cells":[]}'}) + "\n"
        )
        results = {
            "items": [
                {"id": "keep-me", "status": "graded", "pass": True, "cells": 1, "correct": 1},
                {"id": "eval-leak", "status": "graded", "pass": True, "cells": 1, "correct": 1},
                {"id": "fail", "status": "graded", "pass": False, "cells": 1, "correct": 0},
            ]
        }
        conversations, manifest = build(
            run,
            results,
            {"keep-me", "eval-leak", "fail"},
            {"eval-leak"},
            1.0,
            10_000,
        )
        by_id = {r["id"]: r for r in manifest}
        check("kept one conversation", len(conversations) == 1)
        check("kept id marked", by_id["keep-me"]["kept"] is True)
        check("eval leak dropped", by_id["eval-leak"]["reason"] == "eval_leak")
        check("fail dropped", by_id["fail"]["reason"] == "not_pass")
        check("kept formula flag", by_id["keep-me"]["formula_or_tool"] is True)


def main() -> None:
    print("test_trace_sft")
    test_oneshot_adds_format_hint()
    test_oneshot_does_not_double_hint()
    test_agent_keeps_tool_loop()
    test_skip_empty_response()
    test_empty_trace()
    test_passing_filter()
    test_build_drops_eval_and_fails()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

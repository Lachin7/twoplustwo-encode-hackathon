"""Pair successful and failed base-agent first actions for conservative DPO.

The two independent train-set runs give outcome labels without extra sampling.
Only agent tasks where exactly one run passed are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.build_trace_sft import read_jsonl
from train.paths import TRAIN_IDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-a", default="ship/train-traces-base")
    p.add_argument("--run-b", default="ship/base-400-agent")
    p.add_argument("--out", default="data/sft/agent_first_action_dpo.jsonl")
    p.add_argument("--max-chars", type=int, default=64_000)
    return p.parse_args()


def results_by_id(run_dir: Path) -> dict[str, dict]:
    results = json.loads((run_dir / "results.json").read_text())
    return {str(row["id"]): row for row in results["items"]}


def first_agent_row(run_dir: Path, task_id: str) -> dict | None:
    rows = read_jsonl(run_dir / "traces" / f"{task_id}.jsonl")
    rows = [row for row in rows if row.get("mode") == "agent" and row.get("response")]
    return rows[0] if rows else None


def clean_chosen(row: dict) -> bool:
    return not row.get("error") and not str(row.get("tool_output") or "").startswith("ERROR:")


def main() -> None:
    args = parse_args()
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    result_a, result_b = results_by_id(run_a), results_by_id(run_b)
    train_ids = set(map(str, json.loads(Path(TRAIN_IDS).read_text())))
    pairs: list[dict] = []
    skipped = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for task_id in sorted(train_ids):
        a, b = result_a.get(task_id), result_b.get(task_id)
        if not a or not b or bool(a.get("pass")) == bool(b.get("pass")):
            continue
        row_a = first_agent_row(run_a, task_id)
        row_b = first_agent_row(run_b, task_id)
        if not row_a or not row_b:
            skip("not_agent_both")
            continue
        chosen, rejected = (row_a, row_b) if a.get("pass") else (row_b, row_a)
        if not clean_chosen(chosen):
            skip("chosen_first_action_error")
            continue
        prompt = str(chosen.get("prompt") or "")
        chosen_text = str(chosen.get("response") or "")
        rejected_text = str(rejected.get("response") or "")
        if not prompt or not chosen_text or not rejected_text or chosen_text == rejected_text:
            skip("empty_or_same")
            continue
        if len(prompt) + len(chosen_text) + len(rejected_text) > args.max_chars:
            skip("too_long")
            continue
        pairs.append(
            {
                "id": task_id,
                "comparison": {
                    "prompt_conversation": [
                        {
                            "role": "system",
                            "content": (
                                "You are a spreadsheet expert with a live Python/openpyxl workbook. "
                                "Reply with JSON tool calls and fill the requested answer cells."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "completion_A": [{"role": "assistant", "content": chosen_text}],
                    "completion_B": [{"role": "assistant", "content": rejected_text}],
                },
                "label": "A",
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"agent DPO pairs: {len(pairs)} -> {out}; skipped={skipped}")
    if not pairs:
        raise SystemExit("no DPO pairs")


if __name__ == "__main__":
    main()

"""Residual SFT: oracle gold on train fails + a small pass mix.

Pass-only imitation already dropped sheet-level. This trains the misses
(last assistant only) and keeps a few train passes so oneshot/agent format
does not forget. Never includes eval-80.

    uv run train/build_residual_sft.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import FORMAT_HINT, SYSTEM_PROMPT, build_prompt
from sb import load_dataset

from train.build_trace_sft import conversation_chars, conversation_from_rows, read_jsonl
from train.paths import (
    EVAL_IDS,
    RESIDUAL_SFT_JSONL,
    RESIDUAL_SFT_MANIFEST,
    SMALL_CELL_LIMIT,
    TRAIN_IDS,
)
from train.teacher_label import golden_assistant

MAX_CHARS = 16384 * 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="ship/base-400-agent/results.json")
    p.add_argument("--run-dir", default="ship/train-traces-base", help="traces for the pass mix")
    p.add_argument("--ids-file", default=str(TRAIN_IDS))
    p.add_argument("--eval-ids", default=str(EVAL_IDS))
    p.add_argument("--dataset-dir", default="data/spreadsheetbench_verified_400")
    p.add_argument("--out-jsonl", default=str(RESIDUAL_SFT_JSONL))
    p.add_argument("--out-manifest", default=str(RESIDUAL_SFT_MANIFEST))
    p.add_argument("--max-cells", type=int, default=SMALL_CELL_LIMIT)
    p.add_argument("--max-residual", type=int, default=60)
    p.add_argument("--max-oneshot-keep", type=int, default=20)
    p.add_argument("--max-agent-rows", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=MAX_CHARS)
    return p.parse_args()


def load_id_list(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError(f"unrecognized ids file {path}")


def explode_last_assistant(messages: list[dict]) -> list[list[dict]]:
    """One conversation per assistant turn so last-assistant loss is aligned."""
    out: list[list[dict]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            prefix = messages[: i + 1]
            if any(m.get("role") == "user" for m in prefix):
                out.append(prefix)
    return out


def fail_priority(item: dict) -> tuple[int, float]:
    """Near-miss and empty writes first; then other graded fails."""
    cells = item.get("cells") or 0
    acc = ((item.get("correct") or 0) / cells) if cells else 0.0
    mm = item.get("mismatches") or []
    first = mm[0] if mm else {}
    actual = str(first.get("actual"))
    if acc >= 0.9:
        bucket = 0
    elif actual in ("None", "null"):
        bucket = 1
    elif any(err in actual for err in ("#VALUE", "#REF", "#NAME")):
        bucket = 2
    else:
        bucket = 3
    return (bucket, -acc)


def oracle_conversation(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(task) + FORMAT_HINT},
        {"role": "assistant", "content": golden_assistant(task)},
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    train_ids = set(load_id_list(Path(args.ids_file)))
    eval_ids = set(load_id_list(Path(args.eval_ids)))
    train_ids -= eval_ids
    results = json.loads(Path(args.results).read_text())
    by_id = {str(item["id"]): item for item in results.get("items") or []}
    tasks = {t["id"]: t for t in load_dataset(Path(args.dataset_dir))}
    traces_dir = Path(args.run_dir) / "traces"

    conversations: list[dict] = []
    manifest: list[dict] = []

    fails = []
    for task_id in train_ids:
        item = by_id.get(task_id)
        if not item or item.get("status") != "graded" or item.get("pass"):
            continue
        cells = item.get("cells") or 0
        if cells <= 0 or cells > args.max_cells:
            continue
        if task_id not in tasks or not tasks[task_id].get("golden_xlsx"):
            continue
        fails.append(item)
    fails.sort(key=fail_priority)

    residual_n = 0
    for item in fails:
        if residual_n >= args.max_residual:
            break
        task_id = str(item["id"])
        rec = {"id": task_id, "kept": False, "kind": "residual_oracle", "cells": item.get("cells")}
        try:
            messages = oracle_conversation(tasks[task_id])
        except Exception as e:
            rec["reason"] = f"oracle_error:{e}"[:200]
            manifest.append(rec)
            continue
        n_chars = conversation_chars(messages)
        if n_chars > args.max_chars:
            rec["reason"] = f"too_long:{n_chars}"
            manifest.append(rec)
            continue
        conversations.append({"messages": messages})
        rec.update({"kept": True, "reason": "residual", "n_chars": n_chars, "n_assistant": 1})
        manifest.append(rec)
        residual_n += 1

    oneshot_n = 0
    agent_n = 0
    for task_id in sorted(train_ids):
        item = by_id.get(task_id)
        if not item or not item.get("pass"):
            continue
        rows = read_jsonl(traces_dir / f"{task_id}.jsonl")
        messages = conversation_from_rows(rows)
        if messages is None:
            continue
        agent = any(r.get("mode") == "agent" for r in rows)
        exploded = explode_last_assistant(messages)
        if agent:
            exploded = exploded[-2:] if len(exploded) > 2 else exploded
        for prefix in exploded:
            if agent and agent_n >= args.max_agent_rows:
                break
            if (not agent) and oneshot_n >= args.max_oneshot_keep:
                break
            n_chars = conversation_chars(prefix)
            if n_chars > args.max_chars:
                continue
            conversations.append({"messages": prefix})
            manifest.append(
                {
                    "id": task_id,
                    "kept": True,
                    "kind": "agent_reg" if agent else "oneshot_keep",
                    "reason": "pass_mix",
                    "n_chars": n_chars,
                    "n_assistant": 1,
                }
            )
            if agent:
                agent_n += 1
            else:
                oneshot_n += 1
        if oneshot_n >= args.max_oneshot_keep and agent_n >= args.max_agent_rows:
            break

    return conversations, manifest


def main() -> None:
    args = parse_args()
    conversations, manifest = build(args)
    out_jsonl = Path(args.out_jsonl)
    out_manifest = Path(args.out_manifest)
    write_jsonl(out_jsonl, conversations)
    write_jsonl(out_manifest, manifest)
    kept = [r for r in manifest if r.get("kept")]
    kinds = {}
    for r in kept:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    print(f"residual sft {len(conversations)} rows {kinds} -> {out_jsonl}")
    if not conversations:
        raise SystemExit("no residual rows")


if __name__ == "__main__":
    main()

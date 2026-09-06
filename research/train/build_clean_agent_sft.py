"""Build pass-only agent SFT with loss only on successful workbook mutations.

Unlike plain ``all_assistant_messages``, this keeps inspection/error turns as
context but masks their loss.  The targets are valid cell answers, successful
Python mutations, and the final ``done`` after a mutation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from agent import AGENT_SYSTEM, parse_agent_turn
from train.build_trace_sft import load_id_list, read_jsonl
from train.paths import EVAL_IDS, TRAIN_IDS

MUTATION_PATTERNS = (
    r"\[[^\]]+\]\s*=",
    r"\.value\s*=",
    r"\.cell\([^)]*,\s*value\s*=",
    r"\.(?:append|delete_rows|delete_cols|insert_rows|insert_cols|move_range|merge_cells|unmerge_cells)\(",
    r"\.(?:create_sheet|remove)\(",
    r"\.(?:title|number_format|font|fill|border|alignment)\s*=",
)
MUTATION_RE = re.compile("|".join(MUTATION_PATTERNS))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="ship/train-traces-base")
    p.add_argument("--results", default="ship/train-traces-base/results.json")
    p.add_argument("--ids-file", default=str(TRAIN_IDS))
    p.add_argument("--eval-ids", default=str(EVAL_IDS))
    p.add_argument("--out-jsonl", default="data/sft/sft_train_agent_clean.jsonl")
    p.add_argument("--out-manifest", default="data/sft/manifest_train_agent_clean.jsonl")
    p.add_argument("--max-chars", type=int, default=65536)
    return p.parse_args()


def is_clean_target(row: dict, *, mutation_seen: bool) -> tuple[bool, bool, str]:
    """Return (trainable, mutation_seen_after, reason)."""
    response = str(row.get("response") or "").strip()
    if not response:
        return False, mutation_seen, "empty"
    if row.get("error"):
        return False, mutation_seen, "trace_error"
    if str(row.get("tool_output") or "").startswith("ERROR:"):
        return False, mutation_seen, "tool_error"
    try:
        kind, payload = parse_agent_turn(response)
    except Exception:
        return False, mutation_seen, "parse_error"
    if kind == "answer":
        return True, True, "cells"
    if kind == "python":
        mutates = bool(MUTATION_RE.search(str(payload)))
        return mutates, mutation_seen or mutates, "python_mutation" if mutates else "python_inspect"
    if kind == "done":
        return mutation_seen, mutation_seen, "done_after_mutation" if mutation_seen else "early_done"
    return False, mutation_seen, "other"


def build_conversation(rows: list[dict]) -> tuple[list[dict], dict] | None:
    usable = [row for row in rows if str(row.get("response") or "").strip()]
    if not usable or not any(row.get("mode") == "agent" for row in usable):
        return None
    messages = [{"role": "system", "content": AGENT_SYSTEM, "trainable": False}]
    mutation_seen = False
    reasons: list[str] = []
    n_targets = 0
    for row in usable:
        prompt = str(row.get("prompt") or "")
        if prompt:
            messages.append({"role": "user", "content": prompt, "trainable": False})
        trainable, mutation_seen, reason = is_clean_target(row, mutation_seen=mutation_seen)
        reasons.append(reason)
        n_targets += int(trainable)
        messages.append(
            {
                "role": "assistant",
                "content": str(row["response"]),
                "trainable": trainable,
            }
        )
    if not n_targets:
        return None
    return messages, {"n_targets": n_targets, "target_reasons": reasons}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    results = json.loads(Path(args.results).read_text())
    passed = {
        str(item["id"])
        for item in results.get("items", [])
        if item.get("status") == "graded" and item.get("pass")
    }
    train_ids = set(load_id_list(Path(args.ids_file)))
    eval_ids = set(load_id_list(Path(args.eval_ids)))
    allowed = passed & train_ids - eval_ids
    run_dir = Path(args.run_dir)
    conversations: list[dict] = []
    manifest: list[dict] = []
    for task_id in sorted(allowed):
        trace = read_jsonl(run_dir / "traces" / f"{task_id}.jsonl")
        built = build_conversation(trace)
        if built is None:
            continue
        messages, meta = built
        n_chars = sum(len(str(message.get("content") or "")) for message in messages)
        if n_chars > args.max_chars:
            manifest.append({"id": task_id, "kept": False, "reason": "too_long", "n_chars": n_chars})
            continue
        conversations.append({"messages": messages})
        manifest.append(
            {
                "id": task_id,
                "kept": True,
                "n_chars": n_chars,
                **meta,
            }
        )
    write_jsonl(Path(args.out_jsonl), conversations)
    write_jsonl(Path(args.out_manifest), manifest)
    targets = sum(row.get("n_targets", 0) for row in manifest if row.get("kept"))
    print(f"clean agent SFT: {len(conversations)} conversations, {targets} target turns")
    if not conversations:
        raise SystemExit("no clean agent conversations")


if __name__ == "__main__":
    main()

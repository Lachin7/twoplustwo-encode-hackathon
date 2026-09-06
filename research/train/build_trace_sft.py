"""Build rejection-SFT JSONL from a scored train infer run.

Keep conversations that *pass* after LibreOffice. Reconstruct the same
system/user/assistant turns the model actually saw (formulas, python, done).
Never include eval-80 ids.

    uv run train/build_trace_sft.py \\
        --run-dir ship/train-traces-base \\
        --results ship/train-traces-base/results.json \\
        --ids-file data/splits/train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from agent import AGENT_SYSTEM
from common import FORMAT_HINT, SYSTEM_PROMPT

from train.paths import (
    EVAL_IDS,
    TRACE_SFT_JSONL,
    TRACE_SFT_MANIFEST,
    TRAIN_IDS,
)

MAX_CHARS = 16384 * 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="infer out-dir with traces/<id>.jsonl")
    p.add_argument("--results", help="evaluate.py JSON (default: <run-dir>/results.json)")
    p.add_argument("--ids-file", default=str(TRAIN_IDS), help="only keep these ids (train)")
    p.add_argument("--eval-ids", default=str(EVAL_IDS), help="never keep these ids")
    p.add_argument("--out-jsonl", default=str(TRACE_SFT_JSONL))
    p.add_argument("--out-manifest", default=str(TRACE_SFT_MANIFEST))
    p.add_argument(
        "--min-cell-acc",
        type=float,
        default=1.0,
        help="keep if pass, or cell_accuracy >= this (1.0 = pass only)",
    )
    p.add_argument("--max-chars", type=int, default=MAX_CHARS)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_id_list(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "train_ids" in data:
        return [str(x) for x in data["train_ids"]]
    raise ValueError(f"unrecognized ids file {path}")


def passing_ids(results: dict, min_cell_acc: float) -> dict[str, dict]:
    kept: dict[str, dict] = {}
    for item in results.get("items") or []:
        task_id = str(item.get("id") or "")
        if not task_id:
            continue
        if item.get("status") != "graded":
            continue
        cells = item.get("cells") or 0
        correct = item.get("correct") or 0
        acc = (correct / cells) if cells else 0.0
        if item.get("pass") or acc >= min_cell_acc:
            kept[task_id] = {**item, "cell_accuracy": acc}
    return kept


def _oneshot_user(prompt: str) -> str:
    if FORMAT_HINT in prompt:
        return prompt
    return prompt + FORMAT_HINT


def conversation_from_rows(rows: list[dict]) -> list[dict] | None:
    """Rebuild the chat the sampler saw. Agent traces store each last user turn."""
    usable = [r for r in rows if (r.get("response") or "").strip()]
    if not usable:
        return None
    agent = any(r.get("mode") == "agent" for r in rows)
    messages = [{"role": "system", "content": AGENT_SYSTEM if agent else SYSTEM_PROMPT}]
    for row in usable:
        prompt = row.get("prompt") or ""
        if prompt:
            messages.append(
                {"role": "user", "content": prompt if agent else _oneshot_user(prompt)}
            )
        messages.append({"role": "assistant", "content": row["response"]})
    if not any(m["role"] == "assistant" for m in messages):
        return None
    if not any(m["role"] == "user" for m in messages):
        return None
    return messages


def conversation_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def uses_formula_or_tool(rows: list[dict]) -> bool:
    for row in rows:
        text = row.get("response") or ""
        if row.get("tool") in ("python", "done", "cells"):
            if row.get("tool") in ("python", "done"):
                return True
        if '"tool"' in text and ("python" in text or "done" in text):
            return True
        if '"value": "=' in text or '"value":"=' in text:
            return True
    return False


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    run_dir: Path,
    results: dict,
    keep_ids: set[str],
    eval_ids: set[str],
    min_cell_acc: float,
    max_chars: int,
) -> tuple[list[dict], list[dict]]:
    passed = passing_ids(results, min_cell_acc)
    conversations: list[dict] = []
    manifest: list[dict] = []
    traces_dir = run_dir / "traces"
    for task_id in sorted(keep_ids):
        rec = {
            "id": task_id,
            "kept": False,
            "reason": None,
            "n_turns": 0,
            "mode": None,
            "formula_or_tool": False,
        }
        if task_id in eval_ids:
            rec["reason"] = "eval_leak"
            manifest.append(rec)
            continue
        item = passed.get(task_id)
        if item is None:
            rec["reason"] = "not_pass"
            manifest.append(rec)
            continue
        rows = read_jsonl(traces_dir / f"{task_id}.jsonl")
        rec["n_turns"] = len(rows)
        rec["mode"] = "agent" if any(r.get("mode") == "agent" for r in rows) else "oneshot"
        rec["formula_or_tool"] = uses_formula_or_tool(rows)
        messages = conversation_from_rows(rows)
        if messages is None:
            rec["reason"] = "no_conversation"
            manifest.append(rec)
            continue
        n_chars = conversation_chars(messages)
        if n_chars > max_chars:
            rec["reason"] = f"too_long:{n_chars}"
            manifest.append(rec)
            continue
        conversations.append({"messages": messages})
        rec["kept"] = True
        rec["reason"] = "pass"
        rec["cells"] = item.get("cells")
        rec["correct"] = item.get("correct")
        rec["n_chars"] = n_chars
        rec["n_assistant"] = sum(1 for m in messages if m["role"] == "assistant")
        manifest.append(rec)
    return conversations, manifest


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    results_path = Path(args.results) if args.results else run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"missing {results_path}; run evaluate.py first")
    results = json.loads(results_path.read_text())
    keep_ids = set(load_id_list(Path(args.ids_file)))
    eval_ids = set(load_id_list(Path(args.eval_ids)))
    overlap = keep_ids & eval_ids
    if overlap:
        print(f"dropping {len(overlap)} eval ids from keep set")
        keep_ids -= eval_ids
    conversations, manifest = build(
        run_dir,
        results,
        keep_ids,
        eval_ids,
        args.min_cell_acc,
        args.max_chars,
    )
    out_jsonl = Path(args.out_jsonl)
    out_manifest = Path(args.out_manifest)
    write_jsonl(out_jsonl, conversations)
    write_jsonl(out_manifest, manifest)
    kept = sum(1 for r in manifest if r["kept"])
    agent = sum(1 for r in manifest if r["kept"] and r["mode"] == "agent")
    tools = sum(1 for r in manifest if r["kept"] and r["formula_or_tool"])
    print(
        f"trace sft {kept}/{len(manifest)} kept "
        f"(agent={agent} formula_or_tool={tools}) -> {out_jsonl}"
    )
    if not conversations:
        raise SystemExit("no passing traces to train on")


if __name__ == "__main__":
    main()

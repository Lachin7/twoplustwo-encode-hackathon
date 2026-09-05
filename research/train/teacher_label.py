"""Label train-small tasks with Gemini (Google AI Studio). Keep traces that pass the scorer.

    uv run train/teacher_label.py --samples 3
    uv run train/teacher_label.py --oracle
    uv run train/teacher_label.py --resume   # skip ids already kept; append

Needs GOOGLE_API_KEY or GEMINI_API_KEY in .env unless --oracle.
Prefer the wrappers in train/scripts/ for smoke / full / status / merge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import FORMAT_HINT, SYSTEM_PROMPT, SpreadsheetAnswer, build_prompt, load_env, parse_answer, write_output
from evaluate import score_task
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from sb import DEFAULT_DATASET, load_answer_values, load_dataset, transform_value

from train.paths import (
    GEMINI_SFT_DIR,
    GEMINI_SFT_JSONL,
    GEMINI_SFT_MANIFEST,
    SFT_DIR,
    SFT_JSONL,
    SFT_MANIFEST,
    SPLIT_META,
)

DEFAULT_TEACHER = "gemini-2.5-pro"
_IO_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--split", default=str(SPLIT_META))
    p.add_argument(
        "--model",
        default=os.environ.get("TEACHER_MODEL", DEFAULT_TEACHER),
        help="Gemini model id, e.g. gemini-2.5-pro or gemini-2.5-flash",
    )
    p.add_argument("--samples", type=int, default=3, help="teacher samples per task")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=1.0,
        help="keep if pass, or cell_accuracy >= this when < 1",
    )
    p.add_argument("--all-train", action="store_true", help="include huge train tasks (default: train-small only)")
    p.add_argument("--oracle", action="store_true", help="write golden JSON as assistant (no teacher API)")
    p.add_argument("--ids", help="comma-separated train ids to label")
    p.add_argument("--n", type=int, help="label only the first N selected ids (after --ids / train-small)")
    p.add_argument("--resume", action="store_true", help="skip already-kept ids; append to jsonl/manifest")
    p.add_argument("--out-jsonl", default=None, help="default: data/sft/gemini/... (or oracle path with --oracle)")
    p.add_argument("--out-manifest", default=None)
    return p.parse_args()


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.oracle:
        return Path(args.out_jsonl or SFT_JSONL), Path(args.out_manifest or SFT_MANIFEST)
    return Path(args.out_jsonl or GEMINI_SFT_JSONL), Path(args.out_manifest or GEMINI_SFT_MANIFEST)


def teacher_model_id(model: str) -> str:
    """Accept gemini-2.5-pro, google:gemini-2.5-pro, or legacy google/gemini-2.5-pro."""
    if model.startswith("google:") or model.startswith("google-gla:"):
        return model.replace("google-gla:", "google:", 1)
    if model.startswith("google/"):
        return f"google:{model.split('/', 1)[1]}"
    return f"google:{model}"


def jsonable(value):
    return transform_value(value)


def golden_assistant(task: dict) -> str:
    gold = load_answer_values(task["golden_xlsx"], task)
    cells = [{"cell": coord, "value": jsonable(value)} for (_sheet, coord), value in gold.items()]
    return json.dumps({"cells": cells}, ensure_ascii=False)


def conversation(task: dict, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task) + FORMAT_HINT},
            {"role": "assistant", "content": assistant},
        ]
    }


def score_text(task: dict, text: str) -> dict:
    answer = parse_answer(text)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{task['id']}.xlsx"
        write_output(task, answer, out)
        result = score_task(task, str(out), recalc=True, work_dir=tmp)
    cells = result.get("cells") or 0
    correct = result.get("correct") or 0
    accuracy = (correct / cells) if cells else 0.0
    return {
        **result,
        "cell_accuracy": accuracy,
        "assistant": parse_answer(text).model_dump_json(),
    }


def keep_result(result: dict, min_accuracy: float) -> bool:
    if result.get("status") != "graded":
        return False
    if result.get("pass"):
        return True
    return min_accuracy < 1.0 and result.get("cell_accuracy", 0) >= min_accuracy


def load_kept_ids(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    kept: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kept") or (row.get("source") == "oracle" and row.get("pass")):
            kept.add(row["id"])
    return kept


async def label_task(agent: Agent, task: dict, samples: int, min_accuracy: float) -> dict | None:
    best = None
    prompt = build_prompt(task) + FORMAT_HINT
    for _ in range(samples):
        try:
            result = await agent.run(prompt)
            text = result.output.model_dump_json()
            scored = score_text(task, text)
            scored["assistant"] = text
        except Exception as e:
            scored = {"status": "error", "error": str(e)[:200], "cell_accuracy": 0.0, "pass": False}
        if best is None or scored.get("cell_accuracy", 0) > best.get("cell_accuracy", 0):
            best = scored
        if scored.get("pass"):
            break
    if best is None or not keep_result(best, min_accuracy):
        return None
    return best


def selected_train_ids(meta: dict, args: argparse.Namespace) -> list[str]:
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    elif args.all_train:
        ids = list(meta["train_ids"])
    else:
        ids = list(meta["train_small_ids"])
    if args.n is not None:
        ids = ids[: args.n]
    return ids


def write_oracle(tasks: list[dict], out_jsonl: Path, out_manifest: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as jf, out_manifest.open("w", encoding="utf-8") as mf:
        for task in tasks:
            assistant = golden_assistant(task)
            jf.write(json.dumps(conversation(task, assistant), ensure_ascii=False) + "\n")
            n = len(json.loads(assistant)["cells"])
            mf.write(
                json.dumps(
                    {
                        "id": task["id"],
                        "source": "oracle",
                        "kept": True,
                        "pass": True,
                        "cell_accuracy": 1.0,
                        "n_cells": n,
                        "type": task.get("instruction_type"),
                    }
                )
                + "\n"
            )
    print(f"oracle sft {len(tasks)} -> {out_jsonl}")


def append_record(
    out_jsonl: Path,
    out_manifest: Path,
    task: dict,
    scored: dict | None,
    model: str,
) -> bool:
    """Write one task result. Returns True if kept."""
    record = {
        "id": task["id"],
        "source": model,
        "type": task.get("instruction_type"),
    }
    with _IO_LOCK:
        with out_manifest.open("a", encoding="utf-8") as mf:
            if scored is None:
                record.update({"kept": False, "pass": False, "cell_accuracy": 0.0})
                mf.write(json.dumps(record) + "\n")
                print(f"{task['id']:<8} drop")
                return False
            assistant = scored.get("assistant")
            if not assistant:
                record.update({"kept": False, "error": "no assistant"})
                mf.write(json.dumps(record) + "\n")
                return False
            with out_jsonl.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(conversation(task, assistant), ensure_ascii=False) + "\n")
            record.update(
                {
                    "kept": True,
                    "pass": scored.get("pass"),
                    "cell_accuracy": scored.get("cell_accuracy"),
                    "n_cells": scored.get("cells"),
                }
            )
            mf.write(json.dumps(record) + "\n")
            print(f"{task['id']:<8} keep acc={scored.get('cell_accuracy'):.3f} pass={scored.get('pass')}")
            return True


async def write_teacher(tasks: list[dict], args: argparse.Namespace, out_jsonl: Path, out_manifest: Path) -> None:
    model = teacher_model_id(args.model)
    agent = Agent(
        model=model,
        output_type=SpreadsheetAnswer,
        system_prompt=SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.4),
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        already = load_kept_ids(out_manifest)
        before = len(tasks)
        tasks = [t for t in tasks if t["id"] not in already]
        print(f"resume: skip {before - len(tasks)} kept, label {len(tasks)}")
        out_jsonl.touch(exist_ok=True)
        out_manifest.touch(exist_ok=True)
    else:
        out_jsonl.write_text("", encoding="utf-8")
        out_manifest.write_text("", encoding="utf-8")

    if not tasks:
        print(f"nothing to label -> {out_jsonl}")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    kept = 0

    async def one(task: dict) -> bool:
        async with semaphore:
            scored = await label_task(agent, task, args.samples, args.min_accuracy)
        return append_record(out_jsonl, out_manifest, task, scored, model)

    results = await asyncio.gather(*(one(t) for t in tasks))
    kept = sum(1 for ok in results if ok)
    print(f"teacher sft {kept}/{len(tasks)} this run -> {out_jsonl}")


def main() -> None:
    load_env()
    args = parse_args()
    meta = json.loads(Path(args.split).read_text())
    order = selected_train_ids(meta, args)
    ids = set(order)
    by_id = {t["id"]: t for t in load_dataset(Path(args.dataset_dir)) if t["id"] in ids}
    tasks = [by_id[i] for i in order if i in by_id]
    out_jsonl, out_manifest = resolve_outputs(args)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    GEMINI_SFT_DIR.mkdir(parents=True, exist_ok=True)
    if args.oracle:
        write_oracle(tasks, out_jsonl, out_manifest)
        return
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("GOOGLE_API_KEY (or GEMINI_API_KEY) missing; use --oracle or set the key")
    asyncio.run(write_teacher(tasks, args, out_jsonl, out_manifest))


if __name__ == "__main__":
    main()

"""Compare one-shot vs auto agent on a frozen train slice.

    uv run baseline/verify_agent.py
    uv run baseline/verify_agent.py --ids 39903,97-36 --concurrency 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_env
from evaluate import score
from sb import DEFAULT_DATASET, load_dataset, read_jsonl
from train.paths import ROOT, SPLIT_META

# Train ids from base-400: small passes (must not regress) + large fails (hope to help).
DEFAULT_IDS = [
    "39903",  # 5-cell pass
    "40478",  # 3-cell pass
    "384-4",  # 7-cell sheet pass
    "280-17",  # 24-cell sheet pass — agent path, must not break
    "97-36",  # 22-cell sheet fail, near miss
    "49667",  # 60-cell fail, near miss
    "48643",  # 39-cell fail
    "22-47",  # 27-cell sheet fail
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default=",".join(DEFAULT_IDS))
    p.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--renderer", default="qwen3_8_xhigh_reasoning")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--out-root", default=str(ROOT / "ship" / "verify-agent"))
    p.add_argument("--skip-oneshot", action="store_true", help="reuse existing oneshot/ outputs")
    p.add_argument("--auto-name", default="auto", help="subdir for this auto run (e.g. auto-v2)")
    p.add_argument("--compare-only", action="store_true", help="score existing dirs, no infer")
    return p.parse_args()


def infer(out: Path, ids: str, agent: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "baseline" / "tinker_predict.py"),
        "--out-dir",
        str(out),
        "--dataset-dir",
        str(DEFAULT_DATASET),
        "--base-model",
        args.base_model,
        "--renderer",
        args.renderer,
        "--concurrency",
        str(args.concurrency),
        "--ids",
        ids,
        "--agent",
        agent,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    load_env()
    args = parse_args()
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    meta = json.loads(Path(SPLIT_META).read_text())
    train = set(meta["train_ids"])
    leaked = [i for i in ids if i not in train]
    if leaked:
        raise SystemExit(f"refusing eval/unknown ids (keep this on train): {leaked}")
    root = Path(args.out_root)
    oneshot_dir = root / "oneshot"
    auto_dir = root / args.auto_name
    if not args.compare_only:
        if not args.skip_oneshot:
            infer(oneshot_dir, ",".join(ids), "off", args)
        infer(auto_dir, ",".join(ids), "auto", args)

    tasks = [t for t in load_dataset(DEFAULT_DATASET) if t["id"] in set(ids)]
    by_id = {t["id"]: t for t in tasks}
    order = [i for i in ids if i in by_id]
    tasks = [by_id[i] for i in order]
    s1, items1 = score(read_jsonl(oneshot_dir / "predictions.jsonl"), tasks, recalc=True, predictions_path=oneshot_dir / "predictions.jsonl")
    s2, items2 = score(read_jsonl(auto_dir / "predictions.jsonl"), tasks, recalc=True, predictions_path=auto_dir / "predictions.jsonl")
    a1 = {i["id"]: i for i in items1}
    a2 = {i["id"]: i for i in items2}

    prev_dir = root / "auto"
    prev_items = {}
    s_prev = None
    if prev_dir != auto_dir and (prev_dir / "predictions.jsonl").exists():
        s_prev, items_prev = score(
            read_jsonl(prev_dir / "predictions.jsonl"), tasks, recalc=True, predictions_path=prev_dir / "predictions.jsonl"
        )
        prev_items = {i["id"]: i for i in items_prev}

    print(f"\n{'id':<8} {'n':>4} {'oneshot':<8} {args.auto_name:<10} {'vs1':<6} {'prev':<8} {'vsp'}")
    damaged = []
    helped = []
    vs_prev_hurt = []
    vs_prev_help = []
    for i in order:
        p1 = bool(a1[i].get("pass"))
        p2 = bool(a2[i].get("pass"))
        n = a1[i].get("cells") or 0
        mark = "help" if p2 and not p1 else ("hurt" if p1 and not p2 else "same")
        prev_p = bool(prev_items[i].get("pass")) if i in prev_items else None
        prev_mark = ""
        if prev_p is True and not p2:
            vs_prev_hurt.append(i)
            prev_mark = "hurt"
        elif prev_p is False and p2:
            vs_prev_help.append(i)
            prev_mark = "help"
        elif prev_p is not None:
            prev_mark = "same"
        print(f"{i:<8} {n:>4} {str(p1):<8} {str(p2):<10} {mark:<6} {str(prev_p):<8} {prev_mark}")
        if p1 and not p2:
            damaged.append(i)
        if p2 and not p1:
            helped.append(i)
    print()
    print("oneshot", json.dumps(s1))
    print(f"{args.auto_name:<8}", json.dumps(s2))
    if s_prev is not None:
        print("auto    ", json.dumps(s_prev))
    print(f"helped {helped or '-'}  damaged {damaged or '-'}")
    if prev_items:
        print(f"vs prev auto: helped {vs_prev_help or '-'}  damaged {vs_prev_hurt or '-'}")
    summary = {
        "oneshot": s1,
        args.auto_name: s2,
        "prev_auto": s_prev,
        "helped": helped,
        "damaged": damaged,
        "vs_prev_helped": vs_prev_help,
        "vs_prev_damaged": vs_prev_hurt,
        "ids": order,
    }
    root.mkdir(parents=True, exist_ok=True)
    out_json = root / (f"compare-{args.auto_name}.json" if args.auto_name != "auto" else "compare.json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    if damaged:
        raise SystemExit("auto damaged train tasks that one-shot passed")
    if vs_prev_hurt:
        raise SystemExit(f"{args.auto_name} damaged tasks previous auto passed: {vs_prev_hurt}")
    if s2.get("pass_rate", 0) < s1.get("pass_rate", 0):
        raise SystemExit("auto pass_rate below one-shot")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()

"""Run tinker_predict on a split or the full 400, then evaluate.

    uv run train/run_infer.py --ids-file data/splits/eval.json --out-dir ship/eval80
    uv run train/run_infer.py --all --out-dir ship --results ship/results.json
    uv run train/run_infer.py --all --model-path tinker://.../sampler_weights/final --out-dir ship
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import load_env
from sb import DEFAULT_DATASET

from train.paths import EVAL_IDS, ROOT

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RENDERER = "qwen3_8_xhigh_reasoning"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(ROOT / "ship"))
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--model-path")
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    p.add_argument("--ids-file", default=str(EVAL_IDS), help="JSON list of ids")
    p.add_argument("--ids", help="comma-separated ids (overrides --ids-file)")
    p.add_argument("--all", action="store_true", help="run every task in the dataset")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--results", help="write evaluate.py JSON here")
    p.add_argument("--no-recalc", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--agent", choices=("off", "auto", "always"), default="auto")
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "baseline" / "tinker_predict.py"),
        "--out-dir", str(out),
        "--dataset-dir", args.dataset_dir,
        "--base-model", args.base_model,
        "--renderer", args.renderer,
        "--concurrency", str(args.concurrency),
        "--max-tokens", str(args.max_tokens),
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.resume:
        cmd.append("--resume")
    cmd.extend(["--agent", args.agent])
    if not args.all:
        if args.ids:
            cmd.extend(["--ids", args.ids])
        else:
            ids = json.loads(Path(args.ids_file).read_text())
            cmd.extend(["--ids", ",".join(ids)])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    eval_cmd = [
        sys.executable,
        str(ROOT / "evaluate.py"),
        "--predictions", str(out / "predictions.jsonl"),
        "--dataset-dir", args.dataset_dir,
    ]
    if args.all:
        eval_cmd.append("--all")
    if args.no_recalc:
        eval_cmd.append("--no-recalc")
    results = Path(args.results) if args.results else out / "results.json"
    eval_cmd.extend(["--out", str(results)])
    print(" ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=True, cwd=str(ROOT))
    print(f"wrote {results}")


if __name__ == "__main__":
    main()

"""Scorer RL from an SFT checkpoint.

    uv run train/rl_tinker.py --load-checkpoint tinker://<sft>/weights/final
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import load_env
from sb import DEFAULT_DATASET
from tinker_cookbook.rl import train

from train.paths import ROOT, SPLIT_META
from train.spreadsheet_env import SpreadsheetDatasetBuilder

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RENDERER = "qwen3_8_xhigh_reasoning"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--load-checkpoint", required=True, help="SFT tinker://.../weights/final")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--split", default=str(SPLIT_META))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--groups-per-batch", type=int, default=16)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--recalc", action="store_true")
    p.add_argument("--log-path", default=str(ROOT / "train" / "logs" / "rl"))
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    work_dir = Path(args.log_path) / "reward_xlsx"
    builder = SpreadsheetDatasetBuilder(
        split_path=args.split,
        dataset_dir=args.dataset_dir,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        work_dir=str(work_dir),
        recalc=args.recalc,
        small_only=True,
    )
    config = train.Config(
        learning_rate=args.lr,
        dataset_builder=builder,
        model_name=args.model,
        recipe_name="recipe_spreadsheet_rl",
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        load_checkpoint_path=args.load_checkpoint,
        renderer_name=args.renderer,
        lora_rank=args.lora_rank,
        eval_every=0,
        save_every=10,
        loss_fn="importance_sampling",
        max_steps=args.max_steps,
    )
    print(f"rl {args.model} from {args.load_checkpoint} lr={args.lr}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()

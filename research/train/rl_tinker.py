"""Scorer RL from base or an SFT checkpoint (cookbook importance_sampling).

    # Overnight A: from base + KL so we don't forget formulas/agent
    uv run train/rl_tinker.py --from-base --recalc --kl-penalty-coef 0.05 \\
        --group-size 8 --groups-per-batch 16 --max-steps 50 \\
        --log-path train/logs/rl-base-kl
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
from tinker_cookbook.rl.train import KLReferenceConfig, Config
from tinker_cookbook.rl import train as rl_train

from train.paths import ROOT, SPLIT_META
from train.spreadsheet_env import SpreadsheetDatasetBuilder

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RENDERER = "qwen3_8_xhigh_reasoning"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--load-checkpoint",
        help="tinker://.../weights/final. Omit or pass --from-base to start from the base model.",
    )
    p.add_argument(
        "--from-base",
        action="store_true",
        help="ignore --load-checkpoint and RL the untuned base model",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--split", default=str(SPLIT_META))
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--groups-per-batch", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--recalc", action="store_true")
    p.add_argument("--kl-penalty-coef", type=float, default=0.0)
    p.add_argument(
        "--pool",
        choices=("recommended", "train_small"),
        default="recommended",
        help="recommended = 912-unseen + train cases 2–3 (no eval, no SFT case-1)",
    )
    p.add_argument("--log-path", default=str(ROOT / "train" / "logs" / "rl"))
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    checkpoint = None if args.from_base else args.load_checkpoint
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
        pool=args.pool,
    )
    kl_ref = None
    if args.kl_penalty_coef > 0:
        kl_ref = KLReferenceConfig(base_model=args.model, load_checkpoint_path=None)
    config = Config(
        learning_rate=args.lr,
        dataset_builder=builder,
        model_name=args.model,
        recipe_name="recipe_spreadsheet_rl",
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        load_checkpoint_path=checkpoint,
        renderer_name=args.renderer,
        lora_rank=args.lora_rank,
        eval_every=0,
        save_every=args.save_every,
        loss_fn="importance_sampling",
        max_steps=args.max_steps,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_reference_config=kl_ref,
        remove_constant_reward_groups=True,
    )
    print(
        f"rl {args.model} pool={args.pool} from={checkpoint or 'base'} "
        f"lr={args.lr} g={args.group_size}x{args.groups_per_batch} "
        f"kl={args.kl_penalty_coef} recalc={args.recalc}"
    )
    asyncio.run(rl_train.main(config))


if __name__ == "__main__":
    main()

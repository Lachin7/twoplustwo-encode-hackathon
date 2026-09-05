"""LoRA SFT of Qwen3.8-27B on spreadsheet_sft.jsonl.

    uv run train/sft_tinker.py
    uv run train/sft_tinker.py --jsonl data/sft/spreadsheet_sft.jsonl --lora-rank 32
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import load_env
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from train.paths import ROOT, SFT_JSONL

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RENDERER = "qwen3_8_disable_thinking"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=str(SFT_JSONL))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log-path", default=str(ROOT / "train" / "logs" / "sft"))
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}; run train/teacher_label.py --oracle")
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    config = train.Config(
        log_path=args.log_path,
        model_name=args.model,
        recipe_name="recipe_spreadsheet_sft",
        renderer_name=args.renderer,
        dataset_builder=FromConversationFileBuilder(common_config=common, file_path=str(jsonl)),
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        eval_every=0,
        save_every=20,
    )
    print(f"sft {args.model} rank={args.lora_rank} lr={args.lr} data={jsonl}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()

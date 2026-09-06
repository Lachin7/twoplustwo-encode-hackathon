"""LoRA SFT of Qwen3.8-27B on train-small oracle JSONL.

    uv run train/sft_tinker.py
    uv run train/sft_tinker.py --jsonl data/sft/spreadsheet_sft.jsonl --lora-rank 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import load_env
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from train.paths import ROOT, SFT_JSONL, SFT_MANIFEST, SFT_SMALL_JSONL, SMALL_CELL_LIMIT, SPLIT_META

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=str(SFT_JSONL))
    p.add_argument("--manifest", default=str(SFT_MANIFEST))
    p.add_argument("--split", default=str(SPLIT_META))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--renderer", help="default: cookbook recommended renderer for --model")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--all-train", action="store_true", help="do not drop huge / non-train-small rows")
    p.add_argument(
        "--raw-jsonl",
        action="store_true",
        help="use --jsonl as-is (no train-small filter). For 912 pools.",
    )
    p.add_argument(
        "--load-checkpoint",
        help="tinker://.../weights/final to continue from (e.g. the 286 oracle LoRA)",
    )
    p.add_argument(
        "--train-on-what",
        choices=("last_assistant_message", "all_assistant_messages", "customized"),
        default="last_assistant_message",
        help="Use customized with per-message trainable flags to mask failed agent turns.",
    )
    p.add_argument("--log-path", default=str(ROOT / "train" / "logs" / "sft-small"))
    return p.parse_args()


def conversation_chars(row: dict) -> int:
    return sum(len(m.get("content") or "") for m in row.get("messages") or [])


def filter_train_small(jsonl: Path, manifest: Path, split: Path, out: Path, max_length: int) -> Path:
    meta = json.loads(split.read_text())
    keep_ids = set(meta["train_small_ids"])
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    if manifest.exists():
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    max_chars = max_length * 4
    kept: list[dict] = []
    if records and len(records) == len(rows):
        for rec, row in zip(records, rows):
            n_cells = rec.get("n_cells") or 0
            if rec.get("id") not in keep_ids or n_cells > SMALL_CELL_LIMIT:
                continue
            if conversation_chars(row) > max_chars:
                continue
            kept.append(row)
    else:
        for row in rows:
            if conversation_chars(row) > max_chars:
                continue
            kept.append(row)
    if not kept:
        raise SystemExit(f"no train-small rows from {jsonl}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"sft data {len(rows)} -> {len(kept)} train-small -> {out}")
    return out


def main() -> None:
    load_env()
    args = parse_args()
    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}; run train/build_clean_agent_sft.py")
    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    if args.raw_jsonl:
        pass
    elif not args.all_train:
        jsonl = filter_train_small(jsonl, Path(args.manifest), Path(args.split), SFT_SMALL_JSONL, args.max_length)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat(args.train_on_what),
    )
    config = train.Config(
        log_path=args.log_path,
        model_name=args.model,
        recipe_name="recipe_spreadsheet_sft",
        renderer_name=renderer_name,
        dataset_builder=FromConversationFileBuilder(common_config=common, file_path=str(jsonl)),
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        eval_every=0,
        save_every=20,
        load_checkpoint_path=args.load_checkpoint,
    )
    print(
        f"sft {args.model} renderer={renderer_name} rank={args.lora_rank} lr={args.lr} "
        f"train_on={args.train_on_what} data={jsonl} load={args.load_checkpoint or 'base'}"
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()

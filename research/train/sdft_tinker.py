"""SDFT on spreadsheet gold (teacher sees gold ICL; student does not).

Cookbook recipes/sdft — anti-forgetting alternative to oracle SFT.

    uv run train/sdft_tinker.py --jsonl data/sft/sft_512_case1.jsonl \\
        --log-path train/logs/sdft-case1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import SYSTEM_PROMPT, load_env
from tinker_cookbook import renderers
from tinker_cookbook.distillation import sdft
from tinker_cookbook.recipes.sdft.datasets import SDFTDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

from train.paths import ROOT, SFT_512_CASE1_JSONL

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RENDERER = "qwen3_8_xhigh_reasoning"

# Spreadsheet-shaped demo: gold JSON is the ICL answer, not a free-form essay.
DEMO_TEMPLATE = (
    "{question}\n\n"
    "Example of a correct JSON reply for this task:\n"
    "{golden_answer}\n\n"
    "Now produce your own JSON reply for the same task (formulas or values)."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=str(SFT_512_CASE1_JSONL))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--groups-per-batch", type=int, default=16)
    p.add_argument("--group-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=16384 * 3)
    p.add_argument("--log-path", default=str(ROOT / "train" / "logs" / "sdft-case1"))
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


def load_qa(jsonl: Path, max_chars: int) -> tuple[list[str], list[str]]:
    questions: list[str] = []
    goldens: list[str] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        asst = next((m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if not user or not asst:
            continue
        if len(user) + len(asst) > max_chars:
            continue
        questions.append(user)
        goldens.append(asst)
    if not questions:
        raise SystemExit(f"no usable rows in {jsonl}")
    return questions, goldens


def main() -> None:
    load_env()
    args = parse_args()
    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")
    questions, goldens = load_qa(jsonl, args.max_chars)
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer=tokenizer)
    dataset = SDFTDataset(
        questions=questions,
        golden_answers=goldens,
        batch_size=args.groups_per_batch,
        group_size=args.group_size,
        renderer=renderer,
        dataset_name="spreadsheet_sdft",
    )
    config = sdft.Config(
        model_name=args.model,
        recipe_name="recipe_spreadsheet_sdft",
        renderer_name=args.renderer,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
        topk=args.topk,
        demo_template=DEMO_TEMPLATE,
        system_prompt=SYSTEM_PROMPT,
        teacher_sync_every=None,
        log_path=args.log_path,
        eval_every=0,
        save_every=args.save_every,
        max_steps=args.max_steps,
    )
    print(
        f"sdft {args.model} n={len(questions)} topk={args.topk} "
        f"lr={args.lr} g={args.group_size}x{args.groups_per_batch} "
        f"steps={args.max_steps} -> {args.log_path}"
    )
    asyncio.run(sdft.main(config, dataset, None))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge Gemini-kept traces with oracle fill for dropped / missing train-small ids.

Writes data/sft/spreadsheet_sft_merged.jsonl for SFT.

    uv run train/scripts/label_merge_oracle.py
    uv run train/scripts/label_merge_oracle.py --gemini-only   # no oracle fill
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baseline"))

from common import load_env
from sb import DEFAULT_DATASET, load_dataset

from train.paths import (
    GEMINI_SFT_JSONL,
    GEMINI_SFT_MANIFEST,
    MERGED_SFT_JSONL,
    MERGED_SFT_MANIFEST,
    SPLIT_META,
)
from train.teacher_label import conversation, golden_assistant


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--gemini-jsonl", default=str(GEMINI_SFT_JSONL))
    p.add_argument("--gemini-manifest", default=str(GEMINI_SFT_MANIFEST))
    p.add_argument("--out-jsonl", default=str(MERGED_SFT_JSONL))
    p.add_argument("--out-manifest", default=str(MERGED_SFT_MANIFEST))
    p.add_argument("--split", default=str(SPLIT_META))
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--gemini-only", action="store_true", help="do not fill gaps with oracle")
    args = p.parse_args()

    meta = json.loads(Path(args.split).read_text())
    train_small = list(meta["train_small_ids"])
    gemini_manifest = {r["id"]: r for r in load_jsonl(Path(args.gemini_manifest))}
    gemini_kept_ids = {i for i, r in gemini_manifest.items() if r.get("kept")}

    # conversations are 1:1 with kept rows in order of the gemini jsonl; map via re-read + id from manifest order
    # Safer: rebuild gemini conversations only for kept ids by re-running conversation from stored assistant —
    # but jsonl has no id field. Match by counting kept in manifest order vs jsonl lines.
    gemini_convs = load_jsonl(Path(args.gemini_jsonl))
    kept_ordered = [r for r in load_jsonl(Path(args.gemini_manifest)) if r.get("kept")]
    if len(gemini_convs) != len(kept_ordered):
        raise SystemExit(
            f"gemini jsonl ({len(gemini_convs)}) != kept manifest rows ({len(kept_ordered)}); "
            "re-run labelling or fix files"
        )
    by_id_conv = {r["id"]: conv for r, conv in zip(kept_ordered, gemini_convs)}

    tasks = {t["id"]: t for t in load_dataset(Path(args.dataset_dir)) if t["id"] in set(train_small)}
    out_jsonl = Path(args.out_jsonl)
    out_manifest = Path(args.out_manifest)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_gemini = 0
    n_oracle = 0
    with out_jsonl.open("w", encoding="utf-8") as jf, out_manifest.open("w", encoding="utf-8") as mf:
        for tid in train_small:
            task = tasks.get(tid)
            if task is None:
                continue
            if tid in by_id_conv:
                jf.write(json.dumps(by_id_conv[tid], ensure_ascii=False) + "\n")
                row = {**gemini_manifest[tid], "source": gemini_manifest[tid].get("source", "gemini"), "kept": True}
                mf.write(json.dumps(row) + "\n")
                n_gemini += 1
                continue
            if args.gemini_only:
                continue
            assistant = golden_assistant(task)
            jf.write(json.dumps(conversation(task, assistant), ensure_ascii=False) + "\n")
            mf.write(
                json.dumps(
                    {
                        "id": tid,
                        "source": "oracle",
                        "kept": True,
                        "pass": True,
                        "cell_accuracy": 1.0,
                        "type": task.get("instruction_type"),
                        "fill": "dropped_or_missing",
                    }
                )
                + "\n"
            )
            n_oracle += 1

    print(f"merged gemini={n_gemini} oracle_fill={n_oracle} -> {out_jsonl}")
    print(f"manifest -> {out_manifest}")
    print("SFT with: uv run train/sft_tinker.py --jsonl data/sft/spreadsheet_sft_merged.jsonl")


if __name__ == "__main__":
    main()

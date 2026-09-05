#!/usr/bin/env python3
"""Summarize a Gemini / SFT label manifest.

    uv run train/scripts/label_status.py
    uv run train/scripts/label_status.py --manifest data/sft/gemini/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.paths import GEMINI_SFT_MANIFEST, SPLIT_META


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(GEMINI_SFT_MANIFEST))
    p.add_argument("--split", default=str(SPLIT_META))
    args = p.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        raise SystemExit(f"missing {path} — run label_smoke.sh or label_train_small.sh first")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [r for r in rows if r.get("kept")]
    dropped = [r for r in rows if not r.get("kept")]
    by_type = Counter((r.get("type") or "?").split()[0] for r in kept)
    sources = Counter(r.get("source", "?") for r in kept)

    meta = json.loads(Path(args.split).read_text())
    target = len(meta.get("train_small_ids", []))

    print(f"manifest: {path}")
    print(f"rows:     {len(rows)}")
    print(f"kept:     {len(kept)}  ({100 * len(kept) / len(rows):.1f}% of attempted)" if rows else "kept: 0")
    print(f"dropped:  {len(dropped)}")
    print(f"coverage: {len(kept)}/{target} train-small")
    if sources:
        print("sources:  " + ", ".join(f"{k}={v}" for k, v in sources.most_common()))
    if by_type:
        print("kept type:" + ", ".join(f"{k}={v}" for k, v in by_type.most_common()))
    if dropped:
        print("dropped ids (first 20): " + ", ".join(r["id"] for r in dropped[:20]))


if __name__ == "__main__":
    main()

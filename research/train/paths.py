from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = DATA / "splits"
SFT_DIR = DATA / "sft"
GEMINI_SFT_DIR = SFT_DIR / "gemini"
SPLIT_META = SPLITS / "split.json"
TRAIN_IDS = SPLITS / "train.json"
EVAL_IDS = SPLITS / "eval.json"
SFT_JSONL = SFT_DIR / "spreadsheet_sft.jsonl"
SFT_MANIFEST = SFT_DIR / "manifest.jsonl"
SFT_SMALL_JSONL = SFT_DIR / "spreadsheet_sft_small.jsonl"
GEMINI_SFT_JSONL = GEMINI_SFT_DIR / "spreadsheet_sft.jsonl"
GEMINI_SFT_MANIFEST = GEMINI_SFT_DIR / "manifest.jsonl"
MERGED_SFT_JSONL = SFT_DIR / "spreadsheet_sft_merged.jsonl"
MERGED_SFT_MANIFEST = SFT_DIR / "manifest_merged.jsonl"
SMALL_CELL_LIMIT = 200
EVAL_CELL = 56
EVAL_SHEET = 24
SPLIT_SEED = 42

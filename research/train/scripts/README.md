# Gemini labelling (Google AI Studio)

Pass-only teacher traces for train-small. No goldens in the user prompt.

## Setup

1. Key from https://aistudio.google.com/apikey
2. In `research/.env`:

```sh
GOOGLE_API_KEY=...
TEACHER_MODEL=gemini-2.5-pro
```

3. LibreOffice: `export SOFFICE=/Applications/LibreOffice.app/Contents/MacOS/soffice`

## Workflow

```sh
cd research
chmod +x train/scripts/label_smoke.sh train/scripts/label_train_small.sh

# 1) smoke (~5 ids)
./train/scripts/label_smoke.sh

# 2) full train-small (resumable; re-run safely)
./train/scripts/label_train_small.sh

# 3) status
uv run train/scripts/label_status.py

# 4) merge Gemini keeps + oracle fill for drops → SFT jsonl
uv run train/scripts/label_merge_oracle.py
```

Outputs live under `data/sft/gemini/` (not the old oracle `spreadsheet_sft.jsonl`).

| File | Role |
|------|------|
| `data/sft/gemini/spreadsheet_sft.jsonl` | Gemini-kept chats only |
| `data/sft/gemini/manifest.jsonl` | kept / dropped per id |
| `data/sft/spreadsheet_sft_merged.jsonl` | Gemini + oracle gaps |

## Tips

- Cheaper: `MODEL=gemini-2.5-flash ./train/scripts/label_train_small.sh`
- Wipe and restart: `./train/scripts/label_train_small.sh --fresh`
- One-off ids: `./train/scripts/label_smoke.sh 13-1,51-12`
- Low keep rate → check `SOFFICE`, datetime harness, or try more `--samples`

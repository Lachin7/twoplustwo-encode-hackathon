# Training scripts

## Overnight (Tinker)

From `research/`:

```sh
chmod +x train/scripts/run_overnight*.sh train/scripts/run_morning_eval.sh
./train/scripts/run_overnight.sh          # A+B+C in parallel
./train/scripts/run_overnight.sh rl       # A only
./train/scripts/run_overnight.sh traces   # B only
./train/scripts/run_overnight.sh sdft     # C only
```

- **A** `run_overnight_rl.sh` — RL from **base** + KL 0.05 + `--recalc`. Logs: `train/logs/rl-base-kl`
- **B** `run_overnight_trace_sft.sh` — train-only base+agent traces → pass-only SFT at 1e-5 on every assistant turn. Logs: `train/logs/sft-train-traces`
- **C** `run_overnight_sdft.sh` — SDFT on `sft_512_case1.jsonl` (gold is teacher ICL only). Logs: `train/logs/sdft-case1`

Do not point any of these at an oracle SFT checkpoint.

Morning (eval-80 + agent vs base 76.2%):

```sh
# auto: waits for each overnight final, then evals one-by-one
./train/scripts/watch_and_eval.sh

# or manual once you have a path:
MODEL_PATH=tinker://.../sampler_weights/final NAME=rl-base-kl ./train/scripts/run_morning_eval.sh
```

If sheet-level drops, discard the LoRA and ship base+agent.

---

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

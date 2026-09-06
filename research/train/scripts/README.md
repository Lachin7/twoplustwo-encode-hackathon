# Training

The shipped LoRA is masked agent SFT. Only this recipe is kept.

```sh
cd research
uv run train/build_clean_agent_sft.py
# then, or:
./train/scripts/run_clean_agent_sft_smoke.sh
```

`build_clean_agent_sft.py` takes passing **train-320** agent traces, keeps the full conversation as context, and sets `trainable=False` on inspect-only, error, and thin-fill turns. Loss is only on successful mutations, cell answers, and `done` after a mutation.

`sft_tinker.py` trains that JSONL with `--train-on-what customized`.

The checkpoint we ship:

`tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final`

See `train/SHIPPED_CHECKPOINT.txt` and `FAILURE_ANALYSIS.md`.

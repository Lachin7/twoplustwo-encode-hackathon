# Full-400 failure analysis and fast fine-tuning

## Baseline

The shipped base run uses `Qwen/Qwen3.8-27B`,
`qwen3_8_xhigh_reasoning`, and `--agent auto`.

- Full 400: 260/400, or 65.0% pass rate.
- One-shot route: 175/233 (75.1%).
- Python-agent route: 85/167 (50.9%).
- Agent failures account for 82 of the 140 failures.
- The frozen eval-80 baseline is 61/80 (76.25%).

## What failed

The categories below overlap because one trajectory can encounter several
problems.

- 69 failing tasks had at least one Python-tool exception.
- 52 agent failures consumed the full six-turn budget.
- 31 failures had a non-JSON or otherwise unparseable model turn.
- 33 failures had an Excel error such as `#VALUE!` or `#N/A` at the first
  mismatching cell.
- 32 failures had a blank first mismatch, usually from an unfinished range or
  wrong worksheet.
- The broader 127-task Python-error cohort passed only 45.7%; the 47-task
  parse-error cohort passed only 34.0%. Traces without those mechanical
  signatures passed 76.3%.

The most important harness defect was self-inflicted: serialized headings such
as `### Sheet: Data overview` caused the model to use `"Data overview"` as the
literal worksheet name. The restricted Python runtime also omitted ordinary
builtins (`repr`, `type`, `isinstance`, `hasattr`) and rejected imports the
model commonly emitted. These two defects were responsible for repeated,
recoverable turns.

Task `80-42` never reached the model: its prompt was 83,838 tokens before a
16,384-token completion request, beyond Qwen's 65,536-token context. Agent
workbooks now use a shorter 40-row preview, a 100,000-character cap, and a
dynamic completion budget. Its initial prompt is now 27,102 tokens.

## Harness fixes and measured recovery

The agent runtime now:

- labels the exact worksheet name separately from `overview` / `focus`;
- exposes safe, common builtins and allowlisted standard-library imports;
- shortens previews for tasks that can inspect the live workbook;
- caps completion tokens to the model's remaining context;
- gives one-shot JSON a third attempt and falls back to the Python agent only
  after terminal parse failure.

All 47 local harness tests pass. On deliberately selected prior full-400
failures, the worksheet/runtime fixes recovered 5/8 tasks (`353-6`, `398-14`,
`54638`, `57232`, `61-4`). The hard-error retry/context set recovered another
3/7 (`254-34`, `56921`, `9448`). These targeted figures show that the failures
were real harness gaps; they are not an unbiased estimate of a new full-400
score.

## Fine-tuning variants

Mixed rejection SFT on successful one-shot and agent traces used 80 examples,
rank 16, one epoch, and learning rate `1e-5`. It scored 75.0% on eval-80, below
the 76.25% base pipeline. Agent-only rejection SFT at `5e-6` still broke known
one-shot winners. Both trained on every assistant turn, including imports that
failed, inspection-only turns, and wandering before `done`.

The cookbook's `TrainOnWhat.CUSTOMIZED` mode supports a cleaner objective. The
new builder keeps the full successful trajectory as context but gives zero
loss to failed, parse-error, thin-fill, and inspection-only assistant turns.
Only successful workbook mutations, cell answers, and `done` after a mutation
receive loss:

- 66 passing agent conversations from the 320-task train split;
- 128 trainable assistant turns;
- rank 16, one epoch, batch size 4, learning rate `3e-6`;
- 16 optimizer steps and about two minutes wall time;
- no eval-80 tasks or golden workbook values in the training JSONL.

The checkpoint is:

`tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final`

It scored 12/16 on the guarded smoke set versus 11/16 for the same-harness base
control. On the full frozen eval-80 it scored 62/80 (77.5%), one task above the
base+agent result. Sheet-level pass rate improved from 75.0% to 79.17%;
cell-level pass rate stayed at 76.79%. It gained seven tasks and lost six, so
the improvement is measured but narrow.

## Why not DPO, SDFT, or RL now

The cookbook DPO implementation needs chosen and rejected completions for the
same prompt. Two independent train runs disagreed on 48 outcomes, but after
requiring both to be agent traces and the chosen first action to execute
cleanly, only four valid first-action pairs remained. Training DPO on four
pairs would be overfitting, not a credible cheap experiment.

`last_assistant_message` SFT is also unsuitable for full agent traces because
the final target is usually only `{"tool":"done"}`. SDFT previously taught
golden value dumping rather than robust workbook operations. Spreadsheet RL
needs LibreOffice recalculation and has a sparse all-cells-correct reward; a
proper run is slower and more expensive than the remaining deadline allows.

The practical order is therefore: ship the validated harness fixes, promote
the masked LoRA only after a full-400 run beats the base score, and defer
preference/RL work until more paired rollouts can be collected.

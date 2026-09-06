"""Cookbook ProblemEnv over the recommended RL JSONL.

Same shape as tinker_cookbook.recipes.math_rl.math_env.MathEnv: one user
question, one assistant reply, format + answer reward. The question is the
current infer prompt; the answer grader is write_output + score_task.

This is not the Python agent loop. That path is infer-only. The RL pool
(data/rl/rl_train.jsonl) is built for this single-turn env.
"""

from __future__ import annotations

import json
import math
import random
import sys
import uuid
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import chz
import tinker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))

from common import FORMAT_HINT, SYSTEM_PROMPT, build_prompt, parse_answer, write_output
from evaluate import score_task
from sb import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from train.paths import RL_DEV_JSONL, RL_TRAIN_JSONL, SMALL_CELL_LIMIT, SPLIT_META, SPLIT_SEED
from train.rl_data import load_rl_tasks


def reward_from_scores(scores: dict, format_coef: float = 0.1) -> float:
    """Cookbook format term + dense cell score (pass bonus so a full hit stands out)."""
    return (
        format_coef * (scores["format"] - 1.0)
        + scores["cell_accuracy"]
        + 0.25 * scores["pass"]
    )


def score_sample(task: dict, sample_str: str, work_dir: Path, recalc: bool) -> dict:
    try:
        answer = parse_answer(sample_str)
    except Exception:
        return {"format": 0.0, "cell_accuracy": 0.0, "pass": 0.0}
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{task['id']}-{uuid.uuid4().hex[:8]}.xlsx"
    try:
        write_output(task, answer, out)
        result = score_task(task, str(out), recalc, str(work_dir))
    except Exception:
        return {"format": 1.0, "cell_accuracy": 0.0, "pass": 0.0}
    finally:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
    if result.get("status") != "graded":
        return {"format": 1.0, "cell_accuracy": 0.0, "pass": 0.0}
    cells = result.get("cells") or 0
    correct = result.get("correct") or 0
    acc = (correct / cells) if cells else 0.0
    return {"format": 1.0, "cell_accuracy": acc, "pass": float(bool(result.get("pass")))}


class SpreadsheetEnv(ProblemEnv):
    def __init__(
        self,
        task: dict,
        renderer: renderers.Renderer,
        work_dir: Path,
        recalc: bool = False,
        format_coef: float = 0.1,
    ):
        super().__init__(
            renderer,
            convo_prefix=[{"role": "system", "content": SYSTEM_PROMPT}],
            format_coef=format_coef,
            require_stop_sequence_for_format=False,
        )
        self.task = task
        self.work_dir = Path(work_dir)
        self.recalc = recalc

    def get_question(self) -> str:
        return build_prompt(self.task) + FORMAT_HINT

    def check_format(self, sample_str: str) -> bool:
        try:
            parse_answer(sample_str)
            return True
        except Exception:
            return False

    def score_sample(self, sample_str: str) -> dict:
        return score_sample(self.task, sample_str, self.work_dir, self.recalc)

    def check_answer(self, sample_str: str) -> bool:
        return self.score_sample(sample_str)["pass"] >= 1.0

    def get_reference_answer(self) -> str:
        return f"{self.task['id']} {self.task.get('answer_position')}"

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Same single-turn step as ProblemEnv, with cell_accuracy in the reward."""
        message, _termination = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        scores = self.score_sample(content)
        return StepResult(
            reward=reward_from_scores(scores, self.format_coef),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": scores["format"],
                "correct": scores["pass"],
                "cell_accuracy": scores["cell_accuracy"],
            },
        )


class SpreadsheetDataset(RLDataset):
    def __init__(
        self,
        tasks: list[dict],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        work_dir: Path,
        recalc: bool,
        seed: int | None = None,
    ):
        self.tasks = list(tasks)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.tasks)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.work_dir = work_dir
        self.recalc = recalc

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        chunk = self.tasks[start : start + self.batch_size]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    SpreadsheetEnv,
                    task=task,
                    renderer=self.renderer,
                    work_dir=self.work_dir,
                    recalc=self.recalc,
                ),
                num_envs=self.group_size,
                dataset_name="spreadsheetbench",
            )
            for task in chunk
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.tasks) / self.batch_size) if self.tasks else 0


@chz.chz
class SpreadsheetDatasetBuilder(RLDatasetBuilder):
    split_path: str = str(SPLIT_META)
    dataset_dir: str
    groups_per_batch: int = 16
    group_size: int = 4
    model_name_for_tokenizer: str
    renderer_name: str
    work_dir: str
    recalc: bool = False
    small_only: bool = True
    pool: str = "recommended"
    rl_train_path: str = str(RL_TRAIN_JSONL)
    rl_dev_path: str = str(RL_DEV_JSONL)
    seed: int = SPLIT_SEED

    async def __call__(self) -> tuple[SpreadsheetDataset, SpreadsheetDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        work = Path(self.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        if self.pool == "recommended":
            train_tasks = load_rl_tasks(Path(self.rl_train_path))
            dev_tasks = load_rl_tasks(Path(self.rl_dev_path))
            train = SpreadsheetDataset(
                train_tasks,
                self.groups_per_batch,
                self.group_size,
                renderer,
                work,
                self.recalc,
                seed=self.seed,
            )
            dev = SpreadsheetDataset(
                dev_tasks,
                min(self.groups_per_batch, max(1, len(dev_tasks))),
                1,
                renderer,
                work,
                self.recalc,
            )
            return train, dev
        if self.pool != "train_small":
            raise ValueError(f"unknown RL pool {self.pool!r}; use recommended|train_small")
        meta = json.loads(Path(self.split_path).read_text())
        ids = meta["train_small_ids"] if self.small_only else meta["train_ids"]
        id_set = set(ids)
        tasks = [t for t in load_dataset(Path(self.dataset_dir)) if t["id"] in id_set]
        if self.small_only:
            n_cells = meta.get("n_cells") or {}
            tasks = [t for t in tasks if n_cells.get(t["id"], 0) <= SMALL_CELL_LIMIT]
        return (
            SpreadsheetDataset(
                tasks,
                self.groups_per_batch,
                self.group_size,
                renderer,
                work,
                self.recalc,
                seed=self.seed,
            ),
            None,
        )

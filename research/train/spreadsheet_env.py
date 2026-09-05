"""SpreadsheetBench ProblemEnv: reward = format bonus + cell_accuracy."""

from __future__ import annotations

import json
import math
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

from train.paths import SMALL_CELL_LIMIT, SPLIT_META


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
        try:
            answer = parse_answer(sample_str)
        except Exception:
            return {"format": 0.0, "cell_accuracy": 0.0, "pass": 0.0}
        out = self.work_dir / f"{self.task['id']}-{uuid.uuid4().hex[:8]}.xlsx"
        try:
            write_output(self.task, answer, out)
            result = score_task(self.task, str(out), self.recalc, str(self.work_dir))
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

    def check_answer(self, sample_str: str) -> bool:
        return self.score_sample(sample_str)["pass"] >= 1.0

    def get_reference_answer(self) -> str:
        return f"{self.task['id']} {self.task.get('answer_position')}"

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _termination = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        scores = self.score_sample(content)
        total_reward = (
            self.format_coef * (scores["format"] - 1.0)
            + scores["cell_accuracy"]
            + 0.25 * scores["pass"]
        )
        return StepResult(
            reward=total_reward,
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
    ):
        self.tasks = tasks
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.work_dir = work_dir
        self.recalc = recalc

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        chunk = self.tasks[start : start + self.batch_size]
        builders = []
        for task in chunk:
            builders.append(
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
            )
        return builders

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

    async def __call__(self) -> tuple[SpreadsheetDataset, SpreadsheetDataset | None]:
        meta = json.loads(Path(self.split_path).read_text())
        ids = meta["train_small_ids"] if self.small_only else meta["train_ids"]
        id_set = set(ids)
        tasks = [t for t in load_dataset(Path(self.dataset_dir)) if t["id"] in id_set]
        if self.small_only:
            n_cells = meta.get("n_cells") or {}
            tasks = [t for t in tasks if n_cells.get(t["id"], 0) <= SMALL_CELL_LIMIT]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)
        work = Path(self.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        train = SpreadsheetDataset(
            tasks, self.groups_per_batch, self.group_size, renderer, work, self.recalc
        )
        return train, None

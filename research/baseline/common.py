"""Shared by llm_predict.py and tinker_predict.py: prompt, answer schema, output files, traces."""

import asyncio
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl
from openpyxl.cell.cell import MergedCell
from pydantic import BaseModel, Field

from sb import answer_cells, load_dataset, read_jsonl, serialize_workbook

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

SYSTEM_PROMPT = (
    "You are a spreadsheet expert. You get a serialized workbook and a user instruction. "
    "Compute the final values the answer range must contain after the instruction is applied. "
    "Return one entry per cell in the answer range. Use null for cells that must be empty. "
    "Return plain values, not formulas."
)
FORMAT_HINT = (
    '\n\nReply with JSON only, no prose, in this shape: '
    '{"cells": [{"cell": "B6", "value": 42}, {"cell": "B7", "value": null}]}'
)
RETRY_HINT = (
    "\n\nYour previous reply was not valid JSON. "
    "Reply with JSON only, no prose, in this shape: "
    '{"cells": [{"cell": "B6", "value": 42}, {"cell": "B7", "value": null}]}'
)

_IO_LOCK = threading.Lock()


class CellValue(BaseModel):
    cell: str = Field(description="Cell address like A3 or B6")
    value: str | int | float | bool | None = Field(description="Final value for that cell")


class SpreadsheetAnswer(BaseModel):
    cells: list[CellValue]


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def parse_ids(text: str | None) -> set[str] | None:
    return {i.strip() for i in text.split(",") if i.strip()} if text else None


def selected_tasks(dataset_dir: Path, ids: set[str] | None) -> list[dict]:
    tasks = load_dataset(dataset_dir)
    return tasks if ids is None else [t for t in tasks if t["id"] in ids]


def build_prompt(task: dict) -> str:
    return (
        f"## Instruction\n{task['instruction']}\n\n"
        f"## Workbook\n{serialize_workbook(task['init_xlsx'])}\n\n"
        f"## Answer range\nSheet: {task.get('answer_sheet') or 'active sheet'}\nCells: {task['answer_position']}\n"
    )


def coerce_cell_value(value: str | int | float | bool | None) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in ("null", "none"):
            return None
        try:
            number = float(stripped.replace(",", ""))
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    return value


def parse_answer(text: str) -> SpreadsheetAnswer:
    """First {...} block in the reply. Thinking models wrap it in prose or code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    return SpreadsheetAnswer.model_validate(json.loads(text[start:end + 1]))


def align_answer(task: dict, answer: SpreadsheetAnswer, wb=None) -> SpreadsheetAnswer:
    """Write into the graded range. Use the model's keys when they match; otherwise zip in order."""
    expected = [coord.upper() for _sheet, coord in answer_cells(task, wb)]
    raw = [(cell.cell.upper(), coerce_cell_value(cell.value)) for cell in answer.cells]
    by_key = {coord: value for coord, value in raw}
    if expected and all(coord in by_key for coord in expected):
        cells = [CellValue(cell=coord, value=by_key[coord]) for coord in expected]
    elif expected and raw:
        cells = [
            CellValue(cell=coord, value=raw[i][1])
            for i, coord in enumerate(expected)
            if i < len(raw)
        ]
    else:
        cells = [CellValue(cell=coord, value=value) for coord, value in raw]
    return SpreadsheetAnswer(cells=cells)


def set_cell_value(ws, coord: str, value) -> None:
    cell = ws[coord]
    if isinstance(cell, MergedCell):
        for merged in ws.merged_cells.ranges:
            if coord in merged:
                ws.cell(merged.min_row, merged.min_col).value = value
                return
        return
    ws[coord] = value


def write_output(task: dict, answer: SpreadsheetAnswer, out_path: Path) -> None:
    shutil.copy(task["init_xlsx"], out_path)
    wb = openpyxl.load_workbook(out_path)
    aligned = align_answer(task, answer, wb)
    cells = {cell.cell.upper(): cell.value for cell in aligned.cells}
    for sheet, coord in answer_cells(task, wb):
        ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active
        if coord in cells:
            set_cell_value(ws, coord, cells[coord])
    wb.save(out_path)


def prepare_out_dir(out_dir: Path) -> None:
    for sub in ("outputs", "traces"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
        (out_dir / sub).mkdir(parents=True)
    for name in ("predictions.jsonl", "run.log"):
        (out_dir / name).write_text("", encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _IO_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def log(out_dir: Path, line: str) -> None:
    print(line, flush=True)
    with _IO_LOCK:
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


async def predict_task(complete, model: str, task: dict, out_dir: Path) -> str:
    """One model call, retry once on bad JSON. On total failure copy the init workbook.

    `complete(prompt)` is an async function returning (text, input_tokens, output_tokens).
    """
    out = out_dir / "outputs" / f"{task['id']}.xlsx"
    prompt = build_prompt(task)
    status = "ok"
    last_error: Exception | None = None
    for step, extra in enumerate(("", RETRY_HINT), start=1):
        trace = {"step": step, "model": model, "prompt": None, "response": None,
                 "input_tokens": None, "output_tokens": None, "latency_ms": None, "error": None}
        started = time.time()
        try:
            trace["prompt"] = prompt + extra
            text, trace["input_tokens"], trace["output_tokens"] = await complete(trace["prompt"])
            trace["response"] = text
            write_output(task, parse_answer(text), out)
            last_error = None
            status = "ok"
        except Exception as e:
            last_error = e
            trace["error"] = f"{type(e).__name__}: {e}"[:500]
            status = f"error: {e}"[:200]
        trace["latency_ms"] = int((time.time() - started) * 1000)
        append_jsonl(out_dir / "traces" / f"{task['id']}.jsonl", trace)
        if last_error is None:
            break
    if last_error is not None:
        shutil.copy(task["init_xlsx"], out)
    append_jsonl(out_dir / "predictions.jsonl", {"id": task["id"], "output": f"outputs/{task['id']}.xlsx", "status": status})
    return status


async def run(complete, model: str, tasks: list[dict], out_dir: Path, concurrency: int, resume: bool = False) -> None:
    if resume and (out_dir / "predictions.jsonl").exists():
        done = {row["id"] for row in read_jsonl(out_dir / "predictions.jsonl")}
        tasks = [task for task in tasks if task["id"] not in done]
        (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (out_dir / "traces").mkdir(parents=True, exist_ok=True)
        log(out_dir, f"resume skip {len(done)}  remaining {len(tasks)}")
    else:
        prepare_out_dir(out_dir)
    log(out_dir, f"model {model}  tasks {len(tasks)}")
    if not tasks:
        return
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(task: dict) -> None:
        async with semaphore:
            status = await predict_task(complete, model, task, out_dir)
        log(out_dir, f"{task['id']:<8} {status}")

    await asyncio.gather(*(run_one(task) for task in tasks))

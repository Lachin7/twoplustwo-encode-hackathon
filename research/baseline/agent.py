"""Optional Python tool loop around a live openpyxl workbook.

One-shot JSON stays the default for small cell-level tasks. The loop is for
ranges the model cannot dump (auto: more than AGENT_CELL_THRESHOLD graded cells).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import importlib
import io
import json
import re
import shutil
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

import openpyxl
from openpyxl.utils import get_column_letter

from common import (
    FORMAT_HINT,
    SYSTEM_PROMPT,
    SpreadsheetAnswer,
    append_jsonl,
    build_prompt,
    graded_count,
    output_is_thin,
    parse_answer,
    write_output,
)
from sb import answer_cells

AGENT_CELL_THRESHOLD = 20
MAX_TURNS = 10
# Reasoning that runs to the sampling cap never reaches a tool call, so the turn is
# lost. Capping each turn below the context limit trades a little headroom for more
# turns at roughly the same total token spend.
AGENT_MAX_TOKENS = 10240
EXEC_TIMEOUT_S = 20
RESULT_CHARS = 4000

CompleteChat = Callable[..., Awaitable[tuple[str, int | None, int | None]]]

AGENT_SYSTEM = (
    SYSTEM_PROMPT
    + " When the range is large you may edit the live workbook with Python instead of dumping values. "
    "Each turn reply with JSON only, one of: "
    '{"tool":"python","code":"..."} or {"tool":"done"} or '
    '{"cells":[{"cell":"B6","value":"=A6*2"}]}. '
    "Namespace: wb (openpyxl workbook), task (dict), ANSWER (list of (sheet, coord) to fill), "
    "openpyxl, get_column_letter, json, math, datetime, re. Print to inspect. "
    "Worksheet names are exactly the strings in wb.sheetnames; never append display labels such as "
    "'overview' or 'focus'. Common safe modules are already available, so imports are unnecessary. "
    "No network, subprocess, or files. Save by writing cells on wb, then {\"tool\":\"done\"}."
)

AGENT_HINT = (
    FORMAT_HINT
    + "\nOr a tool: {\"tool\":\"python\",\"code\":\"print(wb.sheetnames)\"} "
    "then later {\"tool\":\"done\"}."
)


def should_use_agent(task: dict) -> bool:
    """Keep small one-shot tasks on the JSON path so the loop cannot regress them."""
    return graded_count(task) > AGENT_CELL_THRESHOLD


_FENCE_RE = re.compile(r"```(?:([A-Za-z0-9_+-]*)[^\n]*)?\n(.*?)```", re.S)


def as_python(code: str) -> str | None:
    """Return the code if it parses as Python, retrying once dedented."""
    if not code.strip():
        return None
    for candidate in (code, textwrap.dedent(code)):
        try:
            compile(candidate, "<agent>", "exec")
        except (SyntaxError, ValueError):
            continue
        return candidate
    return None


def _python_fence(text: str) -> str | None:
    """Last fenced block that is really Python.

    Models paste the sheet back inside a fence while reasoning, so a fence is not
    by itself evidence of code — it has to compile.
    """
    blocks = _FENCE_RE.findall(text)
    tagged = [code for lang, code in blocks if lang.lower() == "python"]
    plain = [code for lang, code in blocks if lang.lower() != "python"]
    for code in list(reversed(tagged)) + list(reversed(plain)):
        parsed = as_python(code)
        if parsed is not None:
            return parsed
    return None


def parse_agent_turn(text: str) -> tuple[str, Any]:
    """Return ('python', code), ('done', None), or ('answer', SpreadsheetAnswer)."""
    try:
        answer = parse_answer(text)
        return "answer", answer
    except Exception:
        pass
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = stripped.find("{")
    if start >= 0:
        snippet = stripped[start:]
        payload = None
        try:
            payload, _ = json.JSONDecoder().raw_decode(snippet)
        except json.JSONDecodeError:
            end = snippet.rfind("}")
            if end >= 0:
                repaired = re.sub(r",\s*([}\]])", r"\1", snippet[: end + 1])
                try:
                    payload = json.loads(repaired)
                except json.JSONDecodeError:
                    payload = None
        if isinstance(payload, dict):
            tool = str(payload.get("tool") or "").lower()
            if tool == "done":
                return "done", None
            if tool == "python" and payload.get("code"):
                return "python", str(payload["code"])
            if payload.get("cells") is not None:
                return "answer", SpreadsheetAnswer.model_validate(payload)
    fence = _python_fence(stripped) or _python_fence(text)
    if fence is not None:
        return "python", fence
    raise ValueError(f"no tool or JSON in reply: {text[:120]!r}")


_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "chr": chr,
    "dir": dir,
    "divmod": divmod,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "slice": slice,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

_SAFE_IMPORTS = {
    "collections",
    "copy",
    "datetime",
    "itertools",
    "json",
    "math",
    "openpyxl",
    "re",
    "statistics",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level or root not in _SAFE_IMPORTS:
        raise ImportError(f"module {name!r} is not available")
    return importlib.import_module(name)


_SAFE_BUILTINS["__import__"] = _safe_import


def run_python(code: str, wb, task: dict) -> str:
    """Exec model code against the live workbook. Restricted builtins, time-capped."""
    normalized = as_python(code)
    if normalized is None:
        return (
            "ERROR: that is not valid Python. Send the code as one JSON string with "
            "\\n for newlines, no markdown fence, no leading indentation."
        )
    buf = io.StringIO()
    ns = {
        "__builtins__": _SAFE_BUILTINS,
        "wb": wb,
        "task": {
            k: task[k]
            for k in (
                "id",
                "instruction",
                "instruction_type",
                "answer_position",
                "answer_sheet",
                "data_position",
            )
            if k in task
        },
        "ANSWER": answer_cells(task, wb),
        "openpyxl": openpyxl,
        "get_column_letter": get_column_letter,
        "json": json,
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "re": re,
    }

    def _run() -> None:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(normalized, ns, ns)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run)
            fut.result(timeout=EXEC_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return f"ERROR: timed out after {EXEC_TIMEOUT_S}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    out = buf.getvalue()
    if len(out) > RESULT_CHARS:
        out = out[:RESULT_CHARS] + "\n…truncated"
    return out if out.strip() else "(no output — workbook updated if your code wrote cells)"


def _save_wb(wb, path: Path, task: dict | None = None) -> None:
    if task is not None:
        from common import set_cell_value

        for sheet, coord in answer_cells(task, wb):
            ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active
            value = ws[coord].value
            if isinstance(value, str) and not value.lstrip().startswith("="):
                set_cell_value(ws, coord, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(path)
        return
    except Exception:
        # Some inits have images whose PIL handles are already closed; drop media and retry.
        for sheet in wb.worksheets:
            if hasattr(sheet, "_images"):
                sheet._images = []
        wb.save(path)


async def predict_task_agent(
    complete_chat: CompleteChat,
    model: str,
    task: dict,
    out_dir: Path,
) -> str:
    out = out_dir / "outputs" / f"{task['id']}.xlsx"
    shutil.copy(task["init_xlsx"], out)
    wb = openpyxl.load_workbook(out)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": build_prompt(task) + AGENT_HINT},
    ]
    status = "ok"
    last_error: Exception | None = None
    finished = False
    for turn in range(1, MAX_TURNS + 1):
        trace: dict[str, Any] = {
            "step": turn,
            "model": model,
            "mode": "agent",
            "prompt": messages[-1].get("content"),
            "response": None,
            "tool": None,
            "tool_output": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "error": None,
        }
        started = time.time()
        text: str | None = None
        try:
            text, trace["input_tokens"], trace["output_tokens"] = await complete_chat(
                messages, max_tokens=AGENT_MAX_TOKENS
            )
            trace["response"] = text
            kind, payload = parse_agent_turn(text)
            messages.append({"role": "assistant", "content": text})
            if kind == "answer":
                write_output(task, payload, out)
                last_error = None
                status = "ok"
                trace["tool"] = "cells"
                if output_is_thin(task, out) and turn < MAX_TURNS:
                    trace["error"] = "thin fill"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Too few answer cells changed. Use python to write ANSWER, "
                                'or return one formula per column, then {"tool":"done"}.'
                            ),
                        }
                    )
                else:
                    finished = True
            elif kind == "done":
                _save_wb(wb, out, task)
                last_error = None
                status = "ok"
                trace["tool"] = "done"
                if output_is_thin(task, out) and turn < MAX_TURNS:
                    trace["error"] = "thin fill"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Workbook still looks unfinished. Use python to write ANSWER, "
                                'or return one formula per column, then {"tool":"done"}.'
                            ),
                        }
                    )
                else:
                    finished = True
            else:
                trace["tool"] = "python"
                result = run_python(payload, wb, task)
                _save_wb(wb, out, task)
                trace["tool_output"] = result[:800]
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"## Python result\n```\n{result}\n```\n"
                            'Continue. Fill ANSWER cells, then {"tool":"done"}.'
                        ),
                    }
                )
                last_error = None
                status = "ok"
        except Exception as e:
            last_error = e
            trace["error"] = f"{type(e).__name__}: {e}"[:500]
            status = f"error: {e}"[:200]
            truncated = (trace["output_tokens"] or 0) >= AGENT_MAX_TOKENS - 64
            if truncated:
                trace["error"] = "truncated before acting"
                nudge = (
                    "You ran out of room before replying with a tool. Stop analysing and "
                    'answer now with one line of JSON: {"tool":"python","code":"..."}.'
                )
            else:
                nudge = f"Could not parse that ({e}). JSON only: tool python, tool done, or cells."
            # Keep roles alternating and let the model see its own dead end, otherwise
            # it re-runs the same reasoning and burns every remaining turn.
            if text:
                messages.append({"role": "assistant", "content": text[-1500:]})
            messages.append({"role": "user", "content": nudge})
        trace["latency_ms"] = int((time.time() - started) * 1000)
        append_jsonl(out_dir / "traces" / f"{task['id']}.jsonl", trace)
        if finished:
            break
        if last_error is not None and "prompt too long" in str(last_error):
            break
    if not finished:
        _save_wb(wb, out, task)
        if last_error is not None:
            status = f"error: {last_error}"[:200]
    try:
        wb.close()
    except Exception:
        pass
    append_jsonl(
        out_dir / "predictions.jsonl",
        {"id": task["id"], "output": f"outputs/{task['id']}.xlsx", "status": status, "mode": "agent"},
    )
    return status


async def predict_routed(
    complete,
    complete_chat: CompleteChat | None,
    model: str,
    task: dict,
    out_dir: Path,
    agent: str,
) -> str:
    from common import predict_task

    use = agent == "always" or (agent == "auto" and should_use_agent(task) and complete_chat is not None)
    if use and complete_chat is not None:
        return await predict_task_agent(complete_chat, model, task, out_dir)
    if agent == "auto" and complete_chat is not None:
        status = await predict_task(
            complete,
            model,
            task,
            out_dir,
            record_prediction=False,
        )
        if status.startswith("error:"):
            return await predict_task_agent(complete_chat, model, task, out_dir)
        append_jsonl(
            out_dir / "predictions.jsonl",
            {"id": task["id"], "output": f"outputs/{task['id']}.xlsx", "status": status},
        )
        return status
    return await predict_task(complete, model, task, out_dir)

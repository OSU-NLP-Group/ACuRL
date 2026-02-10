from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PORTLIKE_TOKEN = "«PORTLIKE»"


@dataclass(frozen=True)
class ScienceBoardTask:
    raw: Dict[str, Any]
    task_id: str
    task_type: str
    snapshot: str
    instruction: str
    init_steps: List[Dict[str, Any]]
    eval_steps: List[Dict[str, Any]]
    port: int
    touched_files: Dict[str, str]
    celestia_query: List[Dict[str, Any]]


def _normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


def load_scienceboard_task_json(
    task_json_path: str,
    port: int = 8000,
    task_id_override: Optional[str] = None,
) -> ScienceBoardTask:
    task_json_path = _normalize_path(task_json_path)
    raw = json.loads(Path(task_json_path).read_text(encoding="utf-8"))

    task_type = raw["type"]
    instruction = raw["instruction"]
    snapshot = raw.get("snapshot", "sci_bench")
    init_steps = raw.get("initialize", [])
    eval_steps = raw.get("evaluate", [])
    celestia_query = raw.get("query", [])

    # Prefer explicit id; otherwise use override (e.g., filename stem) for stability.
    if raw.get("id"):
        task_id = str(raw["id"])
    elif task_id_override:
        task_id = str(task_id_override)
    else:
        task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"scienceboard:{task_json_path}"))

    touched_files: Dict[str, str] = {}

    return ScienceBoardTask(
        raw=raw,
        task_id=task_id,
        task_type=task_type,
        snapshot=snapshot,
        instruction=instruction,
        init_steps=init_steps if isinstance(init_steps, list) else [],
        eval_steps=eval_steps if isinstance(eval_steps, list) else [],
        port=int(port),
        touched_files=touched_files,
        celestia_query=celestia_query if isinstance(celestia_query, list) else [],
    )


def _subst_portlike(obj: Any, port: int) -> Any:
    if isinstance(obj, str):
        return obj.replace(PORTLIKE_TOKEN, str(port))
    if isinstance(obj, list):
        return [_subst_portlike(x, port) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst_portlike(v, port) for k, v in obj.items()}
    return obj


def _maybe_wait_step(wait: Optional[float]) -> List[Dict[str, Any]]:
    if wait is None:
        return []
    try:
        w = float(wait)
    except Exception:
        return []
    if w <= 0:
        return []
    return [{"type": "sleep", "parameters": {"seconds": w}}]


def _sb_init_to_setup_steps(sb: ScienceBoardTask) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []

    for item in sb.init_steps:
        func = item.get("func")
        if not isinstance(func, str):
            continue

        wait = item.get("wait")

        if func == "execute":
            steps.append(
                {
                    "type": "execute",
                    "parameters": {
                        "command": _subst_portlike(item.get("command"), sb.port),
                        "shell": bool(item.get("shell", False)),
                    },
                }
            )
            steps.extend(_maybe_wait_step(wait))
        elif func == "launch":
            steps.append(
                {
                    "type": "launch",
                    "parameters": {
                        "command": _subst_portlike(item.get("command"), sb.port),
                        "shell": bool(item.get("shell", False)),
                    },
                }
            )
            steps.extend(_maybe_wait_step(wait))
        elif func == "touch":
            steps.append(
                {
                    "type": "write_file",
                    "parameters": {"path": item["path"], "content": item["text"]},
                }
            )
            # Keep touched_files for potential downstream consumers, even if we
            # don't currently evaluate TeX/Lean tasks here.
            path = item.get("path")
            text = item.get("text")
            if isinstance(path, str) and isinstance(text, str):
                sb.touched_files[path] = text
            steps.extend(_maybe_wait_step(wait))
        elif func == "opt":
            steps.append({"type": "opt", "parameters": {"depth": int(item["depth"])}})
            steps.extend(_maybe_wait_step(wait))

        # KAlgebra init calls (host-side HTTP)
        elif func in {"tab", "func_2d", "func_3d"}:
            params: Dict[str, Any] = {"port": sb.port}
            if func == "tab":
                params["index"] = int(item["index"])
            elif func == "func_2d":
                params["expr"] = str(item["expr"])
            elif func == "func_3d":
                params["expr"] = str(item["expr"])
            steps.append({"type": f"kalgebra_{func}", "parameters": params})
            steps.extend(_maybe_wait_step(wait))
        else:
            # Unknown init func: keep compatibility.
            steps.append({"type": "sleep", "parameters": {"seconds": 0.0}})

    return steps


def to_desktopenv_task_config(
    task_json_path: str,
    port: int = 8000,
    task_id_override: Optional[str] = None,
) -> Dict[str, Any]:
    sb = load_scienceboard_task_json(task_json_path, port=port, task_id_override=task_id_override)
    setup_steps = _sb_init_to_setup_steps(sb)

    return {
        "id": sb.task_id,
        # ScienceBoard compatibility: keep task type explicitly for downstream categorization.
        "type": sb.task_type,
        "instruction": sb.instruction,
        # Use the real VM snapshot name (ScienceBoard tasks typically use "sci_bench").
        # This keeps compatibility with env_manager.py categorization and DesktopEnv snapshot selection.
        "snapshot": sb.snapshot,
        "config": setup_steps,
        "evaluator": {
            "func": "scienceboard",
            "task_type": sb.task_type,
            "port": sb.port,
            "eval_steps": sb.eval_steps,
            "touched_files": sb.touched_files,
            "celestia_query": sb.celestia_query,
        },
    }



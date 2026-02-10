from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


def _safe_eval_lambda(expr: str) -> Callable:
    fn = eval(expr)
    if not callable(fn):
        raise ValueError(f"Expected callable from {expr!r}")
    return fn


def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 120) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def _http_get(url: str, timeout: int = 120) -> requests.Response:
    return requests.get(url, timeout=timeout)


def _kalgebra_get_vars(vm_ip: str, port: int) -> Dict[str, str]:
    resp = _http_get(f"http://{vm_ip}:{port}/vars", timeout=120)
    resp.raise_for_status()
    return resp.json()


def _kalgebra_post_func(vm_ip: str, port: int, dim: int, points: List[List[float]]) -> List[Dict[str, Any]]:
    resp = requests.post(f"http://{vm_ip}:{port}/func/{dim}d", json=points, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def _eval_kalgebra(vm_ip: str, port: int, item: Dict[str, Any]) -> bool:
    t = item["type"]
    if t in {"val", "var"}:
        vars_ = _kalgebra_get_vars(vm_ip, port)
        key = item["key"]
        value = item["value"]
        if t == "var":
            if value == "#UNDEF":
                return key not in vars_
            return vars_.get(key) == value
        left = float(vars_[key])
        right = float(value)
        return abs(left - right) <= 1e-6

    if t == "eqn":
        key = item["key"]
        expected = item["value"]
        if key == "#SIZE":
            eqns = _kalgebra_post_func(vm_ip, port, dim=2, points=[])
            return len(eqns) == expected
        points = key
        dim = len(points[0]) if points else 2
        eqns = _kalgebra_post_func(vm_ip, port, dim=dim, points=points)
        for eqn in eqns:
            ok = True
            for k, v in expected.items():
                if str(eqn.get(k)) != str(v):
                    ok = False
                    break
            if ok:
                return True
        return False

    raise ValueError(f"Unknown KAlgebra eval type: {t}")


def _celestia_dump(vm_ip: str, port: int, query: List[Dict[str, Any]]) -> Dict[str, Any]:
    resp = requests.post(f"http://{vm_ip}:{port}/dump", data=json.dumps(query), timeout=120)
    resp.raise_for_status()
    return resp.json()


def _eval_info_like(dump: Dict[str, Any], item: Dict[str, Any]) -> bool:
    pred: Callable[[Any, Any], bool] = lambda left, right: left == right

    key_expr = item["key"]
    if isinstance(key_expr, str) and key_expr.startswith("lambda"):
        hkey = _safe_eval_lambda(key_expr)
    else:
        hkey = lambda d: d[key_expr]

    if "pred" in item:
        pred = _safe_eval_lambda(item["pred"])
    return bool(pred(hkey(dump), item["value"]))


def _parse_stop_from_action(action: Any) -> Tuple[str, List[str], str]:
    """
    Best-effort extraction of ScienceBoard-style stop_type/stop_args from the last env action.

    We support:
    - string special actions: "DONE" / "FAIL"
    - dict special actions: {"action_type":"DONE", "content":"ANS 4", ...}
    """
    # Default: if we can't extract an answer, treat as DONE with no args.
    if isinstance(action, str):
        a = action.strip().upper()
        if a in {"DONE", "FAIL"}:
            return a, [], ""
        # Sometimes raw text might contain an answer.
        content = action
    elif isinstance(action, dict):
        a = str(action.get("action_type", "")).strip().upper()
        if a in {"DONE", "FAIL"}:
            content = str(action.get("content", "") or "")
        else:
            content = str(action.get("content", "") or "")
    else:
        return "DONE", [], ""

    # Try to parse "ANS <arg1> <arg2> ..."
    m = re.search(r"\bANS\b\s*[:：]?\s*(.*)", content, flags=re.IGNORECASE)
    if not m:
        return "DONE", [], content
    tail = (m.group(1) or "").strip()
    if tail == "":
        return "ANS", [], content
    # Split like ScienceBoard's args list (strings)
    return "ANS", tail.split(), content


def _llm_judge_stop_ans(content: str, expected_args: List[str]) -> bool:
    """
    LLM judge for ScienceBoard-style stop/ANS tasks.
    Returns True if the content expresses that the final answer equals expected_args.
    """
    import os

    content = (content or "").strip()
    if not content:
        return False

    expected_args = [str(x) for x in (expected_args or [])]
    expected = " ".join(expected_args).strip()
    if not expected:
        return False

    # Default judge model
    model = os.getenv("SCIENCEBOARD_STOP_JUDGE_MODEL", "gpt-5-mini")
    # Optional: route judge calls to a vLLM OpenAI-compatible endpoint if provided
    vllm_base_url = os.getenv("SCIENCEBOARD_STOP_JUDGE_VLLM_BASE_URL")

    try:
        if vllm_base_url:
            from agent_system.reward_manager.utils import vllm_OpenaiEngine as _Engine

            engine = _Engine(model=model, base_url=vllm_base_url)
        else:
            from agent_system.reward_manager.utils import OpenaiEngine as _Engine

            engine = _Engine(model=model)
    except Exception:
        # If the engine can't be constructed (missing deps / missing key), do not change behavior.
        return False

    system = (
        "You are a strict answer judge.\n"
        "Decide whether the model output states/provides a final answer that equals the expected answer.\n"
        "The output may be in Chinese or English. Ignore reasoning; only judge the final answer value.\n"
        "Respond with exactly one token: yes or no."
    )
    user = f"Expected answer: {expected}\nModel output: {content}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        resp_list = engine.generate(messages, max_new_tokens=512, temperature=0)
        resp = (resp_list[0] if isinstance(resp_list, list) and resp_list else "").strip().lower()
    except Exception:
        return False

    return bool(re.match(r"^(yes|true|success)\b", resp))


def _eval_stop_steps(env: Any, eval_steps: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluate and remove ScienceBoard stop steps:
      {"type":"stop","value":"ANS","args":["4"]}

    Returns (ok, remaining_steps).
    """
    stop_steps = [it for it in eval_steps if it.get("type") == "stop"]
    remaining = [it for it in eval_steps if it.get("type") != "stop"]
    if not stop_steps:
        return True, eval_steps

    last_action = None
    try:
        hist = getattr(env, "action_history", [])
        if isinstance(hist, list) and hist:
            last_action = hist[-1]
    except Exception:
        last_action = None

    stop_type, stop_args, stop_content = _parse_stop_from_action(last_action)

    for st in stop_steps:
        expected_type = str(st.get("value", "")).strip()
        if expected_type == "":
            return False, remaining
        if expected_type.upper() != stop_type.upper():
            # If strict stop type mismatched, allow optional LLM judge for ANS tasks.
            if expected_type.upper() == "ANS":
                expected_args = st.get("args", [])
                if isinstance(expected_args, list) and _llm_judge_stop_ans(str(stop_content or ""), [str(x) for x in expected_args]):
                    continue
            return False, remaining
        # Only ANS uses args in ScienceBoard tasks.
        if expected_type.upper() == "ANS":
            expected_args = st.get("args", [])
            if not isinstance(expected_args, list):
                return False, remaining
            expected_args = [str(x) for x in expected_args]
            got_args = [str(x) for x in (stop_args or [])]
            if expected_args != got_args:
                # Strict args mismatch: optional LLM judge.
                if _llm_judge_stop_ans(str(stop_content or ""), expected_args):
                    continue
                return False, remaining

    return True, remaining


def evaluate_scienceboard(env, evaluator: Dict[str, Any]) -> bool:
    vm_ip = getattr(env, "vm_ip")
    server_port = int(getattr(env, "server_port", 5000))
    task_type = evaluator.get("task_type")
    port = int(evaluator.get("port", 8000))
    eval_steps = evaluator.get("eval_steps", [])

    ok, eval_steps = _eval_stop_steps(env, eval_steps if isinstance(eval_steps, list) else [])
    if not ok:
        return False

    if task_type == "KAlgebra":
        for it in eval_steps:
            if not _eval_kalgebra(vm_ip, port, it):
                return False
        return True

    if task_type == "Celestia":
        dump = _celestia_dump(vm_ip, port, evaluator.get("celestia_query", []) or [])
        return all(_eval_info_like(dump, it) for it in eval_steps)

    return False



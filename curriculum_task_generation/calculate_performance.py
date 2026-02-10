#!/usr/bin/env python3
"""
Compute per-task performance from rollout result files and write ONE detailed JSON.

It scans:
  results_dir/osworld.*/training_step_0/<rollout_id>/<rollout_id>.json

and uses `cuajudge_success_rate` as the rollout score.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task_pool(task_pool_file: str) -> List[str]:
    data = _read_json(task_pool_file)
    return data["tasks"]


def load_rollout_scores(results_dir: str) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    scores: Dict[str, List[float]] = {}
    goals: Dict[str, str] = {}

    for item in os.listdir(results_dir):
        task_dir = os.path.join(results_dir, item)
        if not (os.path.isdir(task_dir)):
            continue
        step_dir = os.path.join(task_dir, "training_step_0")

        for rollout_id in os.listdir(step_dir):
            result_file = os.path.join(step_dir, rollout_id, f"{rollout_id}.json")
            data = _read_json(result_file)
            goals.setdefault(item, data["task_goal"])
            scores.setdefault(item, []).append(float(data["cuajudge_success_rate"]))

    return scores, goals


def write_detailed_json(
    *,
    tasks: List[str],
    task_scores: Dict[str, List[float]],
    task_goals: Dict[str, str],
    output_file: str,
) -> str:
    output_dir = os.path.dirname(output_file) or "."
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    detailed_path = os.path.join(output_dir, f"{base_name}.json")

    detailed_tasks: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task
        xs = task_scores.get(task_id, [])
        entry = {
            "task_id": task_id,
            "accuracy": (sum(xs) / len(xs)) if xs else None,
            "original_task": {"task_id": task_id, "instruction": task_goals.get(task_id)},
        }
        detailed_tasks.append(entry)

    out = {"task_performances": detailed_tasks}

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return detailed_path


def main() -> None:
    p = argparse.ArgumentParser(description="Write per-task performance JSON")
    p.add_argument("--results_dir", required=True, help="Directory containing training results")
    p.add_argument("--task_pool_file", required=True, help="Path to task pool JSON file")
    p.add_argument("--output_file", required=True, help="Base output path (script writes <base>_detailed.json)")
    args = p.parse_args()

    task_scores, task_goals = load_rollout_scores(args.results_dir)
    tasks = load_task_pool(args.task_pool_file)
    detailed_path = write_detailed_json(
        tasks=tasks,
        task_scores=task_scores,
        task_goals=task_goals,
        output_file=args.output_file,
    )

    print(f"Wrote detailed file: {detailed_path}")


if __name__ == "__main__":
    main()

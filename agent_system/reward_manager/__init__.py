"""Reward manager package for ACuRL agent system.

This package contains:
- `EpisodeRewardManager`: rule-/episode-based reward shaping for RL training
- `CUAJudgeEvaluator`: LLM-based trajectory judge for OSWorld-style tasks

"""

from .episode import EpisodeRewardManager
from .cuajudge import CUAJudgeEvaluator

__all__ = [
    "EpisodeRewardManager",
    "CUAJudgeEvaluator",
]



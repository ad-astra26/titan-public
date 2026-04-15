"""
titan_plugin.logic.mood — Mood Engine and addon system.

Exports:
    MoodEngine: Aggregates growth metrics + hot-loaded addons into mood score.
    MoodRegistry: Dynamic addon loader from titan_plugin/addons/.
    AbstractMoodAddon: Base class for third-party mood addons.
"""
from .engine import MoodEngine, MoodRegistry
from .base import AbstractMoodAddon

__all__ = [
    "MoodEngine",
    "MoodRegistry",
    "AbstractMoodAddon",
]

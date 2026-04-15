"""
titan_plugin.logic — Biological cycles and decision-making logic.

Exports:
    MeditationEpoch: 6-hour Small Epoch memory consolidation.
    RebirthBackup: Meditation-triggered sovereign backup to Arweave.
    ReflectionLogic: LLM policy guard for personality drift protection.
"""
from .meditation import MeditationEpoch
from .backup import RebirthBackup
from .reflection import ReflectionLogic

__all__ = [
    "MeditationEpoch",
    "RebirthBackup",
    "ReflectionLogic",
]

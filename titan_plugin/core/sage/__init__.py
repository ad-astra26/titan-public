"""
titan_plugin.core.sage — TorchRL replay buffer and RL state recording.

Exports:
    SageRecorder: LazyMemmapStorage replay buffer with 3072→128 dim projection.
"""
from .recorder import SageRecorder

__all__ = ["SageRecorder"]

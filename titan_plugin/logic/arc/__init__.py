"""
titan_plugin/logic/arc — ARC-AGI-3 Interactive Reasoning Adapter.

Connects Titan's Trinity architecture to ARC-AGI-3 game environments
via the official arc_agi SDK (v0.9.6):
  - GridPerception: game frame → Trinity tensor features (15D)
  - ActionMapper: nervous system signals → game action selection
  - ArcSession: episode management + RL feedback loop
  - ArcSDKBridge: wrapper around official arc_agi SDK

Usage:
    from titan_plugin.logic.arc import ArcSDKBridge, GridPerception, ActionMapper, ArcSession

    sdk = ArcSDKBridge()
    sdk.initialize()
    perception = GridPerception()
    mapper = ActionMapper()
    session = ArcSession(sdk, perception, mapper, ns_programs=loaded_programs)

    result = session.play_game("ls20")
"""
from .grid_perception import GridPerception
from .action_mapper import ActionMapper
from .api_client import ArcSDKBridge, Frame
from .session import ArcSession, EpisodeResult, SessionReport, HAOVTracker, Hypothesis
from .state_memory import StateActionMemory
from .forward_model import ForwardModel

__all__ = [
    "GridPerception", "ActionMapper", "ArcSDKBridge", "Frame",
    "ArcSession", "EpisodeResult", "SessionReport", "StateActionMemory",
    "ForwardModel", "HAOVTracker", "Hypothesis",
]

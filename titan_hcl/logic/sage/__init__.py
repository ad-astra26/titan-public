"""
titan_hcl.logic.sage — The Sage Pipeline: research, decision-making, and safety.

Lazy imports via __getattr__ for deferred loading (callers import from submodules
directly).

SageGatekeeper + SageScholar (the IQL gatekeeper/scholar, ~204MB torch) RETIRED
with the offline-RL subsystem (RFP_synthesis_decision_authority P1).

Exports:
    SageGuardian: 3-tier safety (keyword → semantic → LLM veto).
    StealthSageResearcher: Multi-modal autonomous research engine.
"""

__all__ = [
    "SageGuardian",
    "StealthSageResearcher",
]


def __getattr__(name: str):
    if name == "SageGuardian":
        from .guardian import SageGuardian
        return SageGuardian
    if name == "StealthSageResearcher":
        from .researcher import StealthSageResearcher
        return StealthSageResearcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

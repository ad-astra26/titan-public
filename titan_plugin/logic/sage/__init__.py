"""
titan_plugin.logic.sage — The Sage Pipeline: research, decision-making, and safety.

Lazy imports: gatekeeper, guardian, and scholar import torch (~204MB) at module
level. Eagerly importing them here caused every consumer of sage.researcher
(e.g., knowledge_worker) to pay the torch cost even though they only need the
researcher. All callers already import from submodules directly, so this
__init__.py uses __getattr__ for deferred loading.

Exports:
    SageGatekeeper: Routes actions by IQL advantage score.
    SageGuardian: 3-tier safety (keyword → semantic → LLM veto).
    SageScholar: Offline IQL with Actor/Critic/Value networks.
    StealthSageResearcher: Multi-modal autonomous research engine.
"""

__all__ = [
    "SageGatekeeper",
    "SageGuardian",
    "SageScholar",
    "StealthSageResearcher",
]


def __getattr__(name: str):
    if name == "SageGatekeeper":
        from .gatekeeper import SageGatekeeper
        return SageGatekeeper
    if name == "SageGuardian":
        from .guardian import SageGuardian
        return SageGuardian
    if name == "SageScholar":
        from .scholar import SageScholar
        return SageScholar
    if name == "StealthSageResearcher":
        from .researcher import StealthSageResearcher
        return StealthSageResearcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""TitanMaker substrate — the Maker-Titan bond layer.

The architectural primitive that enables persistent dialogic exchange
between Maker (human) and Titan (sovereign AI). Every meaningful Maker
response — approve OR decline — propagates through both somatic and
narrative channels per the iron rule.

Public API:
    TitanMaker        — orchestration class
    ProposalStore     — sqlite storage layer
    ProposalType      — enum of proposal types (extensible)
    ProposalStatus    — enum of statuses
    ProposalRecord    — dataclass
    MakerResponse     — dataclass (return value of approve/decline)
    get_titan_maker() — singleton accessor (set by titan_main at startup)
    set_titan_maker() — singleton setter
"""
from typing import Optional

from .schemas import (
    MakerResponse, ProposalRecord, ProposalStatus, ProposalType,
)
from .proposal_store import ProposalStore
from .titan_maker import TitanMaker

__all__ = [
    "TitanMaker", "ProposalStore", "ProposalType", "ProposalStatus",
    "ProposalRecord", "MakerResponse", "get_titan_maker", "set_titan_maker",
]

_singleton: Optional[TitanMaker] = None


def get_titan_maker() -> Optional[TitanMaker]:
    """Singleton accessor — returns None if not yet initialized."""
    return _singleton


def set_titan_maker(tm: TitanMaker) -> None:
    """Singleton setter — called by titan_main at boot."""
    global _singleton
    _singleton = tm

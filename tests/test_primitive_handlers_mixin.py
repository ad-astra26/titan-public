"""OML Phase C piece 4 — PrimitiveHandlersMixin regression guard.

The meta primitive handlers were relocated from ``MetaReasoningEngine`` into a
shared ``PrimitiveHandlersMixin`` so the outer ``OuterMetaReasoningEngine``
(piece 5) can reuse them with no duplication. This guards that the relocation
stays byte-identical-in-behavior + plane-agnostic:

  • MetaReasoningEngine inherits the mixin (MRO);
  • the 12 handlers are DEFINED on the mixin (not the engine) and resolve via
    inheritance to the SAME function objects;
  • the handlers stay plane-agnostic — ZERO ``self._meta_cgn`` (the inner-CGN
    coupling must remain in the tick/conclude orchestration, NOT the handlers,
    so the outer engine can reuse them safely);
  • the mixin is importable standalone.
"""
import inspect

from titan_hcl.logic.meta_reasoning import (
    MetaReasoningEngine,
    PrimitiveHandlersMixin,
)

_RELOCATED = (
    "_prim_formulate", "_prim_recall", "_prim_hypothesize", "_prim_delegate",
    "_prim_delegate_gap_fill", "_check_delegate", "_prim_synthesize",
    "_prim_evaluate", "_save_checkpoint", "_prim_break", "_prim_spirit_self",
    "_prim_introspect",
)


def test_engine_inherits_mixin():
    assert PrimitiveHandlersMixin in MetaReasoningEngine.__mro__
    assert issubclass(MetaReasoningEngine, PrimitiveHandlersMixin)


def test_handlers_defined_on_mixin_not_engine():
    for name in _RELOCATED:
        assert name in PrimitiveHandlersMixin.__dict__, f"{name} not on mixin"
        # defined on the mixin, NOT redefined on the engine (no shadowing)
        assert name not in MetaReasoningEngine.__dict__, (
            f"{name} shadowed on MetaReasoningEngine")


def test_handlers_resolve_via_mro_to_mixin():
    for name in _RELOCATED:
        # the engine's bound resolution is the mixin's function object
        assert getattr(MetaReasoningEngine, name) is PrimitiveHandlersMixin.__dict__[name]


def test_handlers_are_plane_agnostic_no_meta_cgn():
    # The relocated handlers must NOT reference the inner meta-CGN engine —
    # that coupling stays in the tick/conclude orchestration. This is the
    # property that makes outer reuse safe.
    for name in _RELOCATED:
        src = inspect.getsource(PrimitiveHandlersMixin.__dict__[name])
        assert "self._meta_cgn" not in src, f"{name} touches self._meta_cgn"


def test_mixin_importable_standalone():
    # No engine instance required to reference the mixin / its methods.
    assert isinstance(PrimitiveHandlersMixin, type)
    assert callable(PrimitiveHandlersMixin.__dict__["_prim_formulate"])


def test_exactly_twelve_handlers_relocated():
    on_mixin = {n for n in PrimitiveHandlersMixin.__dict__
                if not n.startswith("__") and callable(PrimitiveHandlersMixin.__dict__[n])}
    assert on_mixin == set(_RELOCATED), f"mixin method set drifted: {on_mixin ^ set(_RELOCATED)}"

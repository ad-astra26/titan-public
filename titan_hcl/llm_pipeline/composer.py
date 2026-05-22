"""
titan_hcl.llm_pipeline.composer — DialogueComposerFacade.

Wraps `titan_hcl.logic.dialogue_composer.DialogueComposer` with:
  - Per-call instance management (cached singleton; DialogueComposer
    construction loads grammar rules + composition templates, so
    re-instantiating per call would waste ~50ms × N callsites).
  - State gathering via `state_gather.gather_felt_state_and_vocab`.
  - Hormone-shift derivation from InputExtractor signal.
  - Confidence-gated output (≥0.3 → composed; else empty pre_text,
    signaling caller to fall through to LLM).

Replaces the inline ~40-LOC DialogueComposer block duplicated 5× today
(chat.py:362-400 + chat_pipeline.py:283-321 + core/plugin.py:2738-2776
+ dashboard.py:9825 + autonomous_language_pipeline.py:916).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from . import state_gather

logger = logging.getLogger(__name__)

_composer_singleton: Optional[Any] = None  # logic.dialogue_composer.DialogueComposer
_input_extractor_singleton: Optional[Any] = None  # logic.interface_input.InputExtractor


def _get_composer():
    """Cached DialogueComposer singleton (grammar rules load is heavy)."""
    global _composer_singleton
    if _composer_singleton is None:
        try:
            from titan_hcl.logic.dialogue_composer import DialogueComposer
            _composer_singleton = DialogueComposer()
        except Exception as e:
            logger.warning(
                "[llm_pipeline.composer] DialogueComposer init failed: %s", e
            )
            _composer_singleton = False  # sentinel — failed, don't retry every call
    return _composer_singleton or None


def _get_input_extractor():
    """Cached InputExtractor singleton (lightweight but called per chat)."""
    global _input_extractor_singleton
    if _input_extractor_singleton is None:
        try:
            from titan_hcl.logic.interface_input import InputExtractor
            _input_extractor_singleton = InputExtractor()
        except Exception as e:
            logger.warning(
                "[llm_pipeline.composer] InputExtractor init failed: %s", e
            )
            _input_extractor_singleton = False
    return _input_extractor_singleton or None


def reset_singletons() -> None:
    """Test helper — clear cached singletons so a fresh mock can be installed."""
    global _composer_singleton, _input_extractor_singleton
    _composer_singleton = None
    _input_extractor_singleton = None


class ComposeResult:
    """Result of compose_pre() — passed back to callers.

    Attributes:
        pre_text:    Composed felt-state sentence (empty string if no
                     composition was confident enough — caller proceeds
                     straight to LLM).
        confidence:  Composition confidence 0.0-1.0.
        intent:      Detected intent (empathize / ask_question / etc.).
        level:       CompositionEngine level used (1-7).
        composed:    True iff confidence >= threshold.
        felt_state:  130D-132D felt vector that was used (for downstream
                     OVG chain_state assembly).
        vocabulary:  Vocab list that was used.
    """

    __slots__ = (
        "pre_text", "confidence", "intent", "level",
        "composed", "felt_state", "vocabulary",
    )

    def __init__(
        self,
        pre_text: str = "",
        confidence: float = 0.0,
        intent: str = "",
        level: int = 0,
        composed: bool = False,
        felt_state: Optional[list] = None,
        vocabulary: Optional[list] = None,
    ):
        self.pre_text = pre_text
        self.confidence = confidence
        self.intent = intent
        self.level = level
        self.composed = composed
        self.felt_state = felt_state or []
        self.vocabulary = vocabulary or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_text": self.pre_text,
            "confidence": self.confidence,
            "intent": self.intent,
            "level": self.level,
            "composed": self.composed,
        }


async def compose_pre(
    message: str,
    user_id: str = "",
    *,
    channel: str = "chat",
    min_confidence: float = 0.3,
    max_level: int = 7,
    felt_state: Optional[list] = None,
    vocabulary: Optional[list] = None,
    hormone_shifts: Optional[dict] = None,
) -> ComposeResult:
    """Run the felt-state composition pre-LLM and return a sentence to prepend.

    Args:
        message:        Incoming user message (text only, no context).
        user_id:        Identifier — passed to InputExtractor for valence shift.
        channel:        "chat" / "x_post" / "agent" / "pitch" — reserved for
                        future channel-specific composition; default behavior
                        is uniform across channels today.
        min_confidence: Threshold below which composition is treated as
                        "fallback to LLM only". Default 0.3 matches the
                        existing inline blocks.
        max_level:      Maximum CompositionEngine level (1-7). Default 7
                        (highest sophistication available).
        felt_state:     Optional caller-provided 130D state vector. When
                        supplied (together with `vocabulary`), skips the
                        auto-gather DB reads. Used by batch / research
                        scripts that already hold state in-memory.
        vocabulary:     Optional caller-provided vocab list. Same shape as
                        what `state_gather.gather_felt_state_and_vocab`
                        returns. Skips auto-gather when supplied.
        hormone_shifts: Optional caller-provided hormone shifts dict. When
                        supplied, skips InputExtractor signal derivation.
                        Used by callers that compute shifts from a different
                        source (e.g., neuromod state delta over a chat round).

    Returns:
        ComposeResult. When `composed=True`, caller prepends `pre_text` to
        the LLM message. When False, caller proceeds without prefix.
    """
    # 1. Gather felt_state + vocabulary (skip if caller-provided)
    if felt_state is None or vocabulary is None:
        try:
            _felt, _vocab = await state_gather.gather_felt_state_and_vocab()
        except Exception as e:
            logger.debug(
                "[llm_pipeline.compose_pre] state gather failed: %s", e
            )
            return ComposeResult()
        felt_state = felt_state if felt_state is not None else _felt
        vocabulary = vocabulary if vocabulary is not None else _vocab

    if not felt_state or not vocabulary:
        # No state to compose from — silently fall through to LLM
        return ComposeResult(felt_state=felt_state or [],
                             vocabulary=vocabulary or [])

    # 2. Derive hormone_shifts via InputExtractor (skip if caller-provided)
    if hormone_shifts is None:
        extractor = _get_input_extractor()
        hormone_shifts = {}
        if extractor is not None:
            try:
                signal = extractor.extract(message, user_id)
                hormone_shifts = state_gather.build_hormone_shifts(signal)
            except Exception as e:
                logger.debug(
                    "[llm_pipeline.compose_pre] InputExtractor failed: %s", e
                )

    # 3. Compose via DialogueComposer
    composer = _get_composer()
    if composer is None:
        return ComposeResult(felt_state=felt_state, vocabulary=vocabulary)

    try:
        result = composer.compose_response(
            felt_state=felt_state,
            vocabulary=vocabulary,
            hormone_shifts=hormone_shifts or None,
            message_keywords=message.lower().split()[:10],
            max_level=max_level,
        )
    except Exception as e:
        logger.warning(
            "[llm_pipeline.compose_pre] DialogueComposer raised: %s", e
        )
        return ComposeResult(felt_state=felt_state, vocabulary=vocabulary)

    confidence = float(result.get("confidence", 0.0))
    composed = bool(result.get("composed")) and confidence >= min_confidence

    if not composed:
        return ComposeResult(
            confidence=confidence,
            intent=result.get("intent", ""),
            level=int(result.get("level", 0)),
            composed=False,
            felt_state=felt_state,
            vocabulary=vocabulary,
        )

    return ComposeResult(
        pre_text=result.get("response", "").strip(),
        confidence=confidence,
        intent=result.get("intent", ""),
        level=int(result.get("level", 0)),
        composed=True,
        felt_state=felt_state,
        vocabulary=vocabulary,
    )

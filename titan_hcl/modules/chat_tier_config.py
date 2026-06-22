"""
titan_hcl.modules.chat_tier_config — adaptive prompt classifier.

D-SPEC-79 (Phase 2 Chunk ζ.0, 2026-05-18). Loads the [chat] config section
and provides a single-call `classify(prompt)` API that PreHook uses to
decide WHICH features to load and WHICH abstract model class to invoke.

Design intent (from Maker, 2026-05-18):
    - Tiers are config-tunable; adding a tier = one [[chat.tiers]] block.
    - Features are orthogonal boolean flags toggled per-tier.
    - model_class is provider-agnostic; resolved at runtime by
      InferenceProvider.resolve_model_class().
    - First-match wins; classifier is construction-cheap, compiles regexes
      once. Per-request classify() is O(num_tiers * num_patterns) regex test
      which is microseconds for the current ~12 patterns.

Adding a tier:
    [[chat.tiers]]
    name = "newtier"
    model_class = "light"
    detect.max_chars = 80
    detect.regex_any = ["^pattern$"]
    features = ["directives", "felt_state"]

Adding a feature:
    Append the feature key to a tier's `features` list AND add the gate
    `if "feature_name" in active_features:` in agno_hooks.py PreHook.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierConfig:
    """One tier from [[chat.tiers]] — compiled regexes + feature set."""
    name: str
    model_class: str
    features: frozenset[str]
    max_chars: int | None
    min_chars: int | None
    regex_any: tuple[re.Pattern, ...]
    topic_memory_regex: tuple[re.Pattern, ...]
    # ζ.6 (D-SPEC-79, 2026-05-18) — per-tier response cap. When set, the
    # router swaps agent.model.max_tokens alongside model.id. Greeting tier
    # uses ~60-80 (no need for 1200-char replies to "Hi"); reasoning uses
    # the default (no cap). None = leave the model's default in place.
    max_tokens: int | None
    # ζ.7 (2026-06-11) — per-tier reply-length GUIDANCE injected into the
    # agent's instructions so the model PLANS a complete reply within budget
    # (and finishes its thought) instead of being hard-cut at max_tokens.
    # max_tokens stays the safety ceiling ABOVE the guided length. None = no
    # guidance (model unconstrained beyond the ceiling).
    reply_guidance: str | None = None

    def matches(self, prompt: str) -> bool:
        """True if this tier's detection criteria apply to `prompt`."""
        n = len(prompt)
        if self.max_chars is not None and n > self.max_chars:
            return False
        if self.min_chars is not None and n < self.min_chars:
            return False
        if self.regex_any:
            return any(p.search(prompt) for p in self.regex_any)
        # No regex constraint → length-bound match (or unconditional if no bounds)
        return self.max_chars is not None or self.min_chars is not None

    def dynamic_features(self, prompt: str) -> frozenset[str]:
        """Sub-features triggered by per-tier regex (e.g. topic_memory)."""
        dyn: set[str] = set()
        if self.topic_memory_regex and any(
            p.search(prompt) for p in self.topic_memory_regex
        ):
            dyn.add("topic_memory")
        return frozenset(dyn)


def _compile_regex_list(items: Iterable[str]) -> tuple[re.Pattern, ...]:
    compiled: list[re.Pattern] = []
    for raw in items or ():
        try:
            compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error as e:
            logger.warning(
                "[chat_tier_config] Bad regex %r skipped: %s", raw, e
            )
    return tuple(compiled)


def _build_tier(spec: dict[str, Any], default_model_class: str) -> TierConfig:
    detect = spec.get("detect", {}) or {}
    _mt = spec.get("max_tokens")
    return TierConfig(
        name=spec["name"],
        model_class=spec.get("model_class", default_model_class),
        features=frozenset(spec.get("features", ()) or ()),
        max_chars=detect.get("max_chars"),
        min_chars=detect.get("min_chars"),
        regex_any=_compile_regex_list(detect.get("regex_any", ())),
        topic_memory_regex=_compile_regex_list(detect.get("topic_memory_regex", ())),
        max_tokens=int(_mt) if _mt is not None else None,
        reply_guidance=(str(_rg).strip() or None) if (_rg := spec.get("reply_guidance")) is not None else None,
    )


@dataclass(frozen=True)
class ClassifyResult:
    """Outcome of one classify() call."""
    tier: TierConfig
    active_features: frozenset[str]   # tier.features ∪ dynamic_features


class ChatTierClassifier:
    """Classifier built once from [chat] config; classify() per request.

    Usage in agno_worker startup:
        from titan_hcl.modules.chat_tier_config import ChatTierClassifier
        classifier = ChatTierClassifier.from_config(plugin.config)

    Usage in PreHook:
        result = classifier.classify(prompt_text)
        model_id = provider.resolve_model_class(result.tier.model_class)
        if "directives" in result.active_features: ...
    """

    def __init__(
        self,
        tiers: list[TierConfig],
        default_tier: TierConfig,
        default_model_class: str,
        log_decisions: bool = False,
    ):
        if not tiers:
            raise ValueError("ChatTierClassifier requires at least one tier")
        self._tiers = tuple(tiers)
        self._default_tier = default_tier
        self._default_model_class = default_model_class
        self._log_decisions = log_decisions

    @classmethod
    def from_config(cls, full_config: dict[str, Any]) -> "ChatTierClassifier":
        """Construct from a config's [chat] section.

        Prefers an explicit `chat` section in `full_config` (callers that
        already hold the merged config, and tests that inject one); falls back
        to the live SHM-backed `get_params("chat")` when the passed config has
        no `chat` key (the config-as-SHM canonical read — the agno_worker is
        spawned with only the merged [agent]+[inference] block, so its
        `_full_config` typically lacks [chat]). Restores the `full_config`
        parameter that the Phase B SHM sweep (2c43f78db) had left unused.
        """
        if isinstance(full_config, dict) and "chat" in full_config:
            chat_cfg = full_config.get("chat") or {}
        else:
            chat_cfg = get_params("chat") or {}
        default_model_class = chat_cfg.get("default_model_class", "heavy")
        default_name = chat_cfg.get("default_tier", "casual")
        log_decisions = bool(chat_cfg.get("classifier_log_decisions", False))

        tier_specs = chat_cfg.get("tiers", []) or []
        if not tier_specs:
            # No tiers configured → single passthrough tier with full features.
            logger.warning(
                "[chat_tier_config] No [[chat.tiers]] configured; "
                "falling back to passthrough tier with all features."
            )
            passthrough = TierConfig(
                name="passthrough",
                model_class=default_model_class,
                features=frozenset({
                    "directives", "felt_state", "history",
                    "cgn_grounding", "cgn_social_action",
                    "user_recognition", "topic_memory",
                    "reasoning_chain", "gatekeeper_state", "tools",
                }),
                max_chars=None,
                min_chars=None,
                regex_any=(),
                topic_memory_regex=(),
                max_tokens=None,
            )
            return cls([passthrough], passthrough, default_model_class, log_decisions)

        tiers = [_build_tier(spec, default_model_class) for spec in tier_specs]
        default = next(
            (t for t in tiers if t.name == default_name),
            tiers[0],
        )
        return cls(tiers, default, default_model_class, log_decisions)

    @property
    def tiers(self) -> tuple[TierConfig, ...]:
        return self._tiers

    @property
    def default_tier(self) -> TierConfig:
        return self._default_tier

    def classify(self, prompt: str) -> ClassifyResult:
        """First-match wins; falls back to default_tier when no match.

        active_features = tier.features ∪ tier.dynamic_features(prompt).
        """
        text = prompt or ""
        for tier in self._tiers:
            if tier.matches(text):
                active = tier.features | tier.dynamic_features(text)
                if self._log_decisions:
                    logger.info(
                        "[chat_tier] prompt=%r → tier=%s model_class=%s features=%s",
                        text[:60], tier.name, tier.model_class,
                        sorted(active),
                    )
                return ClassifyResult(tier=tier, active_features=active)

        # No match → default
        active = self._default_tier.features | self._default_tier.dynamic_features(text)
        if self._log_decisions:
            logger.info(
                "[chat_tier] prompt=%r → DEFAULT tier=%s model_class=%s features=%s",
                text[:60], self._default_tier.name,
                self._default_tier.model_class, sorted(active),
            )
        return ClassifyResult(tier=self._default_tier, active_features=active)

"""titan_hcl/logic/body_helpers.py — pure Body-worker reflex intuition.

rFP §3G Phase 10J — extracted from ``titan_hcl/modules/body_worker.py`` so that
``agno_hooks`` (via ``logic/reflex_intuition``) imports the reflex helper from a
torch/cgn-free ``logic/`` surface rather than reaching into the worker module
body (SPEC §11.B.4 — agno_hooks must contain ZERO
``from titan_hcl.modules.*_worker`` imports). ``body_worker`` self-imports this
back for its own internal reflex callsite.

Pure compute — keyword matching + scalar arithmetic over the Body 5D tensor
(interoception / proprioception / somatosensation / entropy / thermal). No bus,
no I/O, no state mutation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _compute_body_reflex_intuition(stimulus: dict, tensor: list) -> list:
    """
    Body's Intuition about which reflexes should fire.

    Body senses infrastructure stress and maps it to reflex confidence:
    - identity_check: Body feels identity stress (low interoception + threat)
    - metabolism_check: Body detects energy concerns (low energy + energy topics)
    - infra_check: Body senses infrastructure problems (low resources/network)
    """
    signals = []
    threat = stimulus.get("threat_level", 0.0)
    intensity = stimulus.get("intensity", 0.0)
    topics = stimulus.get("topics", [])
    topic = stimulus.get("topic", "general")

    # Tensor dims: [0]=interoception [1]=proprioception [2]=somatosensation [3]=entropy [4]=thermal
    intero = tensor[0] if len(tensor) > 0 else 0.5
    proprio = tensor[1] if len(tensor) > 1 else 0.5
    somato = tensor[2] if len(tensor) > 2 else 0.5
    entropy = tensor[3] if len(tensor) > 3 else 0.5
    thermal = tensor[4] if len(tensor) > 4 else 0.5

    # ── identity_check: Body senses sovereignty challenge ──
    # Low interoception (energy stress) + external threat = identity concern
    identity_conf = 0.0
    if threat > 0.2:
        identity_conf += threat * 0.5
    if intero < 0.4:
        identity_conf += (0.4 - intero) * 0.6
    # High entropy (disorder) → identity at risk
    if entropy < 0.3:
        identity_conf += 0.2
    if identity_conf > 0.05:
        signals.append({
            "reflex": "identity_check",
            "source": "body",
            "confidence": min(1.0, identity_conf),
            "reason": f"threat={threat:.2f} intero={intero:.2f} entropy={entropy:.2f}",
        })

    # ── metabolism_check: Body detects energy concern ──
    metab_conf = 0.0
    energy_topics = {"energy", "cost", "sol", "balance", "money", "crypto", "wallet"}
    if any(t in energy_topics for t in topics) or topic == "crypto":
        metab_conf += 0.4
    if intero < 0.4:
        metab_conf += (0.4 - intero) * 0.8
    if intensity > 0.6:
        metab_conf += 0.15
    if metab_conf > 0.05:
        signals.append({
            "reflex": "metabolism_check",
            "source": "body",
            "confidence": min(1.0, metab_conf),
            "reason": f"intero={intero:.2f} intensity={intensity:.2f} topic={topic}",
        })

    # ── infra_check: Body feels infrastructure stress ──
    infra_conf = 0.0
    tech_topics = {"technical", "system", "server", "performance", "health", "status"}
    if any(t in tech_topics for t in topics) or topic == "technical":
        infra_conf += 0.3
    if proprio < 0.4:
        infra_conf += (0.4 - proprio) * 0.6
    if somato < 0.4:
        infra_conf += (0.4 - somato) * 0.6
    if thermal < 0.3:
        infra_conf += 0.2
    if infra_conf > 0.05:
        signals.append({
            "reflex": "infra_check",
            "source": "body",
            "confidence": min(1.0, infra_conf),
            "reason": f"proprio={proprio:.2f} somato={somato:.2f} thermal={thermal:.2f}",
        })

    # ── Body also contributes weak signals for non-body reflexes ──
    # Memory recall: high thermal load → Body remembers strain
    if thermal < 0.4 and intensity > 0.3:
        signals.append({
            "reflex": "memory_recall",
            "source": "body",
            "confidence": min(0.5, intensity * 0.3),
            "reason": f"thermal={thermal:.2f} intensity={intensity:.2f}",
        })

    # Guardian shield: Body's fight-or-flight response to threat
    if threat > 0.3:
        signals.append({
            "reflex": "guardian_shield",
            "source": "body",
            "confidence": min(1.0, threat * 0.8),
            "reason": f"threat={threat:.2f} (fight-or-flight)",
        })

    # ── Action reflex signals (Body confirms resource availability) ──
    # Body gates actions by resource health — won't fire actions if body is stressed

    # Art/Audio: Body confirms CPU/thermal headroom
    if thermal > 0.5 and somato > 0.5:
        body_creative_ok = min(thermal, somato) * 0.4
        if any(kw in topics for kw in ("art", "create", "draw", "paint")):
            signals.append({
                "reflex": "art_generate",
                "source": "body",
                "confidence": body_creative_ok,
                "reason": f"thermal={thermal:.2f} somato={somato:.2f} (resources OK)",
            })
        if any(kw in topics for kw in ("audio", "music", "sound")):
            signals.append({
                "reflex": "audio_generate",
                "source": "body",
                "confidence": body_creative_ok,
                "reason": f"thermal={thermal:.2f} (resources OK)",
            })

    # Research: Body confirms network + energy for web access
    if proprio > 0.5 and intero > 0.3:
        if any(kw in topics for kw in ("research", "search", "find")):
            signals.append({
                "reflex": "research",
                "source": "body",
                "confidence": min(0.5, proprio * 0.3 + intero * 0.2),
                "reason": f"proprio={proprio:.2f} intero={intero:.2f} (network+energy OK)",
            })

    # Social post: Body confirms network health for external communication
    if proprio > 0.5:
        if any(kw in topics for kw in ("post", "tweet", "share")):
            signals.append({
                "reflex": "social_post",
                "source": "body",
                "confidence": min(0.4, proprio * 0.3),
                "reason": f"proprio={proprio:.2f} (network OK)",
            })

    if signals:
        logger.debug("[BodyWorker] Reflex Intuition: %d signals emitted", len(signals))
    return signals

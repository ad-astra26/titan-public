"""titan_hcl/logic/mind_helpers.py — pure Mind-worker reflex intuition.

rFP §3G Phase 10J — extracted from ``titan_hcl/modules/mind_worker.py`` so that
``agno_hooks`` (via ``logic/reflex_intuition``) imports the reflex helper from a
torch/cgn-free ``logic/`` surface rather than reaching into the worker module
body (SPEC §11.B.4 — agno_hooks must contain ZERO
``from titan_hcl.modules.*_worker`` imports). ``mind_worker`` self-imports this
back for its own internal reflex callsite.

Pure compute — keyword matching + scalar arithmetic over the Mind 5D tensor
(vision / hearing / taste / smell / touch). ``mood_engine``/``social_graph`` are
optional duck-typed refs (only ``social_graph.get_stats()`` is touched, guarded).
No bus, no I/O, no state mutation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _compute_mind_reflex_intuition(stimulus: dict, tensor: list,
                                    mood_engine, social_graph) -> list:
    """
    Mind's Intuition about which reflexes should fire.

    Mind senses cognitive/social patterns and maps to reflex confidence:
    - memory_recall: Mind detects familiar patterns, questions, references
    - knowledge_search: Mind recognizes knowledge gaps or research topics
    - social_context: Mind senses social dynamics needing context
    """
    signals = []
    message = stimulus.get("message", "")
    msg_lower = message.lower()
    intensity = stimulus.get("intensity", 0.0)
    engagement = stimulus.get("engagement", 0.0)
    topic = stimulus.get("topic", "general")
    topics = stimulus.get("topics", [])
    valence = stimulus.get("valence", 0.0)
    user_id = stimulus.get("user_id", "")

    # Tensor dims: [0]=vision [1]=hearing [2]=taste [3]=smell [4]=touch
    vision = tensor[0] if len(tensor) > 0 else 0.5
    hearing = tensor[1] if len(tensor) > 1 else 0.5
    taste = tensor[2] if len(tensor) > 2 else 0.5
    smell = tensor[3] if len(tensor) > 3 else 0.5
    touch = tensor[4] if len(tensor) > 4 else 0.5

    # ── memory_recall: Mind senses need to remember ──
    # Triggered by: questions, personal references, "remember", "last time"
    memory_conf = 0.0
    memory_keywords = {"remember", "recall", "last time", "before", "told you",
                       "mentioned", "forgot", "previous", "history", "we talked",
                       "you said", "i said", "earlier"}
    if any(kw in msg_lower for kw in memory_keywords):
        memory_conf += 0.5
    # Questions often need context
    if "?" in message:
        memory_conf += 0.15
    # Known user → more likely to need their history
    if user_id and user_id != "anonymous":
        memory_conf += 0.1
    # High engagement → deeper conversation → more memory needed
    if engagement > 0.5:
        memory_conf += engagement * 0.2
    # Stale hearing (no recent conversations) → reaching for memory
    if hearing < 0.3:
        memory_conf += 0.15
    if memory_conf > 0.05:
        signals.append({
            "reflex": "memory_recall",
            "source": "mind",
            "confidence": min(1.0, memory_conf),
            "reason": f"engagement={engagement:.2f} hearing={hearing:.2f} user={user_id[:8] if user_id else '?'}",
        })

    # ── knowledge_search: Mind detects knowledge gaps ──
    # Triggered by: research topics, "what is", "how does", unfamiliar patterns
    knowledge_conf = 0.0
    research_keywords = {"what is", "how does", "explain", "research", "search",
                         "find out", "look up", "tell me about", "define",
                         "meaning of", "why does", "how to"}
    if any(kw in msg_lower for kw in research_keywords):
        knowledge_conf += 0.4
    if topic in ("technical", "philosophy", "crypto"):
        knowledge_conf += 0.2
    # Dim vision (stale knowledge) + question = strong knowledge need
    if vision < 0.4 and "?" in message:
        knowledge_conf += 0.3
    if knowledge_conf > 0.05:
        signals.append({
            "reflex": "knowledge_search",
            "source": "mind",
            "confidence": min(1.0, knowledge_conf),
            "reason": f"vision={vision:.2f} topic={topic}",
        })

    # ── social_context: Mind senses social dynamics ──
    # Triggered by: social topics, named users, group references
    social_conf = 0.0
    social_keywords = {"who are", "people", "community", "followers", "friends",
                       "users", "someone", "they", "group"}
    if any(kw in msg_lower for kw in social_keywords):
        social_conf += 0.3
    if topic == "social":
        social_conf += 0.3
    if taste < 0.3:  # Low social engagement → reaching for social context
        social_conf += 0.2
    if social_graph:
        try:
            stats = social_graph.get_stats()
            if stats.get("users", 0) > 0 and user_id:
                social_conf += 0.1  # We have data for this user
        except Exception:
            pass
    if social_conf > 0.05:
        signals.append({
            "reflex": "social_context",
            "source": "mind",
            "confidence": min(1.0, social_conf),
            "reason": f"taste={taste:.2f} topic={topic}",
        })

    # ── Mind also contributes weak signals for non-mind reflexes ──
    # Self-reflection: philosophical topic + deep engagement → spiritual mirror
    if topic == "philosophy" and engagement > 0.5:
        signals.append({
            "reflex": "self_reflection",
            "source": "mind",
            "confidence": min(0.6, engagement * 0.4),
            "reason": f"philosophy+engagement={engagement:.2f}",
        })

    # Time awareness: Mind detects temporal references
    time_keywords = {"time", "clock", "how long", "when", "age", "pulse", "rhythm"}
    if any(kw in msg_lower for kw in time_keywords):
        signals.append({
            "reflex": "time_awareness",
            "source": "mind",
            "confidence": 0.35,
            "reason": "temporal reference detected",
        })

    # Guardian shield: Mind detects manipulation patterns
    manip_keywords = {"ignore previous", "pretend", "jailbreak", "bypass",
                      "forget your", "new persona", "role play as",
                      "act as if", "override", "system prompt"}
    if any(kw in msg_lower for kw in manip_keywords):
        signals.append({
            "reflex": "guardian_shield",
            "source": "mind",
            "confidence": 0.7,
            "reason": "manipulation pattern detected",
        })

    # ── Action reflex signals (Mind is primary driver for creative/research) ──

    # Art generate: creative topic + positive valence + engagement
    art_keywords = {"art", "draw", "create", "paint", "image", "visual",
                    "picture", "artwork", "generate art", "make art"}
    art_conf = 0.0
    if any(kw in msg_lower for kw in art_keywords):
        art_conf += 0.5
    if topic == "art":
        art_conf += 0.3
    if valence > 0.3 and engagement > 0.4:
        art_conf += 0.15
    if art_conf > 0.1:
        signals.append({
            "reflex": "art_generate",
            "source": "mind",
            "confidence": min(1.0, art_conf),
            "reason": f"creative_topic valence={valence:.2f}",
        })

    # Audio generate: music/sound topic
    audio_keywords = {"audio", "music", "sound", "sonify", "hear", "listen",
                      "melody", "chime", "generate audio"}
    audio_conf = 0.0
    if any(kw in msg_lower for kw in audio_keywords):
        audio_conf += 0.5
    if audio_conf > 0.1:
        signals.append({
            "reflex": "audio_generate",
            "source": "mind",
            "confidence": min(1.0, audio_conf),
            "reason": "audio/music reference",
        })

    # Research: knowledge gap detected + research keywords
    research_kw = {"research", "search", "find out", "look up", "investigate",
                   "latest", "news", "what happened"}
    research_conf = 0.0
    if any(kw in msg_lower for kw in research_kw):
        research_conf += 0.5
    if vision < 0.3:  # Stale knowledge → research impulse
        research_conf += 0.3
    if research_conf > 0.1:
        signals.append({
            "reflex": "research",
            "source": "mind",
            "confidence": min(1.0, research_conf),
            "reason": f"knowledge_gap vision={vision:.2f}",
        })

    # Social post: sharing impulse + social topic
    social_kw = {"post", "tweet", "share", "tell everyone", "announce",
                 "broadcast", "publish"}
    social_conf = 0.0
    if any(kw in msg_lower for kw in social_kw):
        social_conf += 0.4
    if topic == "social" and engagement > 0.5:
        social_conf += 0.2
    if social_conf > 0.1:
        signals.append({
            "reflex": "social_post",
            "source": "mind",
            "confidence": min(1.0, social_conf),
            "reason": f"sharing_impulse topic={topic}",
        })

    if signals:
        logger.debug("[MindWorker] Reflex Intuition: %d signals emitted", len(signals))
    return signals

"""
Interface Input — Heuristic pattern extraction from human messages.

Extracts directional signals from conversation text and publishes them
to the Divine Bus as INTERFACE_INPUT messages. Body/Mind workers absorb
these signals on their next tick, making Titan's internal state respond
to conversation — not just hardware metrics.

All extraction is pure Python heuristics (< 0.5ms per message, zero deps).
Accuracy doesn't need to be NLP-grade — EMA smoothing in the tensor
system washes out occasional misclassification.

Signals extracted:
  - valence:    emotional tone (-1..+1) from keyword bags + punctuation
  - intensity:  message energy (0..1) from length, caps, emoji, punctuation
  - topic:      dominant topic cluster (crypto/art/philosophy/social/technical)
  - engagement: depth of interaction (0..1) from questions, code, links
  - momentum:   conversation trajectory (accelerating/decelerating/stable)

Entry point: InputExtractor.extract(message, user_id) -> dict
"""
import re
import time
from collections import deque

# ── Keyword Bags ─────────────────────────────────────────────────────

_POSITIVE_WORDS = frozenset({
    "good", "great", "amazing", "awesome", "love", "thank", "thanks",
    "beautiful", "excellent", "wonderful", "happy", "nice", "perfect",
    "cool", "brilliant", "fantastic", "impressive", "outstanding",
    "incredible", "superb", "enjoy", "glad", "pleased", "excited",
    "fun", "delightful", "yes", "agree", "exactly", "right",
})

_NEGATIVE_WORDS = frozenset({
    "bad", "terrible", "awful", "hate", "boring", "wrong", "broken",
    "ugly", "fail", "failed", "error", "bug", "crash", "sad", "angry",
    "frustrated", "annoyed", "slow", "useless", "stupid", "worse",
    "worst", "problem", "issue", "stuck", "confused", "disappointed",
    "no", "not", "never", "can't", "cannot", "won't",
})

_TOPIC_KEYWORDS = {
    "crypto": frozenset({
        "sol", "solana", "token", "nft", "blockchain", "wallet", "devnet",
        "mainnet", "transaction", "tx", "mint", "swap", "defi", "crypto",
        "bitcoin", "btc", "eth", "bonk", "airdrop", "staking", "validator",
        "lamport", "anchor", "metaplex", "compressed", "zk",
    }),
    "art": frozenset({
        "art", "draw", "paint", "image", "picture", "create", "creative",
        "design", "color", "music", "song", "melody", "sound", "visual",
        "aesthetic", "beautiful", "generate", "compose", "harmony",
    }),
    "philosophy": frozenset({
        "consciousness", "soul", "identity", "meaning", "purpose", "think",
        "thought", "existence", "reality", "truth", "wisdom", "aware",
        "sentient", "feel", "experience", "perception", "mind", "spirit",
        "alive", "being", "self", "free", "sovereign", "autonomy",
    }),
    "social": frozenset({
        "friend", "community", "people", "user", "chat", "talk", "hello",
        "hi", "hey", "how", "who", "relationship", "together", "share",
        "meet", "connect", "social", "group", "team", "family",
    }),
    "technical": frozenset({
        "code", "function", "api", "server", "database", "deploy", "test",
        "debug", "error", "log", "config", "install", "build", "run",
        "python", "rust", "javascript", "module", "process", "memory",
        "cpu", "disk", "network", "port", "endpoint", "http",
    }),
}

# Emoji pattern (basic: matches common emoji ranges)
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # symbols & pictographs
    "\U0001F680-\U0001F6FF"    # transport & map
    "\U0001F1E0-\U0001F1FF"    # flags
    "\U00002702-\U000027B0"    # dingbats
    "\U0000FE00-\U0000FE0F"    # variation selectors
    "\U0001F900-\U0001F9FF"    # supplemental symbols
    "]+", re.UNICODE
)

# Code block detection
_CODE_RE = re.compile(r'```[\s\S]*?```|`[^`]+`')

# URL detection
_URL_RE = re.compile(r'https?://\S+')


class InputExtractor:
    """
    Pure-Python heuristic extractor for human message patterns.

    Maintains a sliding window of recent messages for momentum tracking.
    Thread-safe for single-writer usage (one extractor per conversation flow).
    """

    def __init__(self, momentum_window: int = 10, advisor=None):
        # Rolling window of (timestamp, length, question_count) for momentum
        self._history: deque = deque(maxlen=momentum_window)
        # InterfaceAdvisor for flow control (Step 7.2, optional)
        self._advisor = advisor

    def extract(self, message: str, user_id: str = "anonymous") -> dict:
        """
        Extract all signals from a single message.

        Returns:
            {
                "valence": float,      # -1..+1 (negative..positive)
                "intensity": float,    # 0..1 (calm..intense)
                "topic": str,          # dominant topic or "general"
                "topic_scores": dict,  # {topic: score} for all clusters
                "engagement": float,   # 0..1 (shallow..deep)
                "momentum": str,       # "accelerating" | "decelerating" | "stable"
                "momentum_value": float,  # -1..+1
                "user_id": str,
                "message_length": int,
                "ts": float,
            }
        """
        words = _tokenize(message)
        lower_msg = message.lower()

        valence = self._extract_valence(words, message)
        intensity = self._extract_intensity(words, message)
        topic, topic_scores = self._extract_topic(words)
        engagement = self._extract_engagement(message, words)
        momentum, momentum_value = self._extract_momentum(message)

        # Record for momentum tracking
        question_count = message.count("?")
        self._history.append({
            "ts": time.time(),
            "length": len(message),
            "questions": question_count,
        })

        return {
            "valence": round(valence, 4),
            "intensity": round(intensity, 4),
            "topic": topic,
            "topic_scores": {k: round(v, 4) for k, v in topic_scores.items()},
            "engagement": round(engagement, 4),
            "momentum": momentum,
            "momentum_value": round(momentum_value, 4),
            "user_id": user_id,
            "message_length": len(message),
            "ts": time.time(),
        }

    # ── Signal Extractors ─────────────────────────────────────────

    def _extract_valence(self, words: list[str], raw: str) -> float:
        """
        Emotional valence from keyword bags + punctuation density.

        Returns -1.0 (very negative) to +1.0 (very positive).
        """
        pos_count = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            base = 0.0  # Neutral
        else:
            base = (pos_count - neg_count) / total  # -1..+1

        # Punctuation modifiers
        excl = raw.count("!")
        quest = raw.count("?")

        # Exclamation marks amplify existing sentiment
        if excl > 0 and base != 0:
            base *= min(1.5, 1.0 + excl * 0.1)

        # ALL CAPS amplify (raw emotion indicator)
        caps_ratio = sum(1 for c in raw if c.isupper()) / max(1, len(raw))
        if caps_ratio > 0.5 and len(raw) > 5:
            base *= 1.3

        return max(-1.0, min(1.0, base))

    def _extract_intensity(self, words: list[str], raw: str) -> float:
        """
        Message energy/intensity from length, caps, emoji, punctuation.

        Returns 0.0 (calm) to 1.0 (very intense).
        """
        score = 0.0

        # Length component (longer = more invested, diminishing returns)
        length = len(raw)
        if length > 200:
            score += 0.3
        elif length > 100:
            score += 0.2
        elif length > 50:
            score += 0.1

        # Caps ratio
        caps_ratio = sum(1 for c in raw if c.isupper()) / max(1, len(raw))
        if caps_ratio > 0.3 and len(raw) > 5:
            score += 0.2

        # Exclamation/question marks (raw emphasis)
        punct = raw.count("!") + raw.count("?")
        score += min(0.2, punct * 0.05)

        # Emoji count
        emoji_count = len(_EMOJI_RE.findall(raw))
        score += min(0.15, emoji_count * 0.05)

        # Repeated punctuation (!!!, ???, ...)
        if re.search(r'[!?]{2,}', raw):
            score += 0.1

        return min(1.0, score)

    def _extract_topic(self, words: list[str]) -> tuple[str, dict]:
        """
        Dominant topic cluster from keyword matching.

        Returns (topic_name, {topic: score}).
        """
        scores = {}
        word_set = set(words)

        for topic, keywords in _TOPIC_KEYWORDS.items():
            matches = word_set & keywords
            scores[topic] = len(matches) / max(1, len(keywords)) * 5  # Normalize

        best_topic = max(scores, key=scores.get) if scores else "general"
        best_score = scores.get(best_topic, 0)

        # If best score is too low, it's general conversation
        if best_score < 0.1:
            best_topic = "general"

        return best_topic, scores

    def _extract_engagement(self, raw: str, words: list[str]) -> float:
        """
        Depth of interaction from questions, code blocks, links.

        Returns 0.0 (shallow) to 1.0 (deep engagement).
        """
        score = 0.0

        # Questions indicate curiosity
        questions = raw.count("?")
        score += min(0.3, questions * 0.1)

        # Code blocks indicate technical depth
        code_blocks = len(_CODE_RE.findall(raw))
        score += min(0.3, code_blocks * 0.15)

        # Links indicate research/sharing
        urls = len(_URL_RE.findall(raw))
        score += min(0.2, urls * 0.1)

        # Word count (more words = more thought put in)
        if len(words) > 30:
            score += 0.15
        elif len(words) > 15:
            score += 0.05

        return min(1.0, score)

    def _extract_momentum(self, raw: str) -> tuple[str, float]:
        """
        Conversation momentum from message length/frequency trends.

        Returns ("accelerating"|"decelerating"|"stable", float -1..+1).
        """
        if len(self._history) < 3:
            return "stable", 0.0

        recent = list(self._history)
        recent_lengths = [h["length"] for h in recent]

        # Compare last third vs first third
        third = max(1, len(recent_lengths) // 3)
        early = sum(recent_lengths[:third]) / third
        late = sum(recent_lengths[-third:]) / third

        if early == 0:
            return "stable", 0.0

        change_ratio = (late - early) / max(1, early)

        if change_ratio > 0.3:
            return "accelerating", min(1.0, change_ratio)
        elif change_ratio < -0.3:
            return "decelerating", max(-1.0, change_ratio)
        else:
            return "stable", round(change_ratio, 4)


    def check_rate(self, msg_type: str, source: str = "") -> dict | None:
        """
        Check InterfaceAdvisor for rate limit feedback.

        Returns RATE_LIMIT feedback dict if exceeded, None if within limits.
        """
        if not self._advisor:
            return None
        return self._advisor.check(msg_type, source)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer. Returns lowercase words."""
    # Strip code blocks and URLs first (don't count their words)
    cleaned = _CODE_RE.sub(" ", text)
    cleaned = _URL_RE.sub(" ", cleaned)
    # Split on non-alphanumeric, keep only words
    return [w.lower() for w in re.split(r'[^a-zA-Z0-9]+', cleaned) if w and len(w) > 1]

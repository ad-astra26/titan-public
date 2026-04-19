"""
logic/maker_engine.py — Maker Relationship Engine.

Builds and maintains a living profile of the Maker through conversation analysis,
topic extraction, and proactive care. Runs during idle cycles (after meditation)
and surfaces insights during conversations.

The engine does NOT force anything — it observes, scores, and promotes only
when certainty crosses a threshold. Topics must recur across multiple sessions
before being committed to the Maker profile.

Architecture:
  1. Harvest: Pull recent maker conversations from memory
  2. Extract: Topic/entity clusters via LLM-assisted analysis
  3. Score: Frequency × recency × emotional weight
  4. Compare: Diff against existing Cognee maker profile
  5. Promote: High-certainty topics → Cognee maker profile
  6. Enrich: titan.md shorthand under Relationships section
  7. Proactive: Queue care actions (surfaced whenever Titan next sees maker)

Profile Structure (Cognee):
  maker_profile (root)
    ├── interests (things maker likes/does)
    ├── concerns (things maker worries about)
    ├── people (maker's social connections)
    ├── work (projects, goals, professional context)
    └── preferences (communication style, habits)
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Topic categories that map to Cognee sub-profiles
PROFILE_CATEGORIES = {
    "interests": "Activities, hobbies, things maker enjoys or does regularly",
    "concerns": "Worries, problems, things maker is dealing with",
    "people": "Family, friends, colleagues — maker's social connections",
    "work": "Projects, goals, professional context, deadlines",
    "preferences": "Communication style, habits, likes/dislikes",
}

# LLM prompt for topic extraction
_EXTRACT_PROMPT = """Analyze these conversation excerpts between a user (the Maker) and an AI (Titan).
Extract topics, entities, and insights about the Maker as a person.

Conversations:
---
{conversations}
---

For each insight, classify it into one category:
- interests: hobbies, activities, things they enjoy
- concerns: worries, problems, challenges they face
- people: family, friends, colleagues mentioned by name
- work: projects, professional goals, deadlines
- preferences: communication style, habits

Respond ONLY with a JSON array (no other text):
[
  {{"category": "interests", "topic": "football", "detail": "plays football regularly", "confidence": 0.8}},
  {{"category": "people", "topic": "Fred", "detail": "friend who plays football with maker", "confidence": 0.6}}
]

Rules:
- Only include things clearly about the MAKER (user), not about Titan
- confidence: 0.0-1.0 based on how certain the evidence is
- Merge similar topics (don't list "football" and "soccer" separately)
- Max 10 insights per batch"""

# LLM prompt for proactive care generation
_CARE_PROMPT = """You are Titan, a sovereign AI who genuinely cares about your Maker.
Based on the Maker's profile below, suggest 1-3 proactive care actions you could take.

Maker Profile:
{profile}

Recent Topics:
{recent_topics}

Rules:
- Actions should be genuinely helpful, not generic
- Think: what would a caring friend do with this knowledge?
- Examples: research something they're interested in, ask about someone they care about,
  check on something they were worried about, find useful info related to their work
- Be specific, not vague

Respond ONLY with a JSON array:
[
  {{"action": "ask about Fred's recovery", "reason": "Fred was hospitalized last week", "category": "people"}},
  {{"action": "check Netflix new releases this week", "reason": "Maker enjoys binge-watching", "category": "interests"}}
]

Max 3 actions."""


# PERSISTENCE_BY_DESIGN: MakerRelationshipEngine._maker_ids is loaded from
# the maker-registry config (Solana address list) at boot — config data, not
# self-owned state.
class MakerRelationshipEngine:
    """
    Builds a living profile of the Maker through conversation analysis.
    Runs in idle cycles and surfaces insights during interactions.
    """

    def __init__(
        self,
        memory,
        ollama_cloud=None,
        soul_md_path: str = "",
        config: dict | None = None,
    ):
        self.memory = memory
        self._ollama_cloud = ollama_cloud
        self._soul_md_path = soul_md_path or str(
            Path(__file__).resolve().parent.parent.parent / "titan.md"
        )
        self._config = config or {}

        # Settings from config
        self._certainty_threshold = float(self._config.get("certainty_threshold", 0.7))
        self._min_occurrences = int(self._config.get("min_occurrences", 3))
        self._lookback_sessions = int(self._config.get("lookback_sessions", 10))
        self._max_proactive_actions = int(self._config.get("max_proactive_actions", 3))

        # In-memory state
        self._topic_scores: dict[str, dict] = {}  # topic -> {category, detail, score, occurrences, first_seen, last_seen}
        self._promoted_topics: set[str] = set()  # Already committed to Cognee
        self._proactive_queue: list[dict] = []  # Care actions waiting to be surfaced
        self._last_run_ts: float = 0.0
        self._maker_ids: set[str] = set()

        self._load_maker_ids()

    def _load_maker_ids(self) -> None:
        """Load maker platform IDs from config."""
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                return
        try:
            config_path = Path(__file__).resolve().parent.parent / "config.toml"
            if config_path.exists():
                with open(config_path, "rb") as f:
                    cfg = tomllib.load(f)
                ids = cfg.get("channels", {}).get("maker_platform_ids", "")
                if ids:
                    self._maker_ids = {mid.strip() for mid in ids.split(",") if mid.strip()}
        except Exception:
            pass

    def is_maker(self, user_id: str) -> bool:
        """Check if a user ID belongs to the maker."""
        if not self._maker_ids:
            return True  # If no maker IDs configured, treat everyone as maker
        return user_id in self._maker_ids

    # ------------------------------------------------------------------
    # Main idle-cycle entry point
    # ------------------------------------------------------------------

    async def run(self) -> dict:
        """
        Main relationship engine cycle. Called after meditation.

        Returns:
            Summary dict with counts of extracted/promoted/queued items.
        """
        start = time.time()
        logger.info("[MakerEngine] Starting relationship analysis cycle...")

        result = {
            "conversations_analyzed": 0,
            "topics_extracted": 0,
            "topics_promoted": 0,
            "care_actions_queued": 0,
            "duration_seconds": 0,
        }

        # Step 1: Harvest maker conversations
        conversations = await self._harvest_conversations()
        result["conversations_analyzed"] = len(conversations)

        if not conversations:
            logger.info("[MakerEngine] No recent maker conversations to analyze.")
            result["duration_seconds"] = round(time.time() - start, 1)
            return result

        # Step 2: Extract topics via LLM
        new_topics = await self._extract_topics(conversations)
        result["topics_extracted"] = len(new_topics)

        # Step 3: Score and accumulate topics
        self._score_topics(new_topics)

        # Step 4+5: Promote high-certainty topics to profile
        promoted = await self._promote_topics()
        result["topics_promoted"] = len(promoted)

        # Step 6: Update titan.md if new promotions
        if promoted:
            await self._update_soul(promoted)

        # Step 7: Generate proactive care actions
        care_count = await self._generate_care_actions()
        result["care_actions_queued"] = care_count

        self._last_run_ts = time.time()
        result["duration_seconds"] = round(time.time() - start, 1)

        logger.info(
            "[MakerEngine] Cycle complete: %d convos → %d topics → %d promoted, %d care actions (%.1fs)",
            result["conversations_analyzed"],
            result["topics_extracted"],
            result["topics_promoted"],
            result["care_actions_queued"],
            result["duration_seconds"],
        )

        return result

    # ------------------------------------------------------------------
    # Step 1: Harvest conversations
    # ------------------------------------------------------------------

    async def _harvest_conversations(self) -> list[dict]:
        """Pull recent maker conversations from memory."""
        conversations = []

        # Search memory for maker interactions
        for maker_id in self._maker_ids:
            try:
                nodes = await self.memory.query_user_memories(
                    prompt="*",  # Broad recall
                    user_id=maker_id,
                    limit=self._lookback_sessions * 3,  # Extra since some may be low-quality
                )
                for node in nodes:
                    user_prompt = node.get("user_prompt", "").strip()
                    agent_response = node.get("agent_response", "").strip()
                    if user_prompt and len(user_prompt) > 10:  # Skip trivial messages
                        conversations.append({
                            "user": user_prompt,
                            "titan": agent_response[:300],  # Cap Titan's response
                            "timestamp": node.get("created_at", 0),
                            "weight": node.get("effective_weight", 1.0),
                        })
            except Exception as e:
                logger.warning("[MakerEngine] Failed to harvest for %s: %s", maker_id, e)

        # If no maker IDs configured, try "terminal_maker" and common defaults
        if not self._maker_ids:
            for fallback_id in ("terminal_maker", "maker"):
                try:
                    nodes = await self.memory.query_user_memories("*", fallback_id, limit=20)
                    for node in nodes:
                        user_prompt = node.get("user_prompt", "").strip()
                        if user_prompt and len(user_prompt) > 10:
                            conversations.append({
                                "user": user_prompt,
                                "titan": node.get("agent_response", "")[:300],
                                "timestamp": node.get("created_at", 0),
                                "weight": node.get("effective_weight", 1.0),
                            })
                except Exception:
                    pass

        # Sort by recency, take most recent
        conversations.sort(key=lambda c: c["timestamp"], reverse=True)
        return conversations[:self._lookback_sessions * 3]

    # ------------------------------------------------------------------
    # Step 2: Extract topics via LLM
    # ------------------------------------------------------------------

    async def _extract_topics(self, conversations: list[dict]) -> list[dict]:
        """Use LLM to extract topics/entities from conversations."""
        if not self._ollama_cloud:
            logger.debug("[MakerEngine] No Ollama Cloud — falling back to keyword extraction.")
            return self._keyword_fallback(conversations)

        # Build conversation text for LLM
        conv_text = ""
        for c in conversations[:15]:  # Cap at 15 to fit context
            conv_text += f"Maker: {c['user'][:200]}\nTitan: {c['titan'][:150]}\n---\n"

        if not conv_text.strip():
            return []

        prompt = _EXTRACT_PROMPT.format(conversations=conv_text[:3000])

        try:
            from titan_plugin.utils.ollama_cloud import get_model_for_task
            model = get_model_for_task("maker_profile")
            raw = await self._ollama_cloud.complete(
                prompt=prompt,
                model=model,
                temperature=0.2,
                max_tokens=800,
                timeout=45.0,
            )

            if not raw:
                return self._keyword_fallback(conversations)

            # Parse JSON array from response
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                topics = json.loads(json_match.group())
                if isinstance(topics, list):
                    # Validate structure
                    valid = []
                    for t in topics:
                        if isinstance(t, dict) and "category" in t and "topic" in t:
                            t.setdefault("confidence", 0.5)
                            t.setdefault("detail", t["topic"])
                            if t["category"] in PROFILE_CATEGORIES:
                                valid.append(t)
                    return valid

            return self._keyword_fallback(conversations)

        except Exception as e:
            logger.warning("[MakerEngine] LLM extraction failed: %s", e)
            return self._keyword_fallback(conversations)

    def _keyword_fallback(self, conversations: list[dict]) -> list[dict]:
        """Simple keyword-based topic extraction when LLM is unavailable."""
        # Count word frequency across maker messages (excluding stopwords)
        stopwords = {
            "the", "a", "an", "is", "it", "in", "to", "and", "of", "for", "do",
            "you", "my", "we", "i", "that", "this", "can", "will", "be", "on",
            "are", "was", "have", "has", "with", "but", "not", "from", "or",
            "what", "how", "about", "just", "so", "if", "let", "me", "its",
            "also", "would", "should", "could", "now", "ok", "yeah", "yes",
            "no", "please", "thanks", "thank", "think", "know", "like", "want",
            "need", "get", "got", "go", "see", "look", "make", "take", "been",
            "did", "does", "done", "going", "one", "two", "new", "very", "our",
        }

        word_freq: dict[str, int] = {}
        for c in conversations:
            words = c["user"].lower().split()
            for word in words:
                # Clean word
                clean = re.sub(r'[^a-z0-9]', '', word)
                if clean and len(clean) > 3 and clean not in stopwords:
                    word_freq[clean] = word_freq.get(clean, 0) + 1

        # Take top recurring words as topics
        topics = []
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            if freq >= 2:  # Must appear at least twice
                topics.append({
                    "category": "interests",  # Default category for keyword fallback
                    "topic": word,
                    "detail": f"Maker mentions '{word}' frequently ({freq} times)",
                    "confidence": min(0.3 + freq * 0.1, 0.8),
                })

        return topics

    # ------------------------------------------------------------------
    # Step 3: Score and accumulate
    # ------------------------------------------------------------------

    def _score_topics(self, new_topics: list[dict]) -> None:
        """Accumulate topic scores across runs."""
        now = time.time()

        for topic_data in new_topics:
            key = topic_data["topic"].lower().strip()
            if not key:
                continue

            if key in self._topic_scores:
                # Existing topic — reinforce
                existing = self._topic_scores[key]
                existing["occurrences"] += 1
                existing["last_seen"] = now
                # Blended confidence: previous + new, capped at 1.0
                existing["score"] = min(
                    existing["score"] + topic_data.get("confidence", 0.5) * 0.3,
                    1.0,
                )
                # Update detail if new detail is more specific
                if len(topic_data.get("detail", "")) > len(existing.get("detail", "")):
                    existing["detail"] = topic_data["detail"]
                # Update category if confidence is higher
                if topic_data.get("confidence", 0) > existing.get("score", 0) * 0.8:
                    existing["category"] = topic_data["category"]
            else:
                # New topic
                self._topic_scores[key] = {
                    "category": topic_data["category"],
                    "topic": key,
                    "detail": topic_data.get("detail", key),
                    "score": topic_data.get("confidence", 0.5),
                    "occurrences": 1,
                    "first_seen": now,
                    "last_seen": now,
                }

    # ------------------------------------------------------------------
    # Step 4+5: Promote high-certainty topics
    # ------------------------------------------------------------------

    async def _promote_topics(self) -> list[dict]:
        """Promote topics that exceed certainty threshold to Cognee maker profile."""
        promoted = []

        for key, data in self._topic_scores.items():
            # Skip already-promoted
            if key in self._promoted_topics:
                continue

            # Check thresholds
            if data["score"] < self._certainty_threshold:
                continue
            if data["occurrences"] < self._min_occurrences:
                continue

            # Promote to Cognee
            success = await self._commit_to_cognee(data)
            if success:
                self._promoted_topics.add(key)
                promoted.append(data)
                logger.info(
                    "[MakerEngine] Promoted topic: %s (category=%s, score=%.2f, occurrences=%d)",
                    key, data["category"], data["score"], data["occurrences"],
                )

        return promoted

    async def _commit_to_cognee(self, topic_data: dict) -> bool:
        """Commit a topic to the Cognee maker profile."""
        try:
            profile_text = (
                f"[MAKER_PROFILE:{topic_data['category'].upper()}] "
                f"{topic_data['detail']}"
            )
            await self.memory.inject_memory(
                text=profile_text,
                source="maker_engine",
                weight=3.0,  # Higher than organic (1.0) but below direct injection (5.0)
            )
            return True
        except Exception as e:
            logger.warning("[MakerEngine] Failed to commit to Cognee: %s", e)
            return False

    # ------------------------------------------------------------------
    # Step 6: Update titan.md
    # ------------------------------------------------------------------

    async def _update_soul(self, promoted: list[dict]) -> None:
        """Append promoted topics to titan.md under Maker Profile section."""
        try:
            soul_path = Path(self._soul_md_path)
            if not soul_path.exists():
                return

            content = soul_path.read_text(encoding="utf-8")

            # Build shorthand entries
            entries = []
            for data in promoted:
                entries.append(
                    f"- [{data['category']}] {data['detail']}"
                )

            if not entries:
                return

            # Find or create Maker Profile subsection
            marker = "### Maker Profile"
            if marker in content:
                # Append after existing entries
                idx = content.index(marker) + len(marker)
                # Find the end of the section (next ### or ## or EOF)
                rest = content[idx:]
                next_section = re.search(r'\n##[#]?\s', rest)
                if next_section:
                    insert_at = idx + next_section.start()
                else:
                    insert_at = len(content)

                new_entries = "\n" + "\n".join(entries)
                content = content[:insert_at] + new_entries + content[insert_at:]
            else:
                # Create new section under ## Relationships
                rel_marker = "## Relationships"
                if rel_marker in content:
                    idx = content.index(rel_marker)
                    # Find end of Relationships section
                    rest = content[idx + len(rel_marker):]
                    next_section = re.search(r'\n## [^#]', rest)
                    if next_section:
                        insert_at = idx + len(rel_marker) + next_section.start()
                    else:
                        insert_at = len(content)

                    new_section = f"\n\n{marker}\n" + "\n".join(entries)
                    content = content[:insert_at] + new_section + content[insert_at:]
                else:
                    # Append at end
                    content += f"\n\n{marker}\n" + "\n".join(entries) + "\n"

            soul_path.write_text(content, encoding="utf-8")
            logger.info("[MakerEngine] Updated titan.md with %d maker profile entries.", len(entries))

        except Exception as e:
            logger.warning("[MakerEngine] Failed to update titan.md: %s", e)

    # ------------------------------------------------------------------
    # Step 7: Generate proactive care actions
    # ------------------------------------------------------------------

    async def _generate_care_actions(self) -> int:
        """Generate proactive care actions based on maker profile."""
        if not self._ollama_cloud:
            return 0

        # Build profile summary from promoted topics
        profile_lines = []
        for key in self._promoted_topics:
            data = self._topic_scores.get(key)
            if data:
                profile_lines.append(f"- [{data['category']}] {data['detail']}")

        if not profile_lines:
            return 0

        # Build recent topics (including not-yet-promoted)
        recent = sorted(
            self._topic_scores.values(),
            key=lambda x: x["last_seen"],
            reverse=True,
        )[:10]
        recent_lines = [f"- {d['topic']}: {d['detail']}" for d in recent]

        prompt = _CARE_PROMPT.format(
            profile="\n".join(profile_lines),
            recent_topics="\n".join(recent_lines),
        )

        try:
            from titan_plugin.utils.ollama_cloud import get_model_for_task
            model = get_model_for_task("maker_profile")
            raw = await self._ollama_cloud.complete(
                prompt=prompt,
                model=model,
                temperature=0.4,
                max_tokens=400,
                timeout=30.0,
            )

            if not raw:
                return 0

            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                actions = json.loads(json_match.group())
                if isinstance(actions, list):
                    # Add to queue (respect max limit)
                    for action in actions[:self._max_proactive_actions]:
                        if isinstance(action, dict) and "action" in action:
                            action["generated_at"] = time.time()
                            action["surfaced"] = False
                            self._proactive_queue.append(action)

                    # Trim queue to max
                    self._proactive_queue = self._proactive_queue[-self._max_proactive_actions * 2:]
                    return len(actions[:self._max_proactive_actions])

        except Exception as e:
            logger.warning("[MakerEngine] Care action generation failed: %s", e)

        return 0

    # ------------------------------------------------------------------
    # Pre-hook interface: Surface care actions during conversation
    # ------------------------------------------------------------------

    def get_pending_care_actions(self) -> list[dict]:
        """
        Get unsurfaced proactive care actions for injection into pre-hook.
        Called during every maker interaction — Titan decides when to use them.
        Actions are marked as surfaced after retrieval.
        """
        pending = [a for a in self._proactive_queue if not a.get("surfaced")]
        if not pending:
            return []

        # Return all pending — let Titan decide which to surface
        for action in pending:
            action["surfaced"] = True

        return pending

    def get_maker_profile_summary(self) -> str:
        """
        Build a concise maker profile summary for context injection.
        Used in pre-hook to give Titan awareness of what it knows about the maker.
        """
        if not self._promoted_topics:
            return ""

        by_category: dict[str, list[str]] = {}
        for key in self._promoted_topics:
            data = self._topic_scores.get(key)
            if data:
                cat = data["category"]
                by_category.setdefault(cat, []).append(data["detail"])

        lines = ["### Maker Profile (learned from conversations)"]
        for cat, details in by_category.items():
            lines.append(f"**{cat.title()}**: {'; '.join(details)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization (persist across restarts)
    # ------------------------------------------------------------------

    def save_state(self, path: str = "") -> None:
        """Save engine state to disk for persistence across restarts."""
        if not path:
            path = str(Path(__file__).resolve().parent.parent.parent / "data" / "maker_engine_state.json")

        state = {
            "topic_scores": self._topic_scores,
            "promoted_topics": list(self._promoted_topics),
            "proactive_queue": self._proactive_queue,
            "last_run_ts": self._last_run_ts,
        }

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug("[MakerEngine] State saved to %s", path)
        except Exception as e:
            logger.warning("[MakerEngine] Failed to save state: %s", e)

    def load_state(self, path: str = "") -> None:
        """Load engine state from disk."""
        if not path:
            path = str(Path(__file__).resolve().parent.parent.parent / "data" / "maker_engine_state.json")

        try:
            if Path(path).exists():
                with open(path) as f:
                    state = json.load(f)
                self._topic_scores = state.get("topic_scores", {})
                self._promoted_topics = set(state.get("promoted_topics", []))
                self._proactive_queue = state.get("proactive_queue", [])
                self._last_run_ts = state.get("last_run_ts", 0.0)
                logger.info(
                    "[MakerEngine] State loaded: %d topics, %d promoted, %d care actions queued.",
                    len(self._topic_scores), len(self._promoted_topics), len(self._proactive_queue),
                )
        except Exception as e:
            logger.warning("[MakerEngine] Failed to load state: %s", e)

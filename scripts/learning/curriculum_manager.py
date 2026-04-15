"""
scripts/learning/curriculum_manager.py — Curriculum Manager for Learning TestSuite.

Manages learning progression based on developmental age (pi-clusters),
builds diverse daily queues mixing module types, and handles checkpoint/resume
for crash resilience.

Checkpoint is saved after every module completion via atomic write (tmp → rename).
On restart, TestSuite resumes from exactly where it left off.
"""
import json
import logging
import os
import random
import tempfile
import time

logger = logging.getLogger("testsuite.curriculum")

CHECKPOINT_PATH = os.getenv("TESTSUITE_CHECKPOINT", "data/testsuite_checkpoint.json")

# Developmental phase → module mix (weights, must sum to ~1.0)
PHASE_MODULE_MIX = {
    "BIRTH": {       # 0-50 vocab
        "language": 0.65,
        "music": 0.15,
        "art_narration": 0.10,
        "conversation": 0.05,
        "arc_play": 0.05,      # Early spatial awareness exercises
    },
    "EARLY": {       # 50-150 vocab — introduce composition
        "language": 0.35,
        "composition": 0.20,
        "conversation": 0.15,
        "music": 0.10,
        "art_narration": 0.05,
        "arc_play": 0.10,      # Growing puzzle-solving skills
        "assessment": 0.05,
    },
    "GROWING": {     # 150-300 vocab — heavy composition
        "language": 0.20,
        "composition": 0.30,
        "conversation": 0.15,
        "arc_play": 0.10,      # Regular cognitive challenges
        "puzzle": 0.05,
        "music": 0.10,
        "art_narration": 0.05,
        "assessment": 0.05,
    },
    "COMPOSITION": { # 300-500 vocab — grammar mastery
        "language": 0.15,
        "composition": 0.30,
        "conversation": 0.15,
        "arc_play": 0.10,      # Consistent reasoning practice
        "puzzle": 0.10,
        "assessment": 0.10,
        "music": 0.05,
        "art_narration": 0.05,
    },
    "EXPRESSION": {  # 500+ vocab — autonomous speech
        "language": 0.10,
        "composition": 0.25,
        "conversation": 0.30,
        "puzzle": 0.15,
        "assessment": 0.15,
        "music": 0.05,
    },
}

# Words per language module by phase
WORDS_PER_MODULE = {
    "BIRTH": 3,
    "EARLY": 4,
    "GROWING": 5,
    "COMPOSITION": 3,  # Fewer words, more grammar practice
    "EXPRESSION": 2,   # Mostly reinforcement + composition
}


class CurriculumManager:
    """Manages learning progression, queue building, and checkpoint/resume."""

    def __init__(self, checkpoint_path: str = None):
        self._path = checkpoint_path or CHECKPOINT_PATH
        self._checkpoint = self._load_checkpoint()

    @property
    def day(self) -> int:
        return self._checkpoint.get("curriculum_day", 1)

    @property
    def position(self) -> int:
        return self._checkpoint.get("queue_position", 0)

    @property
    def phase(self) -> str:
        return self._checkpoint.get("phase", "BIRTH")

    def _load_checkpoint(self) -> dict:
        """Load existing checkpoint or create fresh."""
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    cp = json.load(f)
                logger.info("[Curriculum] Loaded checkpoint: day=%d pos=%d phase=%s vocab=%d",
                            cp.get("curriculum_day", 1),
                            cp.get("queue_position", 0),
                            cp.get("phase", "BIRTH"),
                            cp.get("vocabulary_now", 0))
                return cp
            except Exception as e:
                logger.warning("[Curriculum] Checkpoint corrupt, starting fresh: %s", e)

        return self._fresh_checkpoint()

    def _fresh_checkpoint(self) -> dict:
        return {
            "curriculum_day": 1,
            "queue": [],
            "queue_position": 0,
            "phase": "BIRTH",
            "vocabulary_at_start": 0,
            "vocabulary_now": 0,
            "words_taught_today": [],
            "modules_completed": 0,
            "modules_completed_today": 0,
            "grammar_rules_learned": 0,
            "composition_confidence": 0.0,
            "last_module_type": None,
            "last_module_result": None,
            "started_ts": time.time(),
            "last_update_ts": time.time(),
        }

    def save_checkpoint(self, module_result: dict = None) -> None:
        """Atomic checkpoint save (write tmp → rename)."""
        self._checkpoint["last_update_ts"] = time.time()
        if module_result:
            self._checkpoint["last_module_type"] = module_result.get("type", "?")
            self._checkpoint["last_module_result"] = {
                "success": module_result.get("success", False),
                "words_taught": module_result.get("words_taught", 0),
                "accuracy": module_result.get("accuracy", 0),
                "duration": module_result.get("duration", 0),
            }
            self._checkpoint["modules_completed"] += 1
            self._checkpoint["modules_completed_today"] += 1

            # Track words taught today
            for w in module_result.get("words_list", []):
                if w not in self._checkpoint["words_taught_today"]:
                    self._checkpoint["words_taught_today"].append(w)

        # Atomic write
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self._path) or ".")
            with os.fdopen(fd, "w") as f:
                json.dump(self._checkpoint, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            logger.error("[Curriculum] Checkpoint save failed: %s", e)

    def update_vocab_count(self, vocab_size: int) -> None:
        """Update vocabulary tracking from live state."""
        if self._checkpoint["vocabulary_at_start"] == 0:
            self._checkpoint["vocabulary_at_start"] = vocab_size
        self._checkpoint["vocabulary_now"] = vocab_size

    def get_phase(self, vocab_size: int) -> str:
        """Determine developmental phase from vocabulary size."""
        if vocab_size < 50:
            return "BIRTH"
        elif vocab_size < 150:
            return "EARLY"
        elif vocab_size < 300:
            return "GROWING"
        elif vocab_size < 500:
            return "COMPOSITION"
        else:
            return "EXPRESSION"

    def get_words_per_module(self) -> int:
        """How many words to teach per language module."""
        return WORDS_PER_MODULE.get(self.phase, 3)

    def build_day_queue(self, vocab_size: int, dev_age: int = 0) -> list:
        """Build a diverse 'kindergarten day' queue of modules.

        Mixes module types based on developmental phase.
        Inserts rest periods between modules.
        Queue is 15-25 modules for a full day (~8-10 hours).
        """
        phase = self.get_phase(vocab_size)
        self._checkpoint["phase"] = phase
        mix = PHASE_MODULE_MIX.get(phase, PHASE_MODULE_MIX["BIRTH"])

        # Build weighted module list
        modules = []
        target_count = 20  # ~20 modules per day

        for module_type, weight in mix.items():
            count = max(1, round(target_count * weight))
            modules.extend([module_type] * count)

        # Shuffle to avoid monotony but keep some structure
        random.shuffle(modules)

        # Insert rest after every 2-3 modules
        queue = []
        for i, mod in enumerate(modules):
            queue.append({"type": mod, "index": len(queue)})
            if (i + 1) % 2 == 0:  # Rest after every 2 modules
                queue.append({"type": "rest", "index": len(queue)})

        self._checkpoint["queue"] = queue
        self._checkpoint["queue_position"] = 0

        logger.info("[Curriculum] Built day %d queue: %d items (phase=%s, vocab=%d)",
                    self.day, len(queue), phase, vocab_size)

        return queue

    def get_next_module(self, state: dict) -> dict:
        """Get next module from queue. Build new day if queue exhausted."""
        queue = self._checkpoint.get("queue", [])
        pos = self._checkpoint.get("queue_position", 0)

        # Build new day queue if needed
        if not queue or pos >= len(queue):
            vocab_size = state.get("vocab_size", 0)
            dev_age = state.get("dev_age", 0)
            queue = self.build_day_queue(vocab_size, dev_age)
            self._checkpoint["curriculum_day"] += 1
            self._checkpoint["modules_completed_today"] = 0
            self._checkpoint["words_taught_today"] = []
            self.save_checkpoint()

        if pos < len(queue):
            module_spec = queue[pos]
            return module_spec

        return {"type": "language", "index": 0}  # Fallback

    def advance_position(self) -> None:
        """Advance queue position after module completion."""
        self._checkpoint["queue_position"] = self._checkpoint.get("queue_position", 0) + 1

    def get_stats(self) -> dict:
        return {
            "day": self.day,
            "phase": self.phase,
            "queue_length": len(self._checkpoint.get("queue", [])),
            "queue_position": self.position,
            "modules_completed": self._checkpoint.get("modules_completed", 0),
            "modules_today": self._checkpoint.get("modules_completed_today", 0),
            "words_today": len(self._checkpoint.get("words_taught_today", [])),
            "vocab_start": self._checkpoint.get("vocabulary_at_start", 0),
            "vocab_now": self._checkpoint.get("vocabulary_now", 0),
        }

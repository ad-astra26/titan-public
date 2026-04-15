"""
titan_plugin/logic/experience_playground.py — Experience Playground Framework.

A modular framework for plugging in learning experiences that Titan processes
through his 130D Trinity bodies. Each experience follows the universal pattern:

    Stimulus → Feel → Record → Learn → Feedback

Experiences are plugins (ExperiencePlugin subclasses) that define:
  - How to generate stimuli (words, music, puzzles, conversations)
  - How stimuli perturb the Trinity tensors (130D felt-resonance)
  - How to evaluate Titan's response (LLM teacher feedback)

Existing experiences (music, persona, ARC) can be migrated as plugins.
New experiences (language, sacred geometry) plug in naturally.

The Playground runner (ExperiencePlayground) orchestrates sessions:
  1. Generate stimulus from plugin
  2. Apply 130D perturbation to Trinity via bus
  3. Wait for hormonal response + nervous system reaction
  4. Collect Titan's expression/action
  5. Teacher evaluates response
  6. Record full experience chain in inner memory
  7. Repeat with progressive difficulty
"""
import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Experience Plugin Base Class ──────────────────────────────────────

class ExperiencePlugin:
    """Base class for all learning experiences.

    Subclasses implement:
      - generate_stimulus(): what to present to Titan
      - compute_perturbation(): how it maps to 130D Trinity felt-tensor
      - evaluate_response(): how LLM teacher scores Titan's response
    """

    name: str = "base"
    description: str = ""
    difficulty_levels: int = 6

    def __init__(self, llm_fn: Optional[Callable] = None,
                 inner_memory=None, hormonal_system=None):
        self._llm_fn = llm_fn
        self._memory = inner_memory
        self._hormonal = hormonal_system
        self._current_level: int = 1
        self._total_stimuli: int = 0
        self._correct_responses: int = 0
        self._session_history: list = []

    async def generate_stimulus(self) -> dict:
        """Generate next stimulus for this experience type.

        Returns:
            {
                "content": str,         # The stimulus content (word, image, puzzle, etc.)
                "type": str,            # Stimulus type (word, music, pattern, conversation)
                "level": int,           # Current difficulty level
                "expected": dict|None,  # Expected response (for evaluation)
                "metadata": dict,       # Plugin-specific metadata
            }
        """
        raise NotImplementedError

    def compute_perturbation(self, stimulus: dict) -> dict:
        """Map stimulus to 130D Trinity tensor perturbation.

        Returns:
            {
                "inner_body": [5 floats],      # Physical sensation
                "inner_mind": [15 floats],     # Cognitive/feeling/willing response
                "inner_spirit": [45 floats],   # Deep consciousness resonance
                "outer_body": [5 floats],      # Physical world engagement
                "outer_mind": [15 floats],     # Practical response
                "outer_spirit": [45 floats],   # Material observer response
                "hormone_stimuli": dict,       # {program: activation_delta}
            }
        """
        raise NotImplementedError

    async def evaluate_response(self, stimulus: dict,
                                response: dict) -> dict:
        """LLM teacher evaluates Titan's response.

        Returns:
            {
                "score": float,        # 0.0 - 1.0
                "feedback": str,       # Teacher's feedback
                "correction": dict|None,  # Correct answer if wrong
                "reinforcement": dict|None,  # What to strengthen if right
            }
        """
        raise NotImplementedError

    def should_advance_level(self) -> bool:
        """Check if Titan is ready for next difficulty level."""
        if self._total_stimuli < 10:
            return False
        # Look at recent accuracy (last 20 stimuli)
        recent = self._session_history[-20:]
        if len(recent) < 10:
            return False
        recent_correct = sum(1 for r in recent if r.get("score", 0) > 0.7)
        return recent_correct / len(recent) > 0.8

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "level": self._current_level,
            "max_level": self.difficulty_levels,
            "total_stimuli": self._total_stimuli,
            "correct": self._correct_responses,
            "accuracy": round(self._correct_responses /
                              max(1, self._total_stimuli), 3),
            "history_size": len(self._session_history),
        }


# ── Experience Playground Runner ──────────────────────────────────────

class ExperiencePlayground:
    """Runs learning experiences through Titan's 130D Trinity bodies.

    Orchestrates the universal experience pattern:
    Stimulus → Feel → Record → Learn → Feedback

    Usage:
        playground = ExperiencePlayground(bus, inner_memory, hormonal_system, llm_fn)
        playground.register(LanguageLearningExperience(...))
        result = await playground.run_session("language", num_stimuli=10)
    """

    def __init__(self, bus=None, inner_memory=None,
                 hormonal_system=None, llm_fn: Optional[Callable] = None):
        self._bus = bus
        self._memory = inner_memory
        self._hormonal = hormonal_system
        self._llm_fn = llm_fn
        self._plugins: dict[str, ExperiencePlugin] = {}
        self._active_experience: Optional[str] = None
        self._active_session: Optional[dict] = None
        self._session_log: list = []
        self._total_sessions: int = 0

    def register(self, plugin: ExperiencePlugin) -> None:
        """Register an experience plugin."""
        self._plugins[plugin.name] = plugin
        logger.info("[Playground] Registered experience: %s (%s)",
                    plugin.name, plugin.description)

    def list_experiences(self) -> list[dict]:
        """List all registered experiences with stats."""
        return [p.get_stats() for p in self._plugins.values()]

    @property
    def is_active(self) -> bool:
        return self._active_experience is not None

    async def run_session(self, experience_name: str,
                          num_stimuli: int = 10,
                          pause_between: float = 5.0) -> dict:
        """
        Run a learning session with the given experience.

        Args:
            experience_name: Registered plugin name
            num_stimuli: How many stimuli to present
            pause_between: Seconds between stimuli (for hormonal response)

        Returns:
            Session result with accuracy, level, and per-stimulus details
        """
        plugin = self._plugins.get(experience_name)
        if not plugin:
            return {"error": f"Unknown experience: {experience_name}",
                    "available": list(self._plugins.keys())}

        if self._active_experience:
            return {"error": f"Session already active: {self._active_experience}"}

        self._active_experience = experience_name
        self._total_sessions += 1
        session_id = f"{experience_name}_{self._total_sessions}"
        self._active_session = {
            "id": session_id,
            "experience": experience_name,
            "started": time.time(),
            "stimuli_target": num_stimuli,
            "stimuli_completed": 0,
        }

        logger.info("[Playground] Session %s started: %s (level=%d, stimuli=%d)",
                    session_id, experience_name, plugin._current_level, num_stimuli)

        results = []
        initial_level = plugin._current_level

        for i in range(num_stimuli):
            try:
                result = await self._run_single_stimulus(plugin, i + 1)
                results.append(result)
                self._active_session["stimuli_completed"] = i + 1

                # Check level advancement
                if plugin.should_advance_level():
                    old_level = plugin._current_level
                    plugin._current_level = min(plugin._current_level + 1,
                                                plugin.difficulty_levels)
                    if plugin._current_level > old_level:
                        logger.info("[Playground] %s LEVEL UP: %d → %d",
                                    experience_name, old_level, plugin._current_level)

                # Pause between stimuli for hormonal processing
                if i < num_stimuli - 1:
                    await asyncio.sleep(pause_between)

            except Exception as e:
                logger.error("[Playground] Stimulus %d failed: %s", i + 1, e)
                results.append({"error": str(e), "sequence": i + 1})

        self._active_experience = None
        self._active_session = None

        session_result = {
            "session_id": session_id,
            "experience": experience_name,
            "stimuli_count": len(results),
            "accuracy": plugin._correct_responses / max(1, plugin._total_stimuli),
            "level_start": initial_level,
            "level_end": plugin._current_level,
            "level_advanced": plugin._current_level > initial_level,
            "results": results,
            "duration": time.time() - results[0].get("timestamp", time.time()) if results else 0,
        }

        self._session_log.append({
            "session_id": session_id,
            "experience": experience_name,
            "stimuli": len(results),
            "accuracy": session_result["accuracy"],
            "level": plugin._current_level,
            "timestamp": time.time(),
        })

        logger.info("[Playground] Session %s complete: accuracy=%.1f%% level=%d",
                    session_id, session_result["accuracy"] * 100,
                    plugin._current_level)

        return session_result

    async def _run_single_stimulus(self, plugin: ExperiencePlugin,
                                   sequence: int) -> dict:
        """Run one stimulus through the full experience pipeline."""
        t0 = time.time()

        # 1. Generate stimulus
        stimulus = await plugin.generate_stimulus()

        # 2. Compute 130D perturbation
        perturbation = plugin.compute_perturbation(stimulus)

        # 3. Publish perturbation to bus as EXPERIENCE_STIMULUS
        if self._bus:
            try:
                from titan_plugin.bus import make_msg
                self._bus.publish(make_msg(
                    "EXPERIENCE_STIMULUS", "playground", "all", {
                        "experience": plugin.name,
                        "stimulus": stimulus,
                        "perturbation": perturbation,
                        "sequence": sequence,
                        "level": plugin._current_level,
                    }))
            except Exception as e:
                logger.debug("[Playground] Bus publish failed: %s", e)

        # 4. Apply hormone stimuli directly if available
        if self._hormonal and perturbation.get("hormone_stimuli"):
            for program, delta in perturbation["hormone_stimuli"].items():
                try:
                    hormone = self._hormonal.get(program)
                    if hormone:
                        hormone.accumulate(delta, dt=1.0)
                except Exception:
                    pass

        # 5. Brief pause for Trinity to process
        await asyncio.sleep(1.0)

        # 6. Collect Titan's response (from recent actions/state)
        response = self._collect_response(plugin.name)

        # 7. Teacher evaluates
        evaluation = await plugin.evaluate_response(stimulus, response)
        score = evaluation.get("score", 0.5)

        # 8. Record in inner memory
        if self._memory:
            try:
                self._memory.record_event(
                    f"experience_{plugin.name}",
                    program="PLAYGROUND",
                    details=json.dumps({
                        "stimulus": stimulus.get("content", ""),
                        "type": stimulus.get("type", ""),
                        "response": response,
                        "score": score,
                        "level": plugin._current_level,
                        "sequence": sequence,
                    }),
                )
            except Exception as e:
                logger.debug("[Playground] Memory record failed: %s", e)

        # 9. Update plugin stats
        plugin._total_stimuli += 1
        if score > 0.7:
            plugin._correct_responses += 1
        plugin._session_history.append({
            "stimulus": stimulus.get("content", ""),
            "score": score,
            "level": plugin._current_level,
            "timestamp": time.time(),
        })

        return {
            "sequence": sequence,
            "stimulus": stimulus,
            "response": response,
            "score": score,
            "feedback": evaluation.get("feedback", ""),
            "timestamp": t0,
            "duration": time.time() - t0,
        }

    def _collect_response(self, experience_name: str) -> dict:
        """Collect Titan's response to the latest stimulus.

        Reads from recent inner memory, hormonal state, and nervous system
        to understand how Titan reacted to the perturbation.
        """
        response = {
            "hormonal_state": {},
            "fired_programs": [],
            "expression": None,
        }

        # Read current hormonal state
        if self._hormonal:
            try:
                for name, hormone in self._hormonal.items():
                    response["hormonal_state"][name] = {
                        "level": round(hormone.level, 3),
                        "fired": hormone.total_fires,
                    }
            except Exception:
                pass

        # Read recent program fires from inner memory
        if self._memory:
            try:
                recent = self._memory.get_recent_fires(limit=3)
                response["fired_programs"] = [
                    {"program": r[0], "intensity": r[1]}
                    for r in recent
                ] if recent else []
            except Exception:
                pass

        return response

    def get_stats(self) -> dict:
        return {
            "registered_experiences": list(self._plugins.keys()),
            "active_experience": self._active_experience,
            "active_session": self._active_session,
            "total_sessions": self._total_sessions,
            "session_log": self._session_log[-10:],  # Last 10 sessions
            "plugins": {name: p.get_stats() for name, p in self._plugins.items()},
        }

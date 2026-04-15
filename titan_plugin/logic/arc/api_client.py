"""
titan_plugin/logic/arc/api_client.py — ARC-AGI-3 SDK Bridge.

Thin wrapper around the official arc_agi SDK (v0.9.6).
Converts SDK FrameDataRaw into our Frame dataclass for backward compatibility.
Derives reward signal from levels_completed delta.

SDK docs: https://docs.arcprize.org/
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Normalized frame from ARC-AGI-3 SDK."""
    grid: np.ndarray                          # (64, 64) int8, values 0-12
    available_actions: list[int] = field(default_factory=list)  # e.g. [1, 2, 3, 4]
    state: str = "NOT_FINISHED"               # NOT_FINISHED / WIN / GAME_OVER
    levels_completed: int = 0
    win_levels: int = 0
    step_count: int = 0
    reward: float = 0.0                       # derived from levels_completed delta
    done: bool = False
    game_id: str = ""


class ArcSDKBridge:
    """
    Bridge between the official arc_agi SDK and Titan's ARC session.

    Manages Arcade instance, environment lifecycle, reward derivation,
    and scorecard tracking. All methods are synchronous (SDK is blocking).
    """

    def __init__(self, api_key: str = "", online: bool = False):
        self._arcade = None
        self._envs: dict[str, object] = {}
        self._step_counts: dict[str, int] = {}
        self._prev_levels: dict[str, int] = {}
        self._initialized = False
        self._api_key = api_key
        self._online = online
        self._scorecard_id: Optional[str] = None

    def initialize(self) -> bool:
        """Create Arcade instance. If api_key + online, uses ONLINE mode for leaderboard."""
        try:
            import arc_agi
            kwargs = {}
            if self._api_key:
                kwargs["arc_api_key"] = self._api_key
            if self._online:
                kwargs["operation_mode"] = arc_agi.OperationMode.ONLINE
            self._arcade = arc_agi.Arcade(**kwargs)
            self._initialized = True
            envs = self._arcade.get_environments()
            mode = "ONLINE" if self._online else ("KEYED" if self._api_key else "ANONYMOUS")
            logger.info("[ArcSDK] Initialized (%s) — %d environments available", mode, len(envs))
            return True
        except Exception as e:
            logger.error("[ArcSDK] Failed to initialize: %s", e)
            return False

    def create_scorecard(self, tags: list[str] = None,
                         source_url: str = None) -> Optional[str]:
        """Create a scorecard for leaderboard tracking. Returns scorecard ID."""
        if not self._arcade:
            return None
        try:
            kwargs = {}
            if tags:
                kwargs["tags"] = tags
            if source_url:
                kwargs["source_url"] = source_url
            self._scorecard_id = self._arcade.create_scorecard(**kwargs)
            logger.info("[ArcSDK] Scorecard created: %s (tags=%s)", self._scorecard_id, tags)
            return self._scorecard_id
        except Exception as e:
            logger.error("[ArcSDK] create_scorecard failed: %s", e)
            return None

    def close_scorecard(self) -> Optional[dict]:
        """Close scorecard and return final results."""
        if not self._arcade or not self._scorecard_id:
            return None
        try:
            result = self._arcade.close_scorecard(scorecard_id=self._scorecard_id)
            logger.info("[ArcSDK] Scorecard closed: %s", self._scorecard_id)
            if result and hasattr(result, "model_dump"):
                return result.model_dump()
            return {"closed": True, "id": self._scorecard_id}
        except Exception as e:
            logger.error("[ArcSDK] close_scorecard failed: %s", e)
            return None

    def get_environments(self) -> list[dict]:
        """List available environments with metadata."""
        if not self._arcade:
            return []
        envs = self._arcade.get_environments()
        result = []
        for e in envs:
            d = e.model_dump() if hasattr(e, "model_dump") else {}
            result.append({
                "game_id": d.get("game_id", ""),
                "title": d.get("title", ""),
                "baseline_actions": d.get("baseline_actions", []),
            })
        return result

    def make_env(self, game_id: str) -> bool:
        """Create environment for a game. Returns True on success."""
        if not self._arcade:
            logger.error("[ArcSDK] Not initialized")
            return False
        try:
            # SDK expects full game_id with hash suffix — find matching env
            target_id = self._resolve_game_id(game_id)
            if not target_id:
                logger.error("[ArcSDK] Game '%s' not found", game_id)
                return False
            make_kwargs = {"render_mode": None, "include_frame_data": True}
            if self._scorecard_id:
                make_kwargs["scorecard_id"] = self._scorecard_id
            env = self._arcade.make(target_id, **make_kwargs)
            if env is None:
                logger.error("[ArcSDK] make() returned None for '%s'", target_id)
                return False
            self._envs[game_id] = env
            self._step_counts[game_id] = 0
            self._prev_levels[game_id] = 0
            logger.info("[ArcSDK] Environment '%s' created (resolved: %s)", game_id, target_id)
            return True
        except Exception as e:
            logger.error("[ArcSDK] make_env(%s) failed: %s", game_id, e)
            return False

    def reset(self, game_id: str) -> Optional[Frame]:
        """Reset environment, return initial frame."""
        env = self._envs.get(game_id)
        if env is None:
            logger.error("[ArcSDK] No env for '%s' — call make_env() first", game_id)
            return None
        try:
            from arcengine import GameAction
            result = env.step(GameAction.RESET)
            if result is None:
                return None
            self._step_counts[game_id] = 0
            self._prev_levels[game_id] = 0
            return self._convert_frame(result, game_id)
        except Exception as e:
            logger.error("[ArcSDK] reset(%s) failed: %s", game_id, e)
            return None

    def step(self, game_id: str, action: int) -> Optional[Frame]:
        """Execute action, return frame with derived reward."""
        env = self._envs.get(game_id)
        if env is None:
            return None
        try:
            from arcengine import GameAction
            ga = GameAction.from_id(action)
            result = env.step(ga)
            if result is None:
                return None
            self._step_counts[game_id] = self._step_counts.get(game_id, 0) + 1
            frame = self._convert_frame(result, game_id)
            # Derive reward from level completion
            prev_levels = self._prev_levels.get(game_id, 0)
            if frame.levels_completed > prev_levels:
                frame.reward = 1.0  # level completed
            else:
                step_count = self._step_counts[game_id]
                frame.reward = -0.001  # small step penalty
                # Stuck penalty after 100 steps on same level
                steps_this_level = step_count  # approximate
                if steps_this_level > 100:
                    frame.reward = -0.01
            self._prev_levels[game_id] = frame.levels_completed
            return frame
        except Exception as e:
            logger.error("[ArcSDK] step(%s, %d) failed: %s", game_id, action, e)
            return None

    def get_scorecard(self) -> Optional[dict]:
        """Get current scorecard as dict."""
        if not self._arcade:
            return None
        try:
            sc = self._arcade.get_scorecard()
            if sc is None:
                return None
            # Scorecard is a string (JSON-formatted) from SDK
            if isinstance(sc, str):
                import json
                return json.loads(sc)
            elif isinstance(sc, dict):
                return sc
            elif hasattr(sc, "model_dump"):
                return sc.model_dump()
            return {"raw": str(sc)}
        except Exception as e:
            logger.error("[ArcSDK] get_scorecard failed: %s", e)
            return None

    def close(self) -> None:
        """Cleanup environments."""
        self._envs.clear()
        self._step_counts.clear()
        self._prev_levels.clear()
        logger.info("[ArcSDK] Closed")

    def get_stats(self) -> dict:
        """Return bridge status."""
        return {
            "initialized": self._initialized,
            "active_envs": list(self._envs.keys()),
            "step_counts": dict(self._step_counts),
        }

    # ── Internal Helpers ────────────────────────────────────────────

    def _resolve_game_id(self, short_id: str) -> Optional[str]:
        """Resolve short game ID (e.g. 'ls20') to full SDK ID (e.g. 'ls20-9607627b')."""
        envs = self._arcade.get_environments()
        for e in envs:
            full_id = e.game_id if hasattr(e, "game_id") else ""
            if full_id.startswith(short_id):
                return full_id
        # Try exact match
        for e in envs:
            if (hasattr(e, "game_id") and e.game_id == short_id):
                return short_id
        return None

    def _convert_frame(self, raw, game_id: str) -> Frame:
        """Convert SDK FrameDataRaw to our Frame dataclass."""
        from arcengine import GameState
        grid = np.zeros((64, 64), dtype=np.int8)
        if hasattr(raw, "frame") and raw.frame:
            grid = raw.frame[0] if isinstance(raw.frame[0], np.ndarray) else np.array(raw.frame[0], dtype=np.int8)

        state_str = "NOT_FINISHED"
        if hasattr(raw, "state"):
            if raw.state == GameState.WIN:
                state_str = "WIN"
            elif raw.state == GameState.GAME_OVER:
                state_str = "GAME_OVER"
            elif raw.state == GameState.NOT_PLAYED:
                state_str = "NOT_PLAYED"

        return Frame(
            grid=grid,
            available_actions=list(raw.available_actions) if hasattr(raw, "available_actions") else [],
            state=state_str,
            levels_completed=getattr(raw, "levels_completed", 0),
            win_levels=getattr(raw, "win_levels", 0),
            step_count=self._step_counts.get(game_id, 0),
            reward=0.0,  # set by caller
            done=state_str in ("WIN", "GAME_OVER"),
            game_id=game_id,
        )

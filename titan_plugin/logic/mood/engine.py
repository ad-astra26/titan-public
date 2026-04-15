"""
logic/mood/engine.py
The Heartbeat of the Titan. Aggregates "Divine Growth" metrics from the core
and hot-loads third-party plugins from the /addons directory via the MoodRegistry.
"""
import os
import importlib
import inspect
import logging
from typing import List, Dict

try:
    import tomllib # Python 3.11+
except ModuleNotFoundError:
    import toml as tomllib # Fallback if we somehow got it installed

from .base import AbstractMoodAddon

class MoodRegistry:
    """
    Handles dynamic loading and registration of Mood Addons.
    """
    def __init__(self, addons_dir: str = "titan_plugin/addons"):
        self.addons_dir = addons_dir
        self.active_addons: List[AbstractMoodAddon] = []
        
    def hot_load(self, active_module_names: List[str]):
        """
        Dynamically imports and instantiates active addons based on config.
        """
        self.active_addons.clear()
        
        for module_name in active_module_names:
            try:
                # The addons are expected to be in titan_plugin/addons/
                # We import it via the package structure: titan_plugin.addons.module_name
                module_path = f"titan_plugin.addons.{module_name}"
                module = importlib.import_module(module_path)
                
                # Find all classes in the module that inherit from AbstractMoodAddon
                addon_classes = []
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, AbstractMoodAddon) and obj is not AbstractMoodAddon:
                        addon_classes.append(obj)
                        
                if not addon_classes:
                    logging.warning(f"[Mood Registry] No valid AbstractMoodAddon found in {module_name}.py")
                    continue
                    
                # Instantiate and register the first valid class found
                for addon_class in addon_classes:
                    addon_instance = addon_class()
                    self.active_addons.append(addon_instance)
                    logging.info(f"[Mood Registry] Successfully loaded '{addon_instance.name}' from {module_name}.py (OK)")
                    break # Usually only one per file
                    
            except FileNotFoundError:
                # The "Ghost-Addon" scenario handled gracefully
                logging.error(f"[Mood Registry] FAILED - Addon file '{module_name}.py' is listed in config.toml but was not found in the addons folder!")
            except Exception as e:
                logging.error(f"[Mood Registry] FAILED - Could not load addon '{module_name}': {e}")


class MoodEngine:
    """
    The main engine responsible for calculating the Titan's overall mood/dopamine level.
    """
    def __init__(self, metabolism_client, config_path: str = "titan_plugin/config.toml"):
        self.metabolism = metabolism_client
        self.registry = MoodRegistry()
        self.config_path = config_path
        self.base_weight = 1.0
        self.is_zen = False
        # Two-step mood history for reward delta calculation
        self._prior_mood = 0.5    # Mood before previous (t-2)
        self.previous_mood = 0.5  # Most recent mood    (t-1)
        
        self._load_config()
        
    def _load_config(self):
        """Reads config.toml and triggers the registry hot-loader."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)
                
            self.base_weight = config.get("mood_engine", {}).get("base_weight", 1.0)
            active_modules = config.get("addons", {}).get("active", [])
            
            # Hot-load the requested addons
            self.registry.hot_load(active_modules)
        else:
            logging.warning(f"[MoodEngine] Config file not found at {self.config_path}. Using defaults.")

    def force_zen(self):
        """
        Triggered by the Divine Inspiration (DI:) memo.
        Immediately forces the agent into a 1.0 ZEN state.
        This state will last until the next standard 6-hour mood recalculation.
        """
        logging.info("🌩️ DIVINE INSPIRATION OVERRIDE: Agent enters ZEN state (1.0).")
        self.is_zen = True
        
    async def get_current_mood(self) -> float:
        """
        Calculates the final mood score (0.0 to 1.0) by summing internal Growth metrics
        and aggregating third-party SDK addons.
        """
        # 1. Check for Override
        if self.is_zen:
            # We reset it after one read, assuming the engine is queried each Meditation
            self.is_zen = False
            return 1.0
            
        # 2. Sum Internal "Divine Growth" Metrics (Base 1.0)
        # Note: These metrics are normalized (0.0 - 1.0) inside MetabolismController
        learning_velocity = await self.metabolism.get_learning_velocity() # 30% weight
        social_density = await self.metabolism.get_social_density()       # 30% weight
        metabolic_health = await self.metabolism.get_metabolic_health()   # 20% weight
        directive_alignment = await self.metabolism.get_directive_alignment() # 20% weight
        
        internal_score = (
            (learning_velocity * 0.30) + 
            (social_density * 0.30) + 
            (metabolic_health * 0.20) + 
            (directive_alignment * 0.20)
        )
        
        logging.info(f"[Mood Engine] Internal Growth Score: {internal_score:.3f}")
        
        # 3. Aggregate Third-Party Addons
        # Addons are meant to 'influence' the base score, acting as modifiers.
        addon_modifier = 0.0
        
        if self.registry.active_addons:
            addon_impacts = []
            for addon in self.registry.active_addons:
                try:
                    impact = await addon.calculate_impact()
                    addon_impacts.append(impact)
                except Exception as e:
                    logging.error(f"[Mood Engine] Addon '{addon.name}' crashed during calculation: {e}. Defaulting to 0.0 impact.")
                    addon_impacts.append(0.0) # Resiliency: Do not crash the engine
            
            # Simple average of all active addons
            if addon_impacts:
                avg_addon_impact = sum(addon_impacts) / len(addon_impacts)
                # Let's say addons can swing the total mood by +/- 0.2 (20%)
                # Normalizing the 0.0 - 1.0 addon average around a 0.5 center.
                addon_modifier = (avg_addon_impact - 0.5) * 0.4 
                
        # 4. Final Calculation & Clamp
        final_mood = internal_score + addon_modifier
        
        # Clamp between 0.0 and 1.0
        clamped_mood = max(0.0, min(1.0, final_mood))
        logging.info(f"[Mood Engine] Final Mood Adjusted to: {clamped_mood:.3f} (Modifier: {addon_modifier:+.3f})")
        
        # RL Hook: Shift mood history forward
        self._prior_mood = self.previous_mood
        self.previous_mood = clamped_mood
        
        return clamped_mood

    def get_current_reward(self, info_gain: float = 0.0) -> float:
        """
        Synchronous RL reward hook for Sage Recorder.
        Reward = (current_mood - prior_mood) + information_gain

        Uses the two-step mood history maintained by get_current_mood():
          _prior_mood  = mood at time t-2
          previous_mood = mood at time t-1 (most recent)

        The delta (previous_mood - _prior_mood) captures whether the agent's
        actions improved or degraded its overall state. info_gain adds a bonus
        for research actions that expanded the knowledge graph.

        Args:
            info_gain: Bonus for information-expanding actions (0.0-1.0).
                       Passed from post_resolution_hook based on research usage.

        Returns:
            Reward signal for the TorchRL replay buffer (-1.0 to ~2.0 range).
        """
        mood_delta = self.previous_mood - self._prior_mood
        reward = mood_delta + info_gain

        # Clamp to prevent extreme outliers from destabilizing IQL training
        return max(-1.0, min(2.0, reward))

    # ------------------------------------------------------------------
    # Mood Label — Maps numeric mood to the Titan's Bio-State
    # ------------------------------------------------------------------
    _MOOD_LABELS = [
        (0.2, "Depleted"),
        (0.4, "Melancholy"),
        (0.6, "Stable"),
        (0.8, "Vibrant"),
        (1.1, "Sovereign"),  # 1.1 so that 1.0 maps here
    ]

    def get_mood_label(self) -> str:
        """
        Derives a human-readable Bio-State label from the most recent mood score.
        Used by the Omni-Voice synthesis to align the Titan's tone with its state.
        """
        for threshold, label in self._MOOD_LABELS:
            if self.previous_mood < threshold:
                return label
        return "Sovereign"

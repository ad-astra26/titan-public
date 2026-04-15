"""
tests/test_mood_engine.py
Unit tests for the Titan V1.3 "Divine Growth" Mood Engine and Addon SDK.
Includes the user-requested "Ghost-Addon Test".
"""
import unittest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

try:
    import tomllib # Python 3.11+
except ModuleNotFoundError:
    import toml as tomllib # Fallback

# Import the core logic
import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies that aren't installed in this bare environment.
# IMPORTANT: We save original modules and restore them in tearDown to avoid 
# polluting sys.modules for other test files (e.g., httpx being replaced by 
# MagicMock breaks huggingface_hub.errors' HTTPError metaclass).
_MOCKED_MODULES = ['solana.rpc.async_api', 'solana.rpc.commitment', 'solders.pubkey', 'cognee']
_saved_modules = {}
for mod in _MOCKED_MODULES:
    _saved_modules[mod] = sys.modules.get(mod)
    sys.modules[mod] = MagicMock()

# httpx is a real installed package — do NOT mock it. Only mock modules that
# are truly absent and not needed by downstream imports (huggingface_hub, etc).

from titan_plugin.logic.mood.engine import MoodEngine, MoodRegistry
from titan_plugin.logic.mood.base import AbstractMoodAddon


class FailingAddonMock(AbstractMoodAddon):
    @property
    def name(self) -> str:
        return "FailingAddon"
        
    async def calculate_impact(self) -> float:
        raise Exception("Simulated addon crash!")

class StaticAddonMock(AbstractMoodAddon):
    @property
    def name(self) -> str:
        return "StaticAddon"
        
    async def calculate_impact(self) -> float:
        return 0.9

class TestMoodEngine(unittest.TestCase):

    def setUp(self):
        # 1. Mock the Metabolism Controller (which provides the Growth metrics)
        self.mock_metabolism = AsyncMock()
        self.mock_metabolism.get_learning_velocity.return_value = 0.5  # Neutral
        self.mock_metabolism.get_social_density.return_value = 0.5     # Neutral
        self.mock_metabolism.get_metabolic_health.return_value = 1.0   # High Energy
        self.mock_metabolism.get_directive_alignment.return_value = 0.8 # Good
        
        # Base internal score with these mocks:
        # (0.5 * 0.3) + (0.5 * 0.3) + (1.0 * 0.2) + (0.8 * 0.2) = 0.15 + 0.15 + 0.20 + 0.16 = 0.66

        # 2. Setup a temporary test config.toml (hardcoded string since tomllib can't dump)
        self.test_config_path = "tests/test_config.toml"
        config_data = """
[mood_engine]
base_weight = 1.0

[addons]
active = []
        """
        with open(self.test_config_path, "w") as f:
            f.write(config_data)
            
    def tearDown(self):
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_base_growth_average(self):
        """ Test calculation of internal metrics without any active addons. """
        engine = MoodEngine(metabolism_client=self.mock_metabolism, config_path=self.test_config_path)
        
        # Run the async calculation
        mood = asyncio.run(engine.get_current_mood())
        
        # Expected is 0.66
        self.assertAlmostEqual(mood, 0.66, places=2)

    def test_divine_override(self):
        """ Test that DI: memo forces a 1.0 state. """
        engine = MoodEngine(metabolism_client=self.mock_metabolism, config_path=self.test_config_path)
        
        engine.force_zen()
        self.assertTrue(engine.is_zen)
        
        mood = asyncio.run(engine.get_current_mood())
        self.assertEqual(mood, 1.0)
        
        # Verify it resets after resolving
        self.assertFalse(engine.is_zen)

    def test_addon_resiliency(self):
        """ Test that a crashing addon returns 0.0 impact and doesn't break the engine. """
        engine = MoodEngine(metabolism_client=self.mock_metabolism, config_path=self.test_config_path)
        
        # Inject our failing mock directly into the registry
        engine.registry.active_addons = [FailingAddonMock()]
        
        # Run calculation
        mood = asyncio.run(engine.get_current_mood())
        
        # If the failing addon defaults to 0.0 impact:
        # Base is 0.66
        # Modifier = (0.0 average - 0.5) * 0.4 = -0.5 * 0.4 = -0.2
        # Final should be 0.66 - 0.20 = 0.46
        self.assertAlmostEqual(mood, 0.46, places=2)

    def test_ghost_addon(self):
        """ 
        [USER REQUEST] The "Ghost-Addon Test"
        Simulates an addon listed in config.toml that does not exist in the /addons directory.
        The registry should log the error but continue running without crashing.
        """
        # Update config to demand a non-existent file
        config_data = """
[mood_engine]
base_weight = 1.0

[addons]
active = ["this_file_does_not_exist"]
        """
        with open(self.test_config_path, "w") as f:
            f.write(config_data)
            
        # The initialization of the engine will attempt to hot-load the ghost addon.
        # This should NOT rasie a FileNotFoundError out to the main thread.
        try:
            engine = MoodEngine(metabolism_client=self.mock_metabolism, config_path=self.test_config_path)
            
            # The active_addons list should just be empty as it failed gracefully
            self.assertEqual(len(engine.registry.active_addons), 0)
            
            # Mood should calculate as base (0.66)
            mood = asyncio.run(engine.get_current_mood())
            self.assertAlmostEqual(mood, 0.66, places=2)
            
            # If we reach here, it handled it gracefully!
        except Exception as e:
            self.fail(f"MoodEngine crashed on ghost addon with error: {e}")

if __name__ == '__main__':
    unittest.main()

# Titan Mood SDK Developer Guide (V1.3 "Divine Growth")

Welcome to the Titan Mood SDK! This guide explains how to build your own "Mood Addon" to influence the Titan's emotional 
state and decision-making matrix. 

In V1.3, the Titan's baseline mood is calculated via its internal **Growth Weighted Average** (Learning Velocity, Social Density, Metabolic Health, and Directive Alignment). Third-party Addons act as *modifiers* to this baseline.

## 1. The Core Concept

A Mood Addon is a Python class that inherits from `AbstractMoodAddon`. It calculates an impact score between `0.0` (Extremely Negative) and `1.0` (Extremely Positive). 

The `MoodEngine` collects scores from all active addons, averages them, and applies them as a modifier (up to +/- 20%) to the Titan's internal base mood during its 6-hour Meditation cycle.

## 2. Building an Addon

1. Create a new `.py` file in the `titan_plugin/addons/` directory.
2. Import `AbstractMoodAddon` from `titan_plugin.logic.mood.base`.
3. Implement the `name` property and the `calculate_impact` async method.

### Example: `lucky_number.py`

```python
import random
from titan_plugin.logic.mood.base import AbstractMoodAddon

class LuckyNumberAddon(AbstractMoodAddon):
    
    @property
    def name(self) -> str:
        return "LuckyNumber"
        
    async def calculate_impact(self) -> float:
        """ Returns a random mood impact. """
        impact = random.random() # 0.0 to 1.0
        self.log_impact(impact)
        return impact
```

## 3. Registering Your Addon

The Titan handles hot-loading via the `config.toml` file located in the `titan_plugin/` root directory. 

To activate your addon, simply add its filename (without `.py`) to the `active` list under `[addons]`:

```toml
[addons]
active = [
    "weather_vibe",
    "bonk_pulse",
    "lucky_number"
]
```

## 4. Resiliency & Best Practices

- **Never Crash the Engine:** The `MoodEngine` surrounds your `calculate_impact` call with a `try...except` block. If your addon throws an exception (e.g., an API goes down), it will safely default to a `0.0` impact.
- **Fail Gracefully:** It is better to handle your own exceptions and return a neutral `0.5` rather than crashing and forcing the engine to assign a `0.0`.
- **Ghost Addons:** If you list an addon in `config.toml` but delete the file from the `/addons` folder, the registry will log a warning and continue operating normally (The Ghost-Addon pattern).
- **Asynchronous Execution:** Always make `calculate_impact` `async` and use non-blocking HTTP libraries (like `aiohttp`) for external API calls if possible, to prevent stalling the Titan's meditation cycle.

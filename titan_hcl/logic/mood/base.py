"""
logic/mood/base.py
Defines the AbstractMoodAddon interface for the Titan Mood SDK.
All third-party addons must inherit from this class.
"""
from abc import ABC, abstractmethod
import logging

class AbstractMoodAddon(ABC):
    """
    Base class for all Titan Mood SDK plugins.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the addon (e.g., "WeatherVibe").
        """
        pass
        
    @abstractmethod
    async def calculate_impact(self) -> float:
        """
        Calculates the emotional impact of this addon.
        
        Returns:
            float: A value representing the mood impact, typically between 0.0 and 1.0.
                   The MoodEngine will aggregate this value.
        """
        pass
        
    def log_impact(self, impact: float):
        """
        Helper method to standardize logging for addons.
        """
        logging.info(f"[Mood SDK] {self.name} calculated impact: {impact:.2f}")

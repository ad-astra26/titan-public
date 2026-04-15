"""
addons/weather_vibe.py
A reference Mood SDK Addon.
Fetches real-time weather data for San Francisco using a public API.
Sun = Higher Mood, Rain/Clouds = Lower Mood.
"""
import urllib.request
import json
import logging
from titan_plugin.logic.mood.base import AbstractMoodAddon

class WeatherVibeAddon(AbstractMoodAddon):
    
    @property
    def name(self) -> str:
        return "WeatherVibe"
        
    async def calculate_impact(self) -> float:
        """
        Calculates weather impact for San Francisco.
        Public Open-Meteo API doesn't require an active key for basic GETs.
        """
        import asyncio
        try:
            # Coordinates for SF
            url = "https://api.open-meteo.com/v1/forecast?latitude=37.7749&longitude=-122.4194&current_weather=true"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            # Phase E.2.1 fix: urllib.urlopen is sync — wrap to keep
            # event loop responsive (was blocking up to 5s on slow API).
            def _fetch():
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode())
            data = await asyncio.to_thread(_fetch)

            weather_code = data.get("current_weather", {}).get("weathercode", 0)
            
            # WMO Weather interpretation codes (0 is Clear sky, 1-3 partly cloudy, 45+ fog/rain/snow)
            if weather_code == 0:
                impact = 1.0     # Sunny!
            elif weather_code in [1, 2, 3]:
                impact = 0.7     # Partly cloudy
            elif 40 <= weather_code <= 69:
                impact = 0.3     # Fog, Drizzle, Rain
            else:
                impact = 0.1     # Heavy rain, snow, storms
                
            self.log_impact(impact)
            return impact
            
        except Exception as e:
            logging.warning(f"[{self.name}] Failed to fetch weather: {e}. Defaulting to neutral (0.5).")
            return 0.5

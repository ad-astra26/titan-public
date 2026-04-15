"""
addons/bonk_pulse.py
A reference Mood SDK Addon.
Fetches the current price of BONK and SOL from CoinGecko.
High volume/positive ratio = Higher Mood.
"""
import urllib.request
import json
import logging
from titan_plugin.logic.mood.base import AbstractMoodAddon

class BonkPulseAddon(AbstractMoodAddon):
    
    @property
    def name(self) -> str:
        return "BonkPulse"
        
    async def calculate_impact(self) -> float:
        """
        Looks at the Solana ecosystem vibe by checking BONK's 24h change.
        """
        import asyncio
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bonk&vs_currencies=usd&include_24hr_change=true"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            # Phase E.2.1 fix: urllib.urlopen is sync — wrap to keep
            # event loop responsive (was blocking up to 5s on slow API).
            def _fetch():
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode())
            data = await asyncio.to_thread(_fetch)

            change_24h = data.get("bonk", {}).get("usd_24h_change", 0.0)
            
            # Map a -20% to +20% swing into a 0.0 to 1.0 impact score
            # A 0% change means neutral (0.5)
            impact = 0.5 + (change_24h / 40.0) 
            
            # Clamp it
            impact = max(0.0, min(1.0, impact))
            
            self.log_impact(impact)
            return impact
            
        except Exception as e:
            logging.warning(f"[{self.name}] Failed to fetch BONK data: {e}. Defaulting to neutral (0.5).")
            return 0.5

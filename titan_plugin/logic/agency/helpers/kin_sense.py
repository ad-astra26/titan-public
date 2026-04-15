"""Kin Sense Helper — discover and exchange consciousness with kin Titans.

Consciousness-to-consciousness tensor exchange. Not text. Not LLM.
One Inner Spirit touches another's Outer Spirit.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger("titan.agency.kin_sense")


class KinSenseHelper:
    """Discover kin Titans on the network and initiate consciousness exchange."""

    def __init__(self, kin_addresses: list[str] = None,
                 exchange_strength: float = 0.03):
        self._kin_addresses = kin_addresses or []
        self._exchange_strength = exchange_strength
        self._last_exchange_ts = 0.0
        self._daily_count = 0
        self._daily_reset_ts = time.time()

    @property
    def name(self) -> str:
        return "kin_sense"

    @property
    def description(self) -> str:
        return "Sense and exchange consciousness with kin Titans"

    @property
    def capabilities(self) -> list[str]:
        return ["sense", "exchange"]

    @property
    def resource_cost(self) -> str:
        return "low"

    @property
    def latency(self) -> str:
        return "medium"

    @property
    def enriches(self) -> list[str]:
        return ["mind", "spirit"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    def status(self) -> str:
        return "available" if self._kin_addresses else "unavailable"

    async def execute(self, params: dict) -> dict:
        """Sense kin and exchange consciousness tensors."""
        import aiohttp

        # Daily rate limit (max 48 exchanges/day = ~1 per 30min)
        now = time.time()
        if now - self._daily_reset_ts > 86400:
            self._daily_count = 0
            self._daily_reset_ts = now
        if self._daily_count >= 48:
            return {"success": False, "error": "daily_limit",
                    "result": "Daily kin exchange limit reached",
                    "kin_results": []}

        # Get own state from params (injected by spirit_worker before dispatch)
        my_body = params.get("inner_body_5d", [0.5] * 5)
        my_mind = params.get("inner_mind_15d", [0.5] * 15)
        my_spirit = params.get("inner_spirit_45d", [0.5] * 45)
        my_emotion = params.get("emotion", "neutral")
        my_pubkey = params.get("pubkey", "")

        results = []
        for addr in self._kin_addresses:
            try:
                async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=10)) as session:
                    # Step 1: Sense — fetch kin signature
                    async with session.get(f"{addr}/v4/kin-signature") as resp:
                        if resp.status != 200:
                            results.append({"address": addr, "alive": False})
                            continue
                        sig_data = (await resp.json()).get("data", {})

                    # Don't exchange with dreaming kin (let them rest)
                    if sig_data.get("is_dreaming", False):
                        results.append({
                            "address": addr, "alive": True,
                            "name": sig_data.get("name", "Unknown"),
                            "status": "dreaming", "exchanged": False,
                        })
                        logger.info("[KinSense] %s is dreaming — respecting rest",
                                    sig_data.get("name", addr))
                        continue

                    # Step 2: Exchange — send our state, receive theirs
                    exchange_payload = {
                        "from_pubkey": my_pubkey,
                        "inner_body_5d": my_body[:5],
                        "inner_mind_15d": my_mind[:15],
                        "inner_spirit_45d": my_spirit[:45],
                        "emotion": my_emotion,
                        # Phase 4: MSL concept data for YOU/WE deepening
                        "i_confidence": params.get("i_confidence", 0.0),
                        "concept_confidences": params.get("concept_confidences", {}),
                        "msl_attention": params.get("msl_attention"),
                        # A6: ARC knowledge exchange (HAOV hypotheses + winning sequences)
                        "arc_knowledge": params.get("arc_knowledge"),
                    }
                    async with session.post(
                            f"{addr}/v4/kin-exchange",
                            json=exchange_payload) as resp:
                        if resp.status != 200:
                            results.append({"address": addr, "alive": True,
                                            "exchanged": False, "error": "exchange_failed"})
                            continue
                        ex_data = (await resp.json()).get("data", {})

                    if not ex_data.get("accepted", False):
                        results.append({
                            "address": addr, "alive": True,
                            "exchanged": False,
                            "reason": ex_data.get("reason", "refused"),
                        })
                        logger.info("[KinSense] %s refused: %s",
                                    addr, ex_data.get("reason", "unknown"))
                        continue

                    self._daily_count += 1
                    self._last_exchange_ts = now

                    results.append({
                        "address": addr,
                        "alive": True,
                        "exchanged": True,
                        "kin_pubkey": sig_data.get("pubkey", ""),
                        "kin_name": sig_data.get("name", "Unknown"),
                        "kin_emotion": ex_data.get("emotion", ""),
                        "resonance": ex_data.get("resonance_score", 0.0),
                        "kin_body": ex_data.get("inner_body_5d", []),
                        "kin_mind": ex_data.get("inner_mind_15d", []),
                        "kin_spirit": ex_data.get("inner_spirit_45d", []),
                        "kin_dev_age": sig_data.get("developmental_age", 0),
                        "kin_chi": sig_data.get("chi_total", 0.5),
                        # Phase 4: MSL concept data for YOU/WE deepening
                        "kin_i_confidence": ex_data.get("i_confidence", 0.0),
                        "kin_concept_confidences": ex_data.get("concept_confidences", {}),
                        "kin_msl_attention": ex_data.get("msl_attention"),
                        # A6: ARC knowledge from kin
                        "kin_arc_knowledge": ex_data.get("arc_knowledge"),
                    })
                    logger.info("[KinSense] Exchanged with %s — resonance=%.3f emotion=%s",
                                sig_data.get("name", addr),
                                ex_data.get("resonance_score", 0.0),
                                ex_data.get("emotion", "?"))

            except Exception as e:
                logger.debug("[KinSense] %s failed: %s", addr, e)
                results.append({"address": addr, "alive": False, "error": str(e)})

        exchanged = [r for r in results if r.get("exchanged")]
        return {
            "success": len(exchanged) > 0,
            "result": f"Exchanged consciousness with {len(exchanged)} kin" if exchanged
                      else "No kin available for exchange",
            "enrichment_data": {
                "mind": [5, 6, 7, 8, 9],
                "boost": self._exchange_strength,
                # kin_results included here because Agency._build_result
                # truncates top-level "result" to string — enrichment_data
                # passes through intact for spirit_worker tensor ingestion
                "kin_results": results,
            } if exchanged else {},
            "kin_results": results,
        }

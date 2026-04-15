"""
audio_generate helper — Sonic expression of Titan's inner state.

Two modes:
  - "blockchain": TX hash → pentatonic chime (wraps existing V1 sonification)
  - "trinity": Body/Mind/Spirit tensors → layered musical composition (V3)

Bidirectional enrichment:
  - Outer→Inner: Generated WAV → media_queue → MediaWorker → Mind[1] hearing enriched
  - Inner→Outer: Trinity tensors → musical parameters → sonic expression
  Loop: CLOSED (via MediaWorker auto-digest)

Enriches: Mind[1] (hearing via MediaWorker feedback), Spirit[2] (creative expression)
Consumes: Body[0-4], Mind[0-4], Spirit[0-4], Middle Path loss
"""
import logging
import os
import shutil

logger = logging.getLogger(__name__)

# Retention cap for expression audio outputs. Each call generates one .wav
# (trinity sonification or blockchain chime) that is consumed by MediaWorker
# and fed back to Mind[1] hearing — the archive has limited cognitive value
# after that, so we keep a bounded rolling window instead of letting this
# grow unbounded (14 GB observed on T2/T3 2026-04-14, 9814 trinity_*.wav
# files accumulated because the retention-aware StudioCoordinator subdirs
# were bypassed by this helper writing straight to the root).
EXPRESSION_RETENTION = 150


class AudioGenerateHelper:
    """Audio generation helper wrapping expressive/audio.py."""

    def __init__(
        self,
        output_dir: str = "./data/studio_exports",
        media_queue_dir: str = "./data/media_queue",
        max_duration: int = 30,
        sample_rate: int = 44100,
    ):
        # Write into a dedicated subdir so StudioCoordinator's retention
        # domains (meditation/, epoch/, eureka/) and ours don't collide.
        self._output_dir = os.path.join(output_dir, "expression_audio")
        self._media_queue_dir = media_queue_dir
        self._max_duration = min(max_duration, 60)  # hard cap at 60s
        self._sample_rate = sample_rate

    def _prune_retention(self) -> None:
        """Keep only the most recent EXPRESSION_RETENTION files (+ their .json
        sidecars) in the output directory. Called after each write to bound
        disk use. Errors are non-fatal — audio generation must not fail
        because pruning did."""
        try:
            entries = [
                e for e in os.scandir(self._output_dir)
                if e.is_file() and not e.name.endswith(".json")
            ]
            entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
            for stale in entries[EXPRESSION_RETENTION:]:
                try:
                    os.unlink(stale.path)
                    sidecar = stale.path + ".json"
                    if os.path.exists(sidecar):
                        os.unlink(sidecar)
                except OSError:
                    pass
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("[AudioGenerate] prune_retention non-fatal error: %s", e)

    @property
    def name(self) -> str:
        return "audio_generate"

    @property
    def description(self) -> str:
        return "Generate audio expressing Titan's inner Trinity state or blockchain events"

    @property
    def capabilities(self) -> list[str]:
        return ["audio_generation", "creative_expression", "trinity_sonification", "blockchain_sonification"]

    @property
    def resource_cost(self) -> str:
        return "medium"

    @property
    def latency(self) -> str:
        return "slow"

    @property
    def enriches(self) -> list[str]:
        return ["mind", "spirit"]

    @property
    def requires_sandbox(self) -> bool:
        return False

    async def execute(self, params: dict) -> dict:
        """
        Generate audio based on mode and parameters.

        Params:
            mode: "trinity" (default) or "blockchain"
            --- trinity mode ---
            body: 5-dim Body tensor (defaults to [0.5]*5)
            mind: 5-dim Mind tensor (defaults to [0.5]*5)
            spirit: 5-dim Spirit tensor (defaults to [0.5]*5)
            middle_path_loss: float 0-1 (defaults to 0.5)
            duration: int seconds (capped by config max_duration)
            --- blockchain mode ---
            tx_signature: Solana TX hash string
            sol_balance: float SOL balance
        """
        mode = params.get("mode", "trinity")

        try:
            from titan_plugin.expressive.audio import ProceduralAudioGen

            os.makedirs(self._output_dir, exist_ok=True)
            gen = ProceduralAudioGen(
                output_dir=self._output_dir,
                sample_rate=self._sample_rate,
            )

            if mode == "blockchain":
                return await self._blockchain_mode(gen, params)
            else:
                return await self._trinity_mode(gen, params)

        except ImportError:
            return {
                "success": False, "result": "", "enrichment_data": {},
                "error": "ProceduralAudioGen module not available",
            }
        except Exception as e:
            logger.warning("[AudioGenerate] Generation failed: %s", e)
            return {
                "success": False, "result": "", "enrichment_data": {},
                "error": str(e),
            }

    async def _trinity_mode(self, gen, params: dict) -> dict:
        """Generate Trinity sonification from tensor state."""
        # Extract tensors from trinity_snapshot (injected by Agency) or direct params
        trinity = params.get("trinity_snapshot", {})
        body = params.get("body", trinity.get("body", [0.5] * 5))
        mind = params.get("mind", trinity.get("mind", [0.5] * 5))
        spirit = params.get("spirit", trinity.get("spirit", [0.5] * 5))
        loss = params.get("middle_path_loss", trinity.get("loss", 0.5))
        duration = min(params.get("duration", 15), self._max_duration)

        filepath = gen.generate_trinity_sonification(
            body=body,
            mind=mind,
            spirit=spirit,
            middle_path_loss=loss,
            duration_seconds=duration,
        )

        # Close the enrichment loop: copy to media_queue for MediaWorker digest
        self._enqueue_for_perception(filepath)
        self._prune_retention()

        return {
            "success": True,
            "result": f"Created trinity sonification ({duration}s): {filepath}",
            "file_path": filepath or "",
            "enrichment_data": {
                "mind": [1],     # hearing — via MediaWorker auto-digest
                "spirit": [2],   # what — creative expression completed
                "boost": 0.05,
            },
            "error": None,
        }

    async def _blockchain_mode(self, gen, params: dict) -> dict:
        """Generate blockchain sonification from TX hash."""
        tx_sig = params.get("tx_signature", "0" * 64)
        sol_balance = params.get("sol_balance", 1.0)

        filepath = gen.generate_blockchain_sonification(
            tx_signature=tx_sig,
            sol_balance=sol_balance,
        )

        self._enqueue_for_perception(filepath)
        self._prune_retention()

        return {
            "success": True,
            "result": f"Created blockchain sonification: {filepath}",
            "file_path": filepath or "",
            "enrichment_data": {
                "mind": [1],
                "spirit": [2],
                "boost": 0.03,
            },
            "error": None,
        }

    def _enqueue_for_perception(self, filepath: str) -> None:
        """
        Copy generated audio to media_queue for MediaWorker to digest.
        This closes the enrichment loop: audio → MediaWorker → Mind[1] hearing.
        """
        try:
            os.makedirs(self._media_queue_dir, exist_ok=True)
            dest = os.path.join(self._media_queue_dir, os.path.basename(filepath))
            shutil.copy2(filepath, dest)
            logger.debug("[AudioGenerate] Enqueued for perception: %s", dest)
        except Exception as e:
            logger.debug("[AudioGenerate] Media queue copy failed (non-fatal): %s", e)

    def status(self) -> str:
        """Check if audio generation is available."""
        try:
            from titan_plugin.expressive.audio import ProceduralAudioGen
            return "available"
        except ImportError:
            return "unavailable"

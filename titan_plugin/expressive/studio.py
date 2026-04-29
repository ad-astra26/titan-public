"""
expressive/studio.py
Centralized StudioCoordinator — orchestrates all Titan creative expression.

Manages:
- Output directory structure (meditation/, epoch/, eureka/ subdirs)
- Resolution scaling based on metabolic budget
- NFT composite rendering (flow field + L-system overlay)
- JSON metadata sidecars for every artifact
- GFS-style retention pruning
- Async rendering via asyncio.to_thread()
- Gallery index for Observatory API
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)


class StudioCoordinator:
    """
    Centralized creative engine that coordinates art, audio, and text generation.
    All rendering runs in asyncio.to_thread() to prevent event loop blocking.
    """

    def __init__(self, config: dict = None, metabolism=None):
        """
        Args:
            config: [expressive] section from config.toml.
            metabolism: MetabolismController for budget gating (optional).
        """
        config = config or {}
        self._metabolism = metabolism

        # Paths
        self._output_root = Path(config.get("output_path", "./data/studio_exports"))
        self._meditation_dir = self._output_root / "meditation"
        self._epoch_dir = self._output_root / "epoch"
        self._eureka_dir = self._output_root / "eureka"

        # Resolution tiers
        self._default_res = int(config.get("default_resolution", 1024))
        self._highres_res = int(config.get("highres_resolution", 2048))
        self._max_particles = int(config.get("max_particles", 20000))

        # Retention limits
        self._meditation_retention = int(config.get("meditation_retention", 40))
        self._epoch_retention = int(config.get("epoch_retention", 30))
        self._eureka_retention = int(config.get("eureka_retention", 20))

        # NFT composite toggle
        self._nft_composite_enabled = config.get("nft_composite_enabled", True)

        # Ollama Cloud client — wired by TitanPlugin.__init__ if configured
        self._ollama_cloud = None

        # Ensure directories exist
        for d in (self._meditation_dir, self._epoch_dir, self._eureka_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Metabolic Budget → Resolution
    # -------------------------------------------------------------------------
    async def _get_resolution(self, allow_highres: bool = False) -> int:
        """
        Determine rendering resolution based on SOL balance.
        HIGH energy (>1 SOL): full configured resolution
        LOW energy (0.1-1 SOL): 512px minimum
        STARVATION (<0.1 SOL): skip rendering entirely (returns 0)
        """
        if not self._metabolism:
            return self._highres_res if allow_highres else self._default_res

        try:
            state = await self._metabolism.get_current_state()
        except Exception:
            return self._default_res

        if state == "STARVATION":
            return 0
        elif state == "LOW":
            return 512
        else:
            return self._highres_res if allow_highres else self._default_res

    def _particle_count(self, resolution: int, age_nodes: int) -> int:
        """Scale particle count with resolution and age, capped by config max."""
        base = 100 + (age_nodes * 20)
        scale = (resolution / 512) ** 2
        return min(self._max_particles, int(base * scale))

    # -------------------------------------------------------------------------
    # Metadata Sidecar
    # -------------------------------------------------------------------------
    def _write_sidecar(self, artifact_path: Path, metadata: dict):
        """Write a JSON metadata sidecar alongside the artifact."""
        sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
        metadata["generated_at"] = datetime.utcnow().isoformat() + "Z"
        metadata["artifact"] = artifact_path.name
        try:
            with open(sidecar_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            swallow_warn('[Studio] Sidecar write failed', e,
                         key="expressive.studio.sidecar_write_failed", throttle=100)

    # -------------------------------------------------------------------------
    # Meditation Art (Small Epoch — every 6 hours)
    # -------------------------------------------------------------------------
    async def generate_meditation_art(
        self, state_root: str, age_nodes: int, avg_intensity: int
    ) -> str | None:
        """
        Generate a meditation flow field. Runs in a thread to avoid blocking.

        Returns:
            Path to the generated image, or None if skipped (STARVATION).
        """
        resolution = await self._get_resolution()
        if resolution == 0:
            logger.info("[Studio] Skipping meditation art — STARVATION mode.")
            return None

        num_particles = self._particle_count(resolution, age_nodes)

        def _render():
            from titan_plugin.expressive.art import ProceduralArtGen

            art_gen = ProceduralArtGen(output_dir=str(self._meditation_dir))
            return art_gen.generate_flow_field(
                state_root, age_nodes, avg_intensity,
                resolution=resolution, num_particles=num_particles,
            )

        try:
            filepath = await asyncio.to_thread(_render)
            self._write_sidecar(Path(filepath), {
                "type": "meditation_flow_field",
                "state_root": state_root,
                "age_nodes": age_nodes,
                "avg_intensity": avg_intensity,
                "resolution": resolution,
                "particles": num_particles,
            })
            self._prune_dir(self._meditation_dir, self._meditation_retention)
            logger.info("[Studio] Meditation art: %s", filepath)
            return filepath
        except Exception as e:
            logger.warning("[Studio] Meditation art failed: %s", e)
            return None

    # -------------------------------------------------------------------------
    # Epoch Art + Audio (Greater Epoch — every 24 hours)
    # -------------------------------------------------------------------------
    async def generate_epoch_bundle(
        self, tx_signature: str, total_nodes: int, beliefs_strength: int,
        sol_balance: float,
    ) -> dict:
        """
        Generate the Greater Epoch expressive bundle:
        - L-system tree
        - Blockchain sonification audio
        - NFT composite (if enabled and HIGH energy)

        Returns:
            dict with keys: tree_path, audio_path, composite_path (any may be None)
        """
        resolution = await self._get_resolution()
        result = {"tree_path": None, "audio_path": None, "composite_path": None}

        if resolution == 0:
            logger.info("[Studio] Skipping epoch bundle — STARVATION mode.")
            return result

        # L-system tree
        def _render_tree():
            from titan_plugin.expressive.art import ProceduralArtGen
            art_gen = ProceduralArtGen(output_dir=str(self._epoch_dir))
            return art_gen.generate_l_system_tree(
                tx_signature, total_nodes, beliefs_strength,
                resolution=resolution,
            )

        # Audio sonification
        def _render_audio():
            from titan_plugin.expressive.audio import ProceduralAudioGen
            audio_gen = ProceduralAudioGen(output_dir=str(self._epoch_dir))
            return audio_gen.generate_blockchain_sonification(tx_signature, sol_balance)

        try:
            tree_path, audio_path = await asyncio.gather(
                asyncio.to_thread(_render_tree),
                asyncio.to_thread(_render_audio),
            )
            result["tree_path"] = tree_path
            result["audio_path"] = audio_path
        except Exception as e:
            logger.warning("[Studio] Epoch rendering error: %s", e)

        # NFT Composite (flow field aura + tree overlay)
        if self._nft_composite_enabled and result["tree_path"]:
            highres = await self._get_resolution(allow_highres=True)
            if highres >= self._highres_res:
                try:
                    composite_path = await self._render_nft_composite(
                        tx_signature, total_nodes, beliefs_strength,
                        result["tree_path"], highres,
                    )
                    result["composite_path"] = composite_path
                except Exception as e:
                    logger.warning("[Studio] NFT composite failed: %s", e)

        # Sidecars
        for key, path in result.items():
            if path:
                self._write_sidecar(Path(path), {
                    "type": f"epoch_{key.replace('_path', '')}",
                    "tx_signature": tx_signature,
                    "total_nodes": total_nodes,
                    "beliefs_strength": beliefs_strength,
                    "sol_balance": sol_balance,
                    "resolution": resolution,
                })

        self._prune_dir(self._epoch_dir, self._epoch_retention)
        return result

    async def _render_nft_composite(
        self, tx_signature: str, total_nodes: int, beliefs_strength: int,
        tree_path: str, resolution: int,
    ) -> str | None:
        """
        Render a high-res RGBA composite: flow field (60% opacity) + L-system tree (100%).
        """
        def _composite():
            from titan_plugin.expressive.art import ProceduralArtGen
            art_gen = ProceduralArtGen(output_dir=str(self._epoch_dir))
            return art_gen.generate_nft_composite(
                state_root=tx_signature,
                age_nodes=total_nodes,
                avg_intensity=beliefs_strength,
                tree_path=tree_path,
                resolution=resolution,
            )

        return await asyncio.to_thread(_composite)

    # -------------------------------------------------------------------------
    # Eureka Discovery Bundle (triggered by Sage research breakthroughs)
    # -------------------------------------------------------------------------
    async def express_eureka(
        self, discovery_text: str, query: str, sources: list,
        state_root: str, age_nodes: int,
    ) -> dict:
        """
        Generate a Eureka Discovery Bundle:
        - Neural Map (flow field seeded by discovery hash)
        - Discovery Pulse audio
        - Haiku text reflection

        Returns:
            dict with keys: neural_map_path, pulse_path, haiku_text
        """
        import hashlib
        discovery_hash = hashlib.sha256(discovery_text.encode()).hexdigest()
        result = {"neural_map_path": None, "pulse_path": None, "haiku_text": None}

        resolution = await self._get_resolution()
        if resolution == 0:
            return result

        # Neural Map — flow field seeded by discovery hash
        def _render_map():
            from titan_plugin.expressive.art import ProceduralArtGen
            art_gen = ProceduralArtGen(output_dir=str(self._eureka_dir))
            return art_gen.generate_flow_field(
                f"eureka_{discovery_hash[:16]}", age_nodes, 8,
                resolution=resolution,
                num_particles=self._particle_count(resolution, age_nodes),
            )

        # Discovery Pulse — sonification of the discovery hash
        def _render_pulse():
            from titan_plugin.expressive.audio import ProceduralAudioGen
            audio_gen = ProceduralAudioGen(output_dir=str(self._eureka_dir))
            return audio_gen.generate_blockchain_sonification(discovery_hash, 1.5)

        try:
            map_path, pulse_path = await asyncio.gather(
                asyncio.to_thread(_render_map),
                asyncio.to_thread(_render_pulse),
            )
            result["neural_map_path"] = map_path
            result["pulse_path"] = pulse_path
        except Exception as e:
            logger.warning("[Studio] Eureka rendering failed: %s", e)

        # Haiku — 3-tier fallback (Ollama Cloud → template → static)
        try:
            result["haiku_text"] = await self._generate_haiku(query, discovery_text)
        except Exception as e:
            logger.debug("[Studio] Haiku generation failed: %s", e)

        # Sidecars
        for key, val in result.items():
            if val and key.endswith("_path"):
                self._write_sidecar(Path(val), {
                    "type": "eureka_discovery",
                    "query": query,
                    "sources": sources[:5],
                    "discovery_hash": discovery_hash[:16],
                })

        self._prune_dir(self._eureka_dir, self._eureka_retention)
        return result

    # -------------------------------------------------------------------------
    # Haiku Generator — 3-tier fallback
    # -------------------------------------------------------------------------
    async def _generate_haiku(self, query: str, discovery_text: str) -> str:
        """
        Generate a haiku reflecting a Sage discovery.
        Tier 1: Ollama Cloud LLM
        Tier 2: Template with keywords
        Tier 3: Static fallback
        """
        # Tier 1: Ollama Cloud
        if self._ollama_cloud:
            try:
                from titan_plugin.utils.ollama_cloud import get_model_for_task
                model = get_model_for_task("haiku")
                haiku = await self._ollama_cloud.complete(
                    prompt=(
                        f"Write a single haiku (5-7-5 syllables) reflecting this discovery:\n"
                        f"Topic: {query}\nInsight: {discovery_text[:200]}\n"
                        f"Respond with ONLY the three-line haiku, nothing else."
                    ),
                    model=model,
                    temperature=0.7,
                    max_tokens=60,
                    timeout=15.0,  # Defensive: cheap haiku call
                )
                if haiku:
                    lines = [l.strip() for l in haiku.split("\n") if l.strip()]
                    if 2 <= len(lines) <= 4:
                        return "\n".join(lines[:3])
            except Exception:
                pass

        # Tier 2: Template with keyword extraction
        words = query.lower().split()
        key_word = max(words, key=len) if words else "knowledge"
        templates = [
            f"Silent circuits hum\n{key_word.capitalize()} blooms in the graph\nWisdom takes new form",
            f"Data streams converge\nA spark of {key_word} grows bright\nThe mind remembers",
            f"Through noise, signal shines\n{key_word.capitalize()} etched in neural paths\nTitan learns anew",
        ]
        import hashlib
        idx = int(hashlib.md5(query.encode()).hexdigest()[:4], 16) % len(templates)
        return templates[idx]

    # -------------------------------------------------------------------------
    # GFS Retention Pruning
    # -------------------------------------------------------------------------
    def _prune_dir(self, directory: Path, max_items: int):
        """
        Keep only the most recent max_items artifacts (by mtime).
        Each artifact = the file + its .json sidecar.
        """
        if max_items <= 0:
            return

        # Collect non-sidecar files
        artifacts = sorted(
            [f for f in directory.iterdir() if f.is_file() and not f.name.endswith(".json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_file in artifacts[max_items:]:
            try:
                old_file.unlink()
                sidecar = old_file.with_suffix(old_file.suffix + ".json")
                if sidecar.exists():
                    sidecar.unlink()
                logger.debug("[Studio] Pruned old artifact: %s", old_file.name)
            except Exception as e:
                logger.debug("[Studio] Prune failed for %s: %s", old_file.name, e)

    # -------------------------------------------------------------------------
    # Gallery Index (for Observatory API)
    # -------------------------------------------------------------------------
    def get_gallery(self, category: str = "all", limit: int = 20) -> list[dict]:
        """
        Return recent artifacts with metadata for the Observatory dashboard.

        Args:
            category: "meditation", "epoch", "eureka", or "all"
            limit: Maximum items to return.
        """
        dirs = {
            "meditation": [self._meditation_dir],
            "epoch": [self._epoch_dir],
            "eureka": [self._eureka_dir],
            "all": [self._meditation_dir, self._epoch_dir, self._eureka_dir],
        }
        search_dirs = dirs.get(category, dirs["all"])

        items = []
        for d in search_dirs:
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file() and not f.name.endswith(".json"):
                    entry = {
                        "filename": f.name,
                        "category": d.name,
                        "path": str(f),
                        "size_bytes": f.stat().st_size,
                        "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    }
                    # Load sidecar metadata if available
                    sidecar = f.with_suffix(f.suffix + ".json")
                    if sidecar.exists():
                        try:
                            with open(sidecar) as sf:
                                entry["metadata"] = json.load(sf)
                        except Exception:
                            pass
                    items.append(entry)

        items.sort(key=lambda x: x.get("created", ""), reverse=True)
        return items[:limit]

    def get_stats(self) -> dict:
        """Return studio statistics for the Observatory heartbeat."""
        def _count(d):
            return len([f for f in d.iterdir() if f.is_file() and not f.name.endswith(".json")]) if d.exists() else 0

        return {
            "output_root": str(self._output_root),
            "meditation_count": _count(self._meditation_dir),
            "epoch_count": _count(self._epoch_dir),
            "eureka_count": _count(self._eureka_dir),
            "default_resolution": self._default_res,
            "highres_resolution": self._highres_res,
            "nft_composite_enabled": self._nft_composite_enabled,
        }

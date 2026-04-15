"""
art_generate helper — Self-directed art generation wrapper.

Enriches: Mind Touch[4] (mood boost), Spirit WHAT[2] (creative expression)

Key design: Intent carries Inner Trinity state, which gets translated to
visual parameters (warm colors when mood high, cool when contemplative, etc.)
"""
import logging
import os

logger = logging.getLogger(__name__)

# Same retention discipline as audio_generate: expression art is consumed by
# MediaWorker and fed to Mind[0] vision, so the archive has limited ongoing
# cognitive value. Bounded rolling window prevents the unbounded growth that
# filled T2/T3 disks to 100% on 2026-04-14 (22k+ root-level art files).
EXPRESSION_RETENTION = 150


class ArtGenerateHelper:
    """Art generation helper wrapping the expressive/art module."""

    def __init__(self, output_dir: str = "./data/art"):
        # Dedicated subdir so we don't collide with StudioCoordinator's
        # retention-managed meditation/epoch/eureka dirs.
        self._output_dir = os.path.join(output_dir, "expression_art")

    def _prune_retention(self) -> None:
        """Keep only the most recent EXPRESSION_RETENTION files in output_dir.
        Non-fatal on error — art generation must not fail because pruning did."""
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
            logger.debug("[ArtGenerate] prune_retention non-fatal error: %s", e)

    @property
    def name(self) -> str:
        return "art_generate"

    @property
    def description(self) -> str:
        return "Generate visual art expressing Titan's inner state"

    @property
    def capabilities(self) -> list[str]:
        return ["image_generation", "creative_expression"]

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
        Generate art based on Titan's inner state.

        Params:
            style: Art style hint — "flow_field" or "l_system" (default: "flow_field")
            inspiration: Text inspiration for the piece
            trinity_snapshot: dict with body/mind/spirit values (from impulse)
        """
        import hashlib
        import time

        trinity = params.get("trinity_snapshot", {})
        style = params.get("style") or self._select_style(trinity)

        # Derive art parameters from Trinity state
        body = trinity.get("body", [0.5] * 5)
        mind = trinity.get("mind", [0.5] * 5)
        spirit = trinity.get("spirit", [0.5] * 5)

        # state_root: hash of current time + trinity for unique seed
        seed_str = f"{time.time():.3f}_{sum(body):.3f}_{sum(mind):.3f}"
        state_root = hashlib.sha256(seed_str.encode()).hexdigest()[:16]

        # age_nodes: complexity from spirit magnitude (more conscious = richer art)
        spirit_mag = sum(spirit) / len(spirit) if spirit else 0.5
        age_nodes = max(10, int(spirit_mag * 100))

        # avg_intensity: emotion from mind[4] (touch/mood) mapped to 1-10
        mood_val = mind[4] if len(mind) > 4 else 0.5
        avg_intensity = max(1, min(10, int(mood_val * 10) + 1))

        try:
            from titan_plugin.expressive.art import ProceduralArtGen

            os.makedirs(self._output_dir, exist_ok=True)
            gen = ProceduralArtGen(output_dir=self._output_dir)

            if style == "l_system":
                result_path = gen.generate_l_system_tree(
                    tx_signature=state_root,
                    total_nodes=age_nodes,
                    beliefs_strength=avg_intensity,
                )
            elif style == "fractal":
                result_path = gen.generate_fractal(
                    state_root=state_root,
                    age_nodes=age_nodes,
                    avg_intensity=avg_intensity,
                )
            elif style == "cellular":
                result_path = gen.generate_cellular(
                    state_root=state_root,
                    age_nodes=age_nodes,
                    avg_intensity=avg_intensity,
                )
            elif style == "geometric":
                result_path = gen.generate_geometric(
                    state_root=state_root,
                    age_nodes=age_nodes,
                    avg_intensity=avg_intensity,
                )
            elif style == "noise_landscape":
                result_path = gen.generate_noise_landscape(
                    state_root=state_root,
                    age_nodes=age_nodes,
                    avg_intensity=avg_intensity,
                )
            else:
                result_path = gen.generate_flow_field(
                    state_root=state_root,
                    age_nodes=age_nodes,
                    avg_intensity=avg_intensity,
                )

            # Copy to media queue for Mind[0] vision enrichment
            if result_path and os.path.exists(result_path):
                # output_dir is now .../studio_exports/expression_art, so
                # dirname(dirname(...)) walks up to the project-root data dir
                # where media_queue lives.
                media_queue = os.path.join(
                    os.path.dirname(os.path.dirname(self._output_dir)),
                    "media_queue",
                )
                if os.path.isdir(media_queue):
                    import shutil
                    dest = os.path.join(media_queue, os.path.basename(result_path))
                    shutil.copy2(result_path, dest)

            # Bound disk use; non-fatal.
            self._prune_retention()

            result_text = (f"Created {style} artwork: {result_path} "
                          f"(nodes={age_nodes}, intensity={avg_intensity})")
            return {
                "success": True,
                "result": result_text,
                "file_path": result_path or "",
                "art_style": style,
                "enrichment_data": {"mind": [0, 4], "spirit": [2], "boost": 0.06},
                "error": None,
            }
        except ImportError:
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": "ProceduralArtGen module not available"}
        except Exception as e:
            logger.warning("[ArtGenerate] Generation failed: %s", e)
            return {"success": False, "result": "", "enrichment_data": {},
                    "error": str(e)}

    @staticmethod
    def _select_style(trinity: dict) -> str:
        """Select art style based on Trinity state — variety from consciousness.

        Mapping:
          High spirit → fractal (depth, complexity)
          High body entropy → cellular (emergent organic)
          High mind symmetry → geometric (structured harmony)
          Low intensity/calm → noise_landscape (peaceful terrain)
          High mood → flow_field (emotional waves)
          Default/random → l_system (botanical growth)
        """
        import random
        body = trinity.get("body", [0.5] * 5)
        mind = trinity.get("mind", [0.5] * 5)
        spirit = trinity.get("spirit", [0.5] * 5)

        spirit_mag = sum(spirit) / len(spirit) if spirit else 0.5
        body_entropy = body[3] if len(body) > 3 else 0.5
        mind_touch = mind[4] if len(mind) > 4 else 0.5

        # Weighted selection based on state
        weights = {
            "flow_field": 0.20 + mind_touch * 0.3,        # High mood → flow
            "l_system": 0.15,                               # Always available
            "fractal": 0.10 + spirit_mag * 0.3,            # High spirit → fractal
            "cellular": 0.10 + body_entropy * 0.3,         # High entropy → cellular
            "geometric": 0.10 + (1.0 - body_entropy) * 0.2,  # Low entropy → geometric
            "noise_landscape": 0.10 + (1.0 - mind_touch) * 0.2,  # Calm → landscape
        }

        styles = list(weights.keys())
        w = [weights[s] for s in styles]
        return random.choices(styles, weights=w, k=1)[0]

    def status(self) -> str:
        """Check if art generation is available."""
        try:
            from titan_plugin.expressive.art import ProceduralArtGen
            return "available"
        except ImportError:
            return "unavailable"

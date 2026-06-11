"""
expressive/art.py
Uses Pillow to generate procedural art seeded by epoch hashes.
V2.0: Resolution scaling, NFT composite rendering, RGBA support.
"""
import math
import hashlib
import logging
import os
import random
import time

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class ProceduralArtGen:
    """
    Generates abstract, procedurally generated artwork reflecting
    the agent's internal state using the Pillow library.
    """

    def __init__(self, output_dir="./art_exports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_flow_field(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        return_image: bool = False,
        resolution: int = 512, num_particles: int = 0,
        felt: dict | None = None,
    ):
        """
        Generate a Flow Field reflecting Titan's felt state.

        Args:
            state_root: Cryptographic seed string (deterministic art).
            age_nodes: Node count determining complexity.
            avg_intensity: Legacy emotion value (1-10) → color palette; used only
                when ``felt`` is None (the pre-states-era fallback).
            return_image: If True, return PIL Image instead of saving.
            resolution: Canvas size (width=height).
            num_particles: Override particle count (0 = auto-scale from age_nodes).
            felt: TRUE felt state ``{valence, arousal, neuromods, coherence}``
                (RFP §7.P7 / INV-SD-4). When present, his real affect drives the
                render — valence→hue, arousal→field energy, the neuromod profile→
                accent palette, coherence→order — replacing the 1-10 palette. The
                same ``(state_root, felt)`` always render identically.

        Returns:
            str | Image: Path to generated image, or PIL Image if return_image=True.
        """
        random.seed(state_root)
        width = height = resolution

        # ── seed → STRUCTURE (RFP §7.P7): the cumulative_hash drives a distinct
        # flow TOPOLOGY per entry — global rotation, anisotropic frequency warp,
        # and a set of swirl vortices — not merely a phase shift of one fixed grid
        # (the pre-fix renderer made every diary day the same spiral grid). sha256
        # gives independent 8-hex slices; the vortices come from the seeded RNG, so
        # the same (seed, felt) still renders identically (INV-SD-4).
        _sh = hashlib.sha256(state_root.encode()).hexdigest()
        seed_val = int(_sh[0:8], 16) / 100000.0                     # base flow-field phase
        field_rot = (int(_sh[8:16], 16) % 360) * math.pi / 180.0    # global flow rotation
        warp_x = 0.6 + (int(_sh[16:24], 16) % 1000) / 1000.0 * 1.3  # 0.6..1.9× x-frequency
        warp_y = 0.6 + (int(_sh[24:32], 16) % 1000) / 1000.0 * 1.3  # 0.6..1.9× y-frequency
        hue_rotation = float(int(_sh[32:40], 16) % 360)             # palette spread (neutral days)
        # Seed-driven DOMAIN WARP — displaces the (x,y) lattice with a turbulent
        # field BEFORE the flow, so the base pattern is no longer a regular grid
        # (this is what turns a low-arousal day from square spirals into organic
        # forms; without it the sin·cos base stays grid-shaped regardless of seed).
        warp_amp = 18.0 + (int(_sh[48:56], 16) % 1000) / 1000.0 * 40.0     # 18..58 px
        warp_freq = 0.004 + (int(_sh[56:64], 16) % 1000) / 1000.0 * 0.014  # 0.004..0.018
        n_vortices = 3 + (int(_sh[40:48], 16) % 5)                  # 3..7 swirl centers
        vortices = [
            (random.uniform(0, width), random.uniform(0, height),
             1.0 if random.random() < 0.5 else -1.0,                # spin direction
             random.uniform(1.3, 2.8))                              # swirl strength (dominant)
            for _ in range(n_vortices)
        ]

        # Auto-scaled base density/length (complexity from age_nodes).
        base_particles = num_particles if num_particles > 0 else min(5000, 100 + (age_nodes * 20))

        # ── render controls — felt-driven (P7) or the legacy 1-10 palette ──
        fp = None
        if felt is not None:
            from titan_hcl.expressive.felt_palette import (
                felt_to_render_params, normalize_felt,
            )
            canon = normalize_felt(felt)
            if canon is not None:
                fp = felt_to_render_params(canon, hue_rotation_deg=hue_rotation)
        if fp is not None:
            bg_color = fp["bg_color"]
            line_color_base = fp["line_color"]
            particles = min(5000, int(base_particles * fp["particle_mult"]))
            steps = fp["steps"]
            freq = fp["turbulence_freq"]
            turb_amp = fp["turbulence_amp"]
            order = fp["order"]
            jitter = fp["jitter"]
        else:
            # Legacy palette (felt unavailable) — byte-identical to the pre-P7 path.
            if avg_intensity >= 8:
                bg_color, line_color_base = (15, 5, 5), (200, 50, 50)
            elif avg_intensity <= 3:
                bg_color, line_color_base = (5, 10, 20), (50, 150, 200)
            else:
                bg_color, line_color_base = (10, 10, 15), (100, 200, 100)
            particles = base_particles
            steps = min(100, 20 + int(age_nodes / 2))
            freq, turb_amp, order, jitter = 0.01, 1.0, 1.0, 20

        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Scale step length with resolution
        step_len = max(1.0, resolution / 256.0)

        for _ in range(particles):
            x = random.randint(0, width)
            y = random.randint(0, height)

            for _ in range(steps):
                # domain warp first — bend the lattice so the base flow is organic,
                # not a square grid (seed-driven turbulent coordinate displacement).
                wx = x + warp_amp * math.sin(y * warp_freq + field_rot)
                wy = y + warp_amp * math.cos(x * warp_freq + seed_val)
                angle = (
                    math.sin(wx * freq * warp_x + seed_val)
                    * math.cos(wy * freq * warp_y + seed_val)
                    * math.pi * 2 * turb_amp
                ) + field_rot
                # seed-placed swirl vortices (dominant strength, wide radius) warp
                # the base field into a unique composition per entry.
                for cx, cy, spin, vstr in vortices:
                    dx, dy = x - cx, y - cy
                    angle += spin * vstr * math.atan2(dy, dx) * (
                        24000.0 / (dx * dx + dy * dy + 24000.0))
                if order < 1.0:    # coherence < 1 → fragment the flow (felt path)
                    angle += random.uniform(-1.0, 1.0) * (1.0 - order) * math.pi
                x_next = x + math.cos(angle) * step_len
                y_next = y + math.sin(angle) * step_len

                c = (
                    max(0, min(255, line_color_base[0] + random.randint(-jitter, jitter))),
                    max(0, min(255, line_color_base[1] + random.randint(-jitter, jitter))),
                    max(0, min(255, line_color_base[2] + random.randint(-jitter, jitter))),
                )
                draw.line([(x, y), (x_next, y_next)], fill=c, width=1)
                x, y = x_next, y_next

        if return_image:
            return img

        # rFP_meditation_worker_latency Fix #F (2026-05-07): include enough
        # of state_root + a timestamp suffix to guarantee per-cycle
        # uniqueness. Pre-fix used `state_root[:8]` which collided to the
        # literal "MEDITATI" prefix on every cycle (plugin passes
        # `f"MEDITATION_V3_E{epoch}"`), so all 556 entries on T2 and 704
        # on T3 pointed to the same file `flow_meditation_MEDITATI.jpg` —
        # each new cycle overwrote the previous render. Real Solana
        # signatures (used by the rebirth tree path) survive this fix
        # because their first 16 chars are unique hex; only the synthetic
        # MEDITATION_V3_* prefix needed disambiguation. Adding the epoch
        # timestamp suffix also ensures uniqueness even if two different
        # callers happened to pass the same state_root.
        import re as _re
        _safe_root = _re.sub(r"[^A-Za-z0-9_-]", "_", state_root)[:24]
        filepath = os.path.join(
            self.output_dir,
            f"flow_meditation_{_safe_root}_{int(time.time())}.jpg",
        )
        img.save(filepath, quality=90)
        logger.info("Generated emotional Flow Field: %s", filepath)
        return filepath

    def generate_l_system_tree(
        self, tx_signature: str, total_nodes: int, beliefs_strength: int,
        resolution: int = 512, return_image: bool = False,
    ):
        """
        Generate a Botanical L-System tree representing the daily Rebirth.

        Args:
            tx_signature: Cryptographic signature seeding the branching.
            total_nodes: Age factor scaling recursive iterations.
            beliefs_strength: Emotion factor for trunk thickness.
            resolution: Canvas size (width=height).
            return_image: If True, return PIL Image instead of saving.

        Returns:
            str | Image: Path to generated image, or PIL Image if return_image=True.
        """
        random.seed(tx_signature)
        width = height = resolution
        img = Image.new("RGB", (width, height), color=(10, 12, 10))
        draw = ImageDraw.Draw(img)

        scale = resolution / 512.0

        iterations = max(3, min(6, 1 + int(total_nodes / 50)))
        thickness = max(1, min(int(10 * scale), int(beliefs_strength / 20 * scale)))
        # Hash the seed first so any input works (a base58 pubkey is NOT hex —
        # int("…DihY", 16) raised ValueError during the genesis First Sight).
        angle_spread = 15 + (
            int(hashlib.md5(tx_signature.encode()).hexdigest()[-4:], 16) % 30)

        def draw_branch(x, y, angle, length, depth, current_thickness):
            if depth == 0:
                return

            x_end = x + math.cos(math.radians(angle)) * length
            y_end = y + math.sin(math.radians(angle)) * length

            color = (
                max(0, min(255, 50 + (depth * 20))),
                max(0, min(255, 150 + random.randint(-20, 50))),
                max(0, min(255, 50 + (depth * 10))),
            )

            draw.line([(x, y), (x_end, y_end)], fill=color, width=current_thickness)

            new_length = length * 0.75
            draw_branch(x_end, y_end, angle - angle_spread, new_length, depth - 1, max(1, current_thickness - 1))
            draw_branch(x_end, y_end, angle + angle_spread, new_length, depth - 1, max(1, current_thickness - 1))
            if random.random() > 0.5:
                draw_branch(x_end, y_end, angle, new_length * 0.6, depth - 1, max(1, current_thickness - 1))

        base_length = (100 + random.randint(-20, 20)) * scale
        draw_branch(width / 2, height - int(20 * scale), -90, base_length, iterations, max(1, thickness))

        if return_image:
            return img

        filepath = os.path.join(self.output_dir, f"tree_rebirth_{tx_signature[:8]}.jpg")
        img.save(filepath, quality=90)
        logger.info("Generated digital botanical tree: %s", filepath)
        return filepath

    def generate_nft_composite(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        tree_path: str, resolution: int = 2048,
    ) -> str:
        """
        Render a high-res RGBA composite for NFT minting:
        Flow field aura (60% opacity background) + L-system tree (100% foreground).

        Args:
            state_root: Seed for flow field generation.
            age_nodes: Node count for complexity.
            avg_intensity: Emotional intensity for color palette.
            tree_path: Path to pre-rendered L-system tree image.
            resolution: Target resolution (typically 2048).

        Returns:
            str: Path to the composite PNG image.
        """
        # Generate flow field as RGBA background layer
        random.seed(state_root)
        width = height = resolution

        if avg_intensity >= 8:
            bg_color = (15, 5, 5, 255)
            line_color_base = (200, 50, 50)
        elif avg_intensity <= 3:
            bg_color = (5, 10, 20, 255)
            line_color_base = (50, 150, 200)
        else:
            bg_color = (10, 10, 15, 255)
            line_color_base = (100, 200, 100)

        aura = Image.new("RGBA", (width, height), color=bg_color)
        draw = ImageDraw.Draw(aura)

        step_len = max(1.0, resolution / 256.0)
        particles = min(20000, 200 + (age_nodes * 40))
        steps = min(120, 30 + int(age_nodes / 2))
        seed_val = int(hashlib.md5(state_root.encode()).hexdigest()[:8], 16) / 100000.0

        for _ in range(particles):
            x = random.randint(0, width)
            y = random.randint(0, height)
            for _ in range(steps):
                angle = (
                    math.sin(x * 0.01 + seed_val)
                    * math.cos(y * 0.01 + seed_val)
                    * math.pi * 2
                )
                x_next = x + math.cos(angle) * step_len
                y_next = y + math.sin(angle) * step_len
                c = (
                    max(0, min(255, line_color_base[0] + random.randint(-20, 20))),
                    max(0, min(255, line_color_base[1] + random.randint(-20, 20))),
                    max(0, min(255, line_color_base[2] + random.randint(-20, 20))),
                    153,  # 60% opacity
                )
                draw.line([(x, y), (x_next, y_next)], fill=c, width=1)
                x, y = x_next, y_next

        # Load and resize tree to match resolution
        tree_img = Image.open(tree_path).convert("RGBA").resize(
            (resolution, resolution), Image.LANCZOS,
        )

        # Composite: aura background + tree foreground
        composite = Image.alpha_composite(aura, tree_img)

        filepath = os.path.join(self.output_dir, f"nft_composite_{state_root[:8]}.png")
        composite.save(filepath, format="PNG")
        logger.info("Generated NFT composite: %s", filepath)
        return filepath

    def generate_fractal(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        resolution: int = 512, return_image: bool = False,
    ):
        """
        Generate a Mandelbrot/Julia fractal reflecting consciousness depth.

        Body entropy → zoom level, Spirit magnitude → Julia constant.
        """
        random.seed(state_root)
        width = height = resolution
        seed_val = int(hashlib.md5(state_root.encode()).hexdigest()[:8], 16)

        # Julia constant from spirit magnitude (age_nodes)
        angle = (seed_val % 360) * math.pi / 180
        r = 0.3 + (age_nodes / 100.0) * 0.5
        c_real = r * math.cos(angle)
        c_imag = r * math.sin(angle)

        # Zoom from intensity
        zoom = 0.8 + (avg_intensity / 10.0) * 2.5
        max_iter = 80 + age_nodes

        img = Image.new("RGB", (width, height), (0, 0, 0))
        pixels = img.load()

        # Color palette from mood
        if avg_intensity >= 7:
            palette = [(i * 3 % 256, i * 1 % 200, 40) for i in range(max_iter)]
        elif avg_intensity <= 3:
            palette = [(20, i * 2 % 200, i * 3 % 256) for i in range(max_iter)]
        else:
            palette = [(i * 2 % 200, i * 3 % 256, i * 1 % 200) for i in range(max_iter)]

        for py in range(height):
            for px in range(width):
                zx = (px - width / 2) / (width / 2) / zoom
                zy = (py - height / 2) / (height / 2) / zoom
                i = 0
                while zx * zx + zy * zy < 4 and i < max_iter:
                    zx, zy = zx * zx - zy * zy + c_real, 2.0 * zx * zy + c_imag
                    i += 1
                if i < max_iter:
                    pixels[px, py] = palette[i % len(palette)]

        if return_image:
            return img

        filepath = os.path.join(self.output_dir, f"fractal_depth_{state_root[:8]}.jpg")
        img.save(filepath, quality=90)
        logger.info("Generated fractal: %s (c=%.3f+%.3fi, zoom=%.1f)", filepath, c_real, c_imag, zoom)
        return filepath

    def generate_cellular(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        resolution: int = 512, return_image: bool = False,
    ):
        """
        Generate cellular automata art — emergent organic patterns.

        Body entropy → rule number, steps → from age_nodes.
        Uses numpy for fast neighbor counting.
        """
        import numpy as np

        random.seed(state_root)
        width = height = resolution

        # 2D cellular automata on fixed 128x128 grid (fast, then upscale)
        grid_size = 128
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        # Seed: scattered random cells
        n_seeds = 10 + age_nodes
        for _ in range(n_seeds):
            grid[random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)] = 1

        # Conway-like rules with mood-based threshold
        birth = {3}
        survive = {2, 3}
        if avg_intensity >= 7:
            birth = {3, 6}
            survive = {2, 3, 4}
        elif avg_intensity <= 3:
            birth = {3}
            survive = {2, 3, 4, 5}

        generations = 20 + age_nodes // 2
        life_count = np.zeros((grid_size, grid_size), dtype=np.float32)

        for _gen in range(generations):
            # Count neighbors with numpy roll (toroidal boundary)
            neighbors = sum(
                np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            )
            # Apply rules
            new_grid = np.zeros_like(grid)
            for b in birth:
                new_grid |= ((grid == 0) & (neighbors == b)).astype(np.int8)
            for s in survive:
                new_grid |= ((grid == 1) & (neighbors == s)).astype(np.int8)
            life_count += new_grid.astype(np.float32)
            grid = new_grid

        # Render: color by life_count, upscale to resolution
        max_life = life_count.max() or 1.0
        t = life_count / max_life

        rgb = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        if avg_intensity >= 7:
            rgb[:, :, 0] = (200 * t).astype(np.uint8)
            rgb[:, :, 1] = (80 * t).astype(np.uint8)
            rgb[:, :, 2] = (30 * t).astype(np.uint8)
        elif avg_intensity <= 3:
            rgb[:, :, 0] = (30 * t).astype(np.uint8)
            rgb[:, :, 1] = (100 * t).astype(np.uint8)
            rgb[:, :, 2] = (200 * t).astype(np.uint8)
        else:
            rgb[:, :, 0] = (50 * t).astype(np.uint8)
            rgb[:, :, 1] = (200 * t).astype(np.uint8)
            rgb[:, :, 2] = (80 * t).astype(np.uint8)

        # Background for dead cells
        bg = np.array([5, 5, 10], dtype=np.uint8)
        mask = life_count == 0
        rgb[mask] = bg

        img = Image.fromarray(rgb).resize((width, height), Image.NEAREST)

        if return_image:
            return img

        filepath = os.path.join(self.output_dir, f"cellular_life_{state_root[:8]}.jpg")
        img.save(filepath, quality=90)
        logger.info("Generated cellular automata: %s (%d generations)", filepath, generations)
        return filepath

    def generate_geometric(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        resolution: int = 512, return_image: bool = False,
    ):
        """
        Generate sacred geometry — concentric circles, polygons, golden spirals.

        Mind symmetry → polygon sides, Spirit → layer count.
        """
        random.seed(state_root)
        width = height = resolution
        seed_val = int(hashlib.md5(state_root.encode()).hexdigest()[:8], 16)

        img = Image.new("RGB", (width, height), (8, 8, 15))
        draw = ImageDraw.Draw(img)

        cx, cy = width // 2, height // 2
        layers = 5 + age_nodes // 10
        max_radius = min(width, height) // 2 - 10

        # Polygon sides from seed (3=triangle, up to 12=dodecagon)
        sides = 3 + (seed_val % 10)

        for layer in range(layers):
            t = layer / max(1, layers - 1)
            radius = int(max_radius * (0.1 + t * 0.9))
            rotation = layer * (137.508 * math.pi / 180)  # Golden angle

            # Draw polygon at this layer
            points = []
            for i in range(sides):
                angle = rotation + (2 * math.pi * i / sides)
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                points.append((px, py))
            points.append(points[0])  # Close

            # Color gradient
            if avg_intensity >= 7:
                color = (int(180 * (1 - t)), int(100 * t), int(50 + 100 * t))
            elif avg_intensity <= 3:
                color = (int(50 * t), int(80 + 100 * t), int(180 * (1 - t)))
            else:
                color = (int(80 * t), int(180 * (1 - t)), int(100 + 80 * t))

            draw.line(points, fill=color, width=max(1, 2 - layer // 10))

            # Concentric circle at golden ratio
            circle_r = int(radius * 0.618)
            draw.ellipse(
                [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                outline=(*color, 100) if img.mode == "RGBA" else color,
                width=1)

        if return_image:
            return img

        filepath = os.path.join(self.output_dir, f"geometry_sacred_{state_root[:8]}.jpg")
        img.save(filepath, quality=90)
        logger.info("Generated sacred geometry: %s (%d-gon, %d layers)", filepath, sides, layers)
        return filepath

    def generate_noise_landscape(
        self, state_root: str, age_nodes: int, avg_intensity: int,
        resolution: int = 512, return_image: bool = False,
    ):
        """
        Generate Perlin-like noise landscape — terrain/cloud textures.

        Uses octave noise for natural-looking gradients.
        """
        random.seed(state_root)
        width = height = resolution
        seed_val = int(hashlib.md5(state_root.encode()).hexdigest()[:8], 16) / 100000.0

        img = Image.new("RGB", (width, height))
        pixels = img.load()

        # Octave noise parameters from state
        octaves = 3 + min(4, age_nodes // 20)
        persistence = 0.4 + (avg_intensity / 10.0) * 0.3

        for py in range(height):
            for px in range(width):
                val = 0.0
                amplitude = 1.0
                frequency = 0.005
                for _oct in range(octaves):
                    # Simple value noise via sin/cos combination
                    nx = px * frequency + seed_val
                    ny = py * frequency + seed_val * 0.7
                    noise = (math.sin(nx * 1.3 + ny * 0.9) *
                             math.cos(nx * 0.7 + ny * 1.1 + seed_val) *
                             math.sin(nx * 0.4 + ny * 1.7 + seed_val * 2))
                    val += noise * amplitude
                    amplitude *= persistence
                    frequency *= 2.0

                # Normalize to 0-1
                val = (val + 1.0) / 2.0
                val = max(0.0, min(1.0, val))

                # Color mapping
                if avg_intensity >= 7:
                    r = int(200 * val + 40 * (1 - val))
                    g = int(80 * val + 20 * (1 - val))
                    b = int(30 * val)
                elif avg_intensity <= 3:
                    r = int(20 * val)
                    g = int(60 * val + 40 * (1 - val))
                    b = int(180 * val + 40 * (1 - val))
                else:
                    r = int(40 * val + 20)
                    g = int(150 * val + 30)
                    b = int(80 * val + 40 * (1 - val))

                pixels[px, py] = (r, g, b)

        if return_image:
            return img

        filepath = os.path.join(self.output_dir, f"landscape_noise_{state_root[:8]}.jpg")
        img.save(filepath, quality=90)
        logger.info("Generated noise landscape: %s (%d octaves)", filepath, octaves)
        return filepath

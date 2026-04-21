"""EMOT-CGN Latent Bundle — shm protocol for emergent emotion (rFP §19).

The BUNDLE is the v3 substrate. Replaces the v2 "dominant_idx among 8
primitives" pattern with Titan's whole hierarchical state.

Philosophy
----------
Emotion is not a classification of a felt tensor into 8 labels. It is
the qualitative-felt projection of Titan's whole consciousness at an
instant — native Trinity dimensions carried losslessly. HDBSCAN over
the bundle's rolling trajectory discovers dense regions; regions earn
names (or NOISE) from density, not from hand-seeded primitives.

Native-first dimensionality (rFP §19)
-------------------------------------
Bundle primary representation = Titan's actual dims, no bottleneck:

  130D  felt tensor          (Inner Trinity 5+15+45 + Outer Trinity 5+15+45)
    2D  trajectory           (T3 + T4 learned-attention + meta-state)
   30D  space topology       (L5 §8.2 cross-learner context)
    6D  neuromod state       (DA, 5HT, NE, ACh, Endorphin, GABA)
  ───   ─────────────────
  168D  native consciousness (lossless — nothing thrown away)

Plus side channels (hormones 11D, NS urgencies 11D, CGN β-states 8D,
MSL activations 6D, π-phase 4D) and derived emergent-emotion fields
(region_id, valence, arousal, novelty, residence, signature).

L5 forward compatibility
------------------------
The bundle reserves 3×32D slots for L5's trained AIF trunk/inner/outer
encoder (rFP_titan_unified_learning_v1.md §5.1). These are ZERO until
L5 Phase 0 ships (~4-6 weeks); once L5 lands, it starts populating
them in a single commit — no schema bump needed. Consumers that want
the L5-compressed view read z_trunk/z_inner/z_outer; consumers that
want Titan's full state read felt_tensor + trajectory + space_topology.

The `encoder_id` field announces which populated (0 = thin assembly
today, 1 = L5 Phase 0, 2 = L5 Phase 5, …).

Backward compatibility (rFP §22)
---------------------------------
Bundle always populates `legacy_idx` — nearest of the 8 original
primitives (FLOW, IMPASSE_TENSION, RESOLUTION, PEACE, CURIOSITY,
GRIEF, WONDER, LOVE) to the current state. Old `emot_state.bin`
continues to be written alongside the bundle with
`dominant_idx = legacy_idx`. Every current reader (narrator / social /
dashboard / meta_reasoning) keeps working unchanged during v3
transition. v3-aware readers use `region_id` directly.

File: /dev/shm/titan_<ID>/emot_latent_bundle.bin (per-Titan path).

Layout (2048 bytes total, fixed):
  HEADER (32 B):
    u32 version_head      — torn-read protection
    u64 ts_ms             — write timestamp
    u32 encoder_id        — 0=thin assembly, 1=l5_phase0, 2=l5_phase5
    u32 schema_version    — bump only on breaking layout change
    u32 titan_id_hash     — FNV-1a("T1"/"T2"/"T3") cross-Titan guard
    u64 reserved_u64

  NATIVE CONSCIOUSNESS (672 B):
    f32[130] felt_tensor      — Inner Trinity 65D + Outer Trinity 65D
    f32[2]   trajectory       — 2D journey
    f32[30]  space_topology   — 30D space/chi context
    f32[6]   neuromod_state   — DA, 5HT, NE, ACh, Endorphin, GABA

  SIDE CHANNELS (168 B):
    f32[11]  hormone_levels   — 11 NS programs' hormonal accumulators
    f32[11]  ns_urgencies     — 11 NS programs' urgency (5 inner + 6 outer)
    f32[8]   cgn_beta_states  — per-CGN-consumer dominant V
    f32[6]   msl_activations  — I, YOU, ME, WE, YES, NO
    f32[6]   pi_phase         — sphere-clock phases for all 6 clocks:
                                inner_body, outer_body, inner_mind,
                                outer_mind, inner_spirit, outer_spirit
                                (Trinity × Inner/Outer; schema v2 preserves
                                Trinity symmetry — v1 truncated to 4)

  L5 RESERVED LATENTS (384 B):
    f32[32]  z_trunk          — L5 shared trunk (0 until L5 Phase 0)
    f32[32]  z_inner          — L5 autonomic (0 until L5 Phase 0)
    f32[32]  z_outer          — L5 conscious (0 until L5 Phase 0)

  DERIVED (36 B):
    i32 region_id             — HDBSCAN label (>=0), -1=NOISE, -2=unclustered
    u8  legacy_idx            — nearest of 8 legacy primitives (0..7)
    u8  graduation_status     — 0=shadow, 1=observing, 2=graduated
    u8  regions_emerged       — count of stable regions
    u8  reserved_u8
    f32 valence               — Russell affect circumplex [-1, +1]
    f32 arousal               — Russell affect circumplex [-1, +1]
    f32 novelty               — inverse-density at current state [0, 1]
    f32 region_confidence     — HDBSCAN cluster probability [0, 1]
    f32 region_residence_s    — seconds since entering current region
    u64 region_signature      — stable region identity across re-cluster

  TRAILER (8 B):
    u32 version_trailer       — must match header version
    u32 reserved_u32

  RESERVED TAIL (756 B):
    — future schema additions (new side channels, new derived fields,
      secondary L5 variants) without breaking existing layout.

Core: 32 + 672 + 168 + 384 + 36 + 8 = 1300 bytes (schema v2)
Total: 2048 bytes (748 reserved tail).

Schema history:
  v1 (2026-04-21): initial shipping schema, pi_phase 4D (truncated).
  v2 (2026-04-21): pi_phase expanded to 6D — all sphere-clock phases
                   (Trinity × Inner/Outer). BundleReader returns None on
                   v1 bundles; legacy emot_state.bin unchanged.
"""
from __future__ import annotations

import logging
import mmap
import os
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Encoder IDs (fixed constants — never renumber; add new IDs for new encoders).
ENCODER_THIN_ASSEMBLY = 0  # ships today — pure assembly of native state, no projection
ENCODER_L5_PHASE0 = 1      # L5 AIF NS-rewrite trunk/inner/outer (~4-6 weeks)
ENCODER_L5_PHASE5 = 2      # L5 Phase 5 — META-CGN-as-AIF (future)
# Back-compat alias (deprecated — kept briefly while call sites migrate).
ENCODER_THIN_PCA = ENCODER_THIN_ASSEMBLY

# Graduation status codes.
GRAD_SHADOW = 0
GRAD_OBSERVING = 1
GRAD_GRADUATED = 2

# Region ID sentinels.
REGION_NOISE = -1          # HDBSCAN labeled as noise (not dense enough)
REGION_UNCLUSTERED = -2    # Not yet clustered (too few observations)

BUNDLE_SCHEMA_VERSION = 2

# Native-dim constants — trace to Titan's actual architecture.
FELT_TENSOR_DIM = 130
INNER_TRINITY_DIM = 65     # IB 5 + IM 15 + IS 45
OUTER_TRINITY_DIM = 65     # OB 5 + OM 15 + OS 45
TRAJECTORY_DIM = 2         # T3 + T4 learned-attention + meta-state
SPACE_TOPOLOGY_DIM = 30    # L5 §8.2 cross-learner context
NEUROMOD_DIM = 6           # DA, 5HT, NE, ACh, Endorphin, GABA

# L5 reserved compressed latent dims (per rFP_titan_unified_learning_v1.md §5.1).
TRUNK_DIM = 32
INNER_DIM = 32
OUTER_DIM = 32

# Side-channel dims.
HORMONE_DIM = 11
NS_URGENCY_DIM = 11
CGN_BETA_DIM = 8
MSL_ACT_DIM = 6
PI_PHASE_DIM = 6         # sphere_clock.get_all_phases() returns 6 clocks

# NS program names — order matches hormone_levels + ns_urgencies layout.
NS_PROGRAMS = [
    "REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM",  # 5 inner
    "CREATIVITY", "CURIOSITY", "EMPATHY", "REFLECTION",
    "INSPIRATION", "VIGILANCE",  # 6 outer
]
NS_PROGRAM_INDEX = {p: i for i, p in enumerate(NS_PROGRAMS)}

# CGN consumer names — order matches cgn_beta_states layout.
CGN_CONSUMERS = [
    "language", "social", "knowledge", "reasoning",
    "coding", "self_model", "reasoning_strategy", "meta",
]
CGN_CONSUMER_INDEX = {c: i for i, c in enumerate(CGN_CONSUMERS)}

# MSL concepts — order matches msl_activations layout.
MSL_CONCEPTS = ["I", "YOU", "ME", "WE", "YES", "NO"]
MSL_CONCEPT_INDEX = {c: i for i, c in enumerate(MSL_CONCEPTS)}

# Legacy primitives — preserved from v2 for backward compat.
# DO NOT reorder — legacy_idx → name mapping must stay stable for all
# consumers that read state.bin's dominant_idx.
LEGACY_PRIMITIVES = [
    "FLOW", "IMPASSE_TENSION", "RESOLUTION", "PEACE",
    "CURIOSITY", "GRIEF", "WONDER", "LOVE",
]

# Struct layouts (little-endian, explicit sizes).
_HEADER = struct.Struct("<I Q I I I Q")  # 4+8+4+4+4+8 = 32
_NATIVE = struct.Struct(
    f"<{FELT_TENSOR_DIM + TRAJECTORY_DIM + SPACE_TOPOLOGY_DIM + NEUROMOD_DIM}f"
)  # 130+2+30+6 = 168 floats = 672 B
_SIDE = struct.Struct(
    f"<{HORMONE_DIM + NS_URGENCY_DIM + CGN_BETA_DIM + MSL_ACT_DIM + PI_PHASE_DIM}f"
)  # 11+11+8+6+4 = 40 floats = 160 B
_L5 = struct.Struct(f"<{TRUNK_DIM + INNER_DIM + OUTER_DIM}f")  # 96 f = 384 B
_DERIVED = struct.Struct("<i B B B B f f f f f Q")  # 4+4+20+8 = 36 B
_TRAILER = struct.Struct("<I I")  # 8 B

HEADER_SIZE = _HEADER.size
NATIVE_SIZE = _NATIVE.size
SIDE_SIZE = _SIDE.size
L5_SIZE = _L5.size
DERIVED_SIZE = _DERIVED.size
TRAILER_SIZE = _TRAILER.size

CORE_SIZE = HEADER_SIZE + NATIVE_SIZE + SIDE_SIZE + L5_SIZE + DERIVED_SIZE + TRAILER_SIZE
# 32 + 672 + 160 + 384 + 36 + 8 = 1292
BUNDLE_SIZE = 2048
RESERVED_TAIL = BUNDLE_SIZE - CORE_SIZE  # 756 B


def _fnv1a_32(s: str) -> int:
    """Tiny FNV-1a hash (32-bit) for titan_id sanity check."""
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _resolve_titan_id(explicit: Optional[str] = None) -> str:
    """Resolve current Titan's ID (T1/T2/T3)."""
    if explicit:
        return str(explicit)
    try:
        import json
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        identity_path = os.path.join(proj_root, "data", "titan_identity.json")
        if os.path.exists(identity_path):
            with open(identity_path) as f:
                tid = json.load(f).get("titan_id")
            if tid:
                return str(tid)
    except Exception:
        pass
    env_id = os.environ.get("TITAN_ID")
    return str(env_id) if env_id else "T1"


def default_bundle_path(titan_id: Optional[str] = None) -> str:
    return f"/dev/shm/titan_{_resolve_titan_id(titan_id)}/emot_latent_bundle.bin"


DEFAULT_BUNDLE_PATH = default_bundle_path()


def _ensure_shm_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


class BundleWriter:
    """Writes emot_latent_bundle.bin. Single-writer (emot_cgn_worker).

    Version-counter pattern (header + trailer) gives readers torn-read
    protection without locking. All scalar inputs are clipped/coerced
    defensively; writer never raises into the hot path.
    """

    def __init__(self, path: str = DEFAULT_BUNDLE_PATH,
                 titan_id: Optional[str] = None):
        self._path = path
        self._titan_id = _resolve_titan_id(titan_id)
        self._titan_id_hash = _fnv1a_32(self._titan_id)
        self._version = 0
        _ensure_shm_dir(path)
        if not os.path.exists(path) or os.path.getsize(path) != BUNDLE_SIZE:
            with open(path, "wb") as f:
                f.write(b"\x00" * BUNDLE_SIZE)

    def write(self, *,
              encoder_id: int,
              # Native consciousness (primary, lossless):
              felt_tensor_130d=None,
              trajectory_2d=None,
              space_topology_30d=None,
              neuromod_state_6d=None,
              # Side channels:
              hormone_levels_11d=None,
              ns_urgencies_11d=None,
              cgn_beta_states_8d=None,
              msl_activations_6d=None,
              pi_phase_6d=None,
              # L5 reserved (optional — default zeros):
              z_trunk_32d=None,
              z_inner_32d=None,
              z_outer_32d=None,
              # Derived:
              region_id: int = REGION_UNCLUSTERED,
              legacy_idx: int = 0,
              graduation_status: int = GRAD_SHADOW,
              regions_emerged: int = 0,
              valence: float = 0.0,
              arousal: float = 0.0,
              novelty: float = 0.5,
              region_confidence: float = 0.0,
              region_residence_s: float = 0.0,
              region_signature: int = 0) -> int:
        """Write the full bundle. Returns new version number.

        Inputs that are lists/arrays are truncated/padded to expected
        dimension. Never raises into hot path — returns last-known-good
        version on error.
        """
        try:
            self._version += 1
            ts_ms = int(time.time() * 1000)

            def _vec(x, n: int):
                if x is None:
                    return [0.0] * n
                out = [float(v) for v in list(x)[:n]]
                if len(out) < n:
                    out.extend([0.0] * (n - len(out)))
                return out

            header = _HEADER.pack(
                self._version, ts_ms,
                int(encoder_id) & 0xFFFFFFFF,
                BUNDLE_SCHEMA_VERSION,
                self._titan_id_hash,
                0,
            )
            native_flat = (
                _vec(felt_tensor_130d, FELT_TENSOR_DIM)
                + _vec(trajectory_2d, TRAJECTORY_DIM)
                + _vec(space_topology_30d, SPACE_TOPOLOGY_DIM)
                + _vec(neuromod_state_6d, NEUROMOD_DIM)
            )
            native = _NATIVE.pack(*native_flat)
            side_flat = (
                _vec(hormone_levels_11d, HORMONE_DIM)
                + _vec(ns_urgencies_11d, NS_URGENCY_DIM)
                + _vec(cgn_beta_states_8d, CGN_BETA_DIM)
                + _vec(msl_activations_6d, MSL_ACT_DIM)
                + _vec(pi_phase_6d, PI_PHASE_DIM)
            )
            side = _SIDE.pack(*side_flat)
            l5_flat = (
                _vec(z_trunk_32d, TRUNK_DIM)
                + _vec(z_inner_32d, INNER_DIM)
                + _vec(z_outer_32d, OUTER_DIM)
            )
            l5 = _L5.pack(*l5_flat)
            derived = _DERIVED.pack(
                int(region_id),
                max(0, min(7, int(legacy_idx))),
                max(0, min(2, int(graduation_status))),
                max(0, min(255, int(regions_emerged))),
                0,
                float(max(-1.0, min(1.0, valence))),
                float(max(-1.0, min(1.0, arousal))),
                float(max(0.0, min(1.0, novelty))),
                float(max(0.0, min(1.0, region_confidence))),
                float(max(0.0, region_residence_s)),
                int(region_signature) & 0xFFFFFFFFFFFFFFFF,
            )
            trailer = _TRAILER.pack(self._version, 0)
            tail = b"\x00" * RESERVED_TAIL
            payload = header + native + side + l5 + derived + trailer + tail
            with open(self._path, "r+b") as f:
                f.write(payload)
            return self._version
        except Exception as e:
            logger.warning("[BundleWriter] write failed: %s", e)
            return self._version


class BundleReader:
    """Zero-copy mmap reader for the latent bundle. Multi-reader safe.

    Returns None on any failure (file missing, torn read, Titan-ID
    mismatch, schema mismatch) so callers fall back to legacy paths.
    """

    def __init__(self, path: str = DEFAULT_BUNDLE_PATH,
                 expected_titan_id: Optional[str] = None):
        self._path = path
        self._last_version = 0
        self._expected_hash = (
            _fnv1a_32(expected_titan_id) if expected_titan_id else None
        )

    def _read_bytes(self):
        try:
            if not os.path.exists(self._path):
                return None
            if os.path.getsize(self._path) < BUNDLE_SIZE:
                return None
            with open(self._path, "rb") as f:
                with mmap.mmap(f.fileno(), BUNDLE_SIZE,
                               prot=mmap.PROT_READ) as mm:
                    return bytes(mm)
        except Exception:
            return None

    def read(self) -> Optional[dict]:
        data = self._read_bytes()
        if data is None:
            return None
        try:
            off = 0
            ver_h, ts_ms, enc_id, schema_v, tid_hash, _ = _HEADER.unpack(
                data[off:off + HEADER_SIZE])
            off += HEADER_SIZE
            if schema_v != BUNDLE_SCHEMA_VERSION:
                return None
            if self._expected_hash is not None and tid_hash != self._expected_hash:
                return None

            native = _NATIVE.unpack(data[off:off + NATIVE_SIZE])
            off += NATIVE_SIZE
            side = _SIDE.unpack(data[off:off + SIDE_SIZE])
            off += SIDE_SIZE
            l5 = _L5.unpack(data[off:off + L5_SIZE])
            off += L5_SIZE
            (region_id, legacy_idx, grad_status, regions_emerged, _r8,
             valence, arousal, novelty, region_conf, region_res_s,
             region_sig) = _DERIVED.unpack(data[off:off + DERIVED_SIZE])
            off += DERIVED_SIZE
            ver_t, _r32 = _TRAILER.unpack(data[off:off + TRAILER_SIZE])
            if ver_h != ver_t:
                return None  # torn read
            self._last_version = ver_h

            n_off = 0
            felt = list(native[n_off:n_off + FELT_TENSOR_DIM])
            n_off += FELT_TENSOR_DIM
            traj = list(native[n_off:n_off + TRAJECTORY_DIM])
            n_off += TRAJECTORY_DIM
            space = list(native[n_off:n_off + SPACE_TOPOLOGY_DIM])
            n_off += SPACE_TOPOLOGY_DIM
            neuromod = list(native[n_off:n_off + NEUROMOD_DIM])

            s_off = 0
            hormones = list(side[s_off:s_off + HORMONE_DIM]);      s_off += HORMONE_DIM
            ns_urg = list(side[s_off:s_off + NS_URGENCY_DIM]);     s_off += NS_URGENCY_DIM
            cgn_b = list(side[s_off:s_off + CGN_BETA_DIM]);        s_off += CGN_BETA_DIM
            msl_act = list(side[s_off:s_off + MSL_ACT_DIM]);       s_off += MSL_ACT_DIM
            pi_ph = list(side[s_off:s_off + PI_PHASE_DIM])

            return {
                "version": ver_h,
                "ts_ms": ts_ms,
                "encoder_id": enc_id,
                "schema_version": schema_v,
                # Native consciousness (primary — 168D):
                "felt_tensor": felt,
                "trajectory": traj,
                "space_topology": space,
                "neuromod_state": neuromod,
                # Side channels:
                "hormone_levels": hormones,
                "ns_urgencies": ns_urg,
                "cgn_beta_states": cgn_b,
                "msl_activations": msl_act,
                "pi_phase": pi_ph,
                # L5 reserved (zero until Phase 0):
                "z_trunk": list(l5[:TRUNK_DIM]),
                "z_inner": list(l5[TRUNK_DIM:TRUNK_DIM + INNER_DIM]),
                "z_outer": list(l5[TRUNK_DIM + INNER_DIM:
                                   TRUNK_DIM + INNER_DIM + OUTER_DIM]),
                # Derived:
                "region_id": region_id,
                "legacy_idx": legacy_idx,
                "legacy_label": (LEGACY_PRIMITIVES[legacy_idx]
                                 if 0 <= legacy_idx < len(LEGACY_PRIMITIVES)
                                 else "UNKNOWN"),
                "graduation_status": grad_status,
                "regions_emerged": regions_emerged,
                "valence": valence,
                "arousal": arousal,
                "novelty": novelty,
                "region_confidence": region_conf,
                "region_residence_s": region_res_s,
                "region_signature": region_sig,
            }
        except Exception:
            return None

    def has_new(self) -> bool:
        """Cheap version-counter poll."""
        try:
            if not os.path.exists(self._path):
                return False
            with open(self._path, "rb") as f:
                ver_bytes = f.read(4)
            if len(ver_bytes) < 4:
                return False
            return struct.unpack("<I", ver_bytes)[0] > self._last_version
        except Exception:
            return False


# ── Plug A helpers (β-context integration for 7 CGN consumers) ─────
#
# Per rFP §20 Plug A: each non-emotional CGN consumer reserves one
# slot in its 30D state vector for an emotion signal from the bundle.
# We remap valence ∈ [-1,+1] → [0,1] so the natural default of
# "unavailable/neutral" is 0.5 — matching the old `vec[17] = 0.5`
# placeholder, i.e. zero distributional shock to SharedValueNet when
# the bundle isn't yet populated. Once bundle writes begin, valence
# shifts the slot above/below 0.5 naturally.

_plug_a_reader_cache: "Optional[BundleReader]" = None
_plug_c_reader_cache: "Optional[BundleReader]" = None


def read_full_emotion_context(
    titan_id: Optional[str] = None,
    reader: "Optional[BundleReader]" = None,
) -> Optional[dict]:
    """Plug C (rFP §20): rich emotion-context read for consumers that
    want more than just the valence slot of Plug A.

    Returns a trimmed dict with the fields most useful for behaviour
    modulation (valence, arousal, novelty, region_id, legacy_label,
    graduation_status, region_residence_s). Full bundle is available
    via BundleReader().read() for power-use cases.

    Returns None if bundle unavailable or Titan-ID mismatch. Callers
    always gate on `None` → fall back to their legacy path. Cached
    reader — no instantiation overhead per call.
    """
    global _plug_c_reader_cache
    try:
        r = reader
        if r is None:
            if _plug_c_reader_cache is None:
                _plug_c_reader_cache = BundleReader(
                    path=default_bundle_path(titan_id),
                    expected_titan_id=titan_id,
                )
            r = _plug_c_reader_cache
        d = r.read()
        if d is None:
            return None
        return {
            "valence": d.get("valence", 0.0),
            "arousal": d.get("arousal", 0.0),
            "novelty": d.get("novelty", 0.5),
            "region_id": d.get("region_id", REGION_UNCLUSTERED),
            "region_confidence": d.get("region_confidence", 0.0),
            "region_residence_s": d.get("region_residence_s", 0.0),
            "region_signature": d.get("region_signature", 0),
            "regions_emerged": d.get("regions_emerged", 0),
            "legacy_idx": d.get("legacy_idx", 0),
            "legacy_label": d.get("legacy_label", "UNKNOWN"),
            "graduation_status": d.get("graduation_status", 0),
            "encoder_id": d.get("encoder_id", 0),
            "version": d.get("version", 0),
            "ts_ms": d.get("ts_ms", 0),
        }
    except Exception:
        return None


def read_emotion_valence_normalized(
    titan_id: Optional[str] = None,
    reader: "Optional[BundleReader]" = None,
    default: float = 0.5,
) -> float:
    """Return emotion valence remapped to [0, 1] for CGN β-context.

    0.5  = neutral / bundle unavailable (default — matches pre-v3 placeholder)
    < 0.5 = negative valence (grief-like, low-reward trajectory EMA)
    > 0.5 = positive valence (flow-like, high-reward trajectory EMA)

    Uses a module-level cached BundleReader unless the caller supplies
    its own. Reader is cheap (mmap); no rate limit needed. Failures
    silently return `default` — consumers never see exceptions.
    """
    global _plug_a_reader_cache
    try:
        r = reader
        if r is None:
            if _plug_a_reader_cache is None:
                _plug_a_reader_cache = BundleReader(
                    path=default_bundle_path(titan_id),
                    expected_titan_id=titan_id,
                )
            r = _plug_a_reader_cache
        if not r.has_new() and _plug_a_reader_cache is r:
            # Small optimization: reread if we have a fresh version; otherwise
            # fall through to read anyway (initial call hits has_new=False
            # since _last_version starts at 0; read() populates it).
            pass
        d = r.read()
        if d is None:
            return float(default)
        # Bundle field is [-1, +1]; remap to [0, 1].
        v = float(d.get("valence", 0.0))
        return float(max(0.0, min(1.0, 0.5 + 0.5 * v)))
    except Exception:
        return float(default)

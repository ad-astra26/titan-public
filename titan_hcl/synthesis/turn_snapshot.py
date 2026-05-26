"""Chat-turn felt-state snapshot + tool-calls extraction — arch §7 /
rFP §8.2 content fields.

Phase 3 (rFP §18 Phase 3 — episode model). Captures the §7 fields whose
source is Titan's *current* state at chat-turn time, plus normalized
tool-call records from agno's RunOutput:

  - `neuromods`      — {name: level} for the 6 modulators (read_neuromod)
  - `embedding_hash` — sha256 hex of the 132D unified-spirit vector
                       (trinity full_130dt[0:130] + journey[160:162])
                       per `_phase_c_constants.UNIFIED_SPIRIT_132D_SCHEMA_VERSION`
  - `importance`     — cold-start default 0.5 per arch §5.3 / rFP §20 Q2
                       (lazy-scored in next dream cycle from bridge salience)
  - `tool_calls`     — normalized list from `run_output.tools` (agno
                       ToolExecution dataclass), shape per §8.2:
                       `{tool, args_hash, result_hash, latency_ms,
                       exception}`. Args + result are HASHED (not stored
                       inline) for chat-TX leanness — full bodies live
                       in the §8 procedural-fork TX (Phase 8) or via
                       per-tool oracle anchors. Inline hashes give §10
                       cross-ref via Kuzu `Production` edges (Phase 4).

This module runs in the writer process (agno PostHook) so the §7 content
travels with the TX at write time — no cross-process sync (G19) needed.

All readers are SHM-watermark-gated and soft-fail to empty/zero values;
**no exception is ever raised to the caller**. OVG.build_timechain_payload
must never raise (Phase 2 closure D-P2-closure-defensive) so this module
keeps the same contract: a partially-failing trinity read still produces
a valid (degraded) snapshot.

Surfaced concerns from the Phase 3 plan (to revisit after live soak):
  1. neuromods shape — currently full 6-modulator dict; spec is silent on
     compaction. Easy to switch to a 6-float array later if observability
     shows the dict overhead is meaningful.
  2. embedding_hash binds the trinity at PostHook time, NOT the LLM call
     time. Sub-second drift is typically negligible (LLM call is the
     long-pole, ~200ms–2s). Revisit if soak shows the snapshot is
     systematically misaligned with the felt-state at response generation.
  3. tool_calls inline-hashing vs raw args — currently hashed (PII-safe
     + lean). Phase 8 procedural-fork TX is the canonical store for raw
     args/result. If retrieval needs args-grep, surface a tools_args
     index in Phase 4 rather than fattening every chat TX.
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Default importance for a fresh chat-turn TX (arch §5.3 cold-start).
DEFAULT_IMPORTANCE: float = 0.5

# Width of the 132D unified-spirit vector per
# `_phase_c_constants.UNIFIED_SPIRIT_132D_SCHEMA_VERSION` (130 trinity
# + 2 journey). Used to detect a partial/short read.
UNIFIED_SPIRIT_DIM: int = 132


def _hash_132d(values: list[float]) -> str:
    """sha256 hex of the 132 floats serialized as little-endian float32.

    Deterministic across runs + machines (struct '<132f' is platform-
    independent). Returns empty string when input is wrong shape, so the
    caller can distinguish "no snapshot" from a valid-but-zero hash.
    """
    if len(values) != UNIFIED_SPIRIT_DIM:
        return ""
    try:
        buf = struct.pack(f"<{UNIFIED_SPIRIT_DIM}f", *(float(v) for v in values))
    except (struct.error, TypeError, ValueError):
        return ""
    return hashlib.sha256(buf).hexdigest()


def capture_turn_snapshot(reader_bank: Any = None) -> dict[str, Any]:
    """Build the arch §7 felt-state portion of a chat-turn TX content.

    Args:
        reader_bank: ShmReaderBank instance (or anything with
            `read_neuromod()` + `read_trinity()` returning the
            documented shapes). When None, a fresh ShmReaderBank is
            constructed — the same lazy-init pattern the agno_hooks
            RecallBridge already uses.

    Returns a dict with stable keys (always present, may carry
    empty/zero values on SHM unavailability):

        {
            "neuromods":      {name: level} for 6 modulators (empty {} on failure),
            "embedding_hash": 64-char sha256 hex (empty "" on failure),
            "importance":     0.5 default (arch §5.3 cold-start),
        }

    NEVER raises. Logs WARNING on partial failures so the issue surfaces
    in journalctl without breaking the chat path.
    """
    out: dict[str, Any] = {
        "neuromods": {},
        "embedding_hash": "",
        "importance": DEFAULT_IMPORTANCE,
    }

    if reader_bank is None:
        try:
            from titan_hcl.api.shm_reader_bank import ShmReaderBank
            reader_bank = ShmReaderBank()
        except Exception as exc:
            # Boot-time race or SHM not yet mapped — degraded snapshot
            # is the spec-correct response (defensive Phase 2 pattern).
            logger.debug(
                "[turn_snapshot] ShmReaderBank unavailable, "
                "shipping empty snapshot: %s", exc)
            return out

    # 1) Neuromodulator levels — 6 modulators per NEUROMOD_NAMES.
    try:
        nm = reader_bank.read_neuromod()
        if nm:
            mods = nm.get("modulators") or {}
            out["neuromods"] = {
                str(name): float(spec.get("level", 0.0))
                for name, spec in mods.items()
                if isinstance(spec, dict)
            }
    except Exception as exc:
        logger.warning(
            "[turn_snapshot] read_neuromod failed: %s — empty neuromods", exc)

    # 2) 132D unified-spirit embedding hash — trinity full_130dt + journey.
    try:
        trinity = reader_bank.read_trinity()
        if trinity:
            full_130 = trinity.get("full_130dt") or []
            journey = trinity.get("journey") or {}
            curvature = float(journey.get("curvature", 0.0))
            density = float(journey.get("density", 0.0))
            vec_132 = list(full_130) + [curvature, density]
            out["embedding_hash"] = _hash_132d(vec_132)
    except Exception as exc:
        logger.warning(
            "[turn_snapshot] read_trinity failed: %s — empty embedding_hash", exc)

    return out


def _safe_json_canonical(obj: Any) -> bytes:
    """Canonical JSON encoding for stable hashing across runs/machines.

    Sorted keys + no whitespace + UTF-8. Wide-net fallback to repr() so
    non-JSON-serializable values still hash deterministically per-process
    (cross-process drift on weird types is acceptable — chat tool args
    are JSON-flat in practice).
    """
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          default=str).encode("utf-8")
    except Exception:
        return repr(obj).encode("utf-8", errors="replace")


def extract_tool_calls(tools: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Normalize agno `RunOutput.tools` into the §8.2 chat-TX shape.

    Per arch §7 chat-TX `tool_calls[]` carry. Args + result are HASHED
    (not stored inline) — full bodies live in the procedural-fork TX
    (Phase 8) or per-tool oracle anchors; chat TX stores hashes for
    integrity + Phase 4 cross-ref.

    Args:
        tools: list of agno `ToolExecution` dataclass instances (or
            anything exposing `.tool_name` / `.tool_args` / `.result` /
            `.metrics` / `.tool_call_error`). None or empty → empty list.

    Returns:
        list[dict] each:
          {
            "tool":        str (tool_name, empty if missing),
            "args_hash":   64-char sha256 hex (empty for no args),
            "result_hash": 64-char sha256 hex (empty for no result),
            "latency_ms":  int (0 if metrics unavailable),
            "exception":   bool (tool_call_error),
          }

    Soft-fails per-entry — a malformed ToolExecution skips with WARN
    rather than failing the whole list.
    """
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for tx in tools:
        try:
            tool_name = str(getattr(tx, "tool_name", "") or "")
            args = getattr(tx, "tool_args", None)
            args_hash = (hashlib.sha256(_safe_json_canonical(args)).hexdigest()
                         if args is not None else "")
            result = getattr(tx, "result", None)
            result_hash = (hashlib.sha256(
                (result if isinstance(result, (bytes, bytearray))
                 else str(result).encode("utf-8", errors="replace"))
            ).hexdigest() if result is not None else "")
            # ToolCallMetrics typically exposes `time` (seconds float).
            metrics = getattr(tx, "metrics", None)
            latency_ms = 0
            if metrics is not None:
                t = getattr(metrics, "time", None)
                if t is not None:
                    try:
                        latency_ms = int(float(t) * 1000)
                    except (TypeError, ValueError):
                        latency_ms = 0
            exception = bool(getattr(tx, "tool_call_error", False))
            out.append({
                "tool": tool_name,
                "args_hash": args_hash,
                "result_hash": result_hash,
                "latency_ms": latency_ms,
                "exception": exception,
            })
        except Exception as exc:
            # Don't let one bad tool entry kill the chat-TX content.
            logger.warning(
                "[turn_snapshot] extract_tool_calls skipped bad entry: %s", exc)
    return out


__all__ = [
    "capture_turn_snapshot",
    "extract_tool_calls",
    "DEFAULT_IMPORTANCE",
    "UNIFIED_SPIRIT_DIM",
]

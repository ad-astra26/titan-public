"""LLMJudge — Tier-1-LLM fallback scorer (Phase 8, INV-Syn-21).

Runs in dream consolidation BEFORE ProceduralMiner. For every tool-call TX
in the dream window with `scored_by=null` (P6 oracle routing returned
`unknown` or didn't fire), produces a `{verdict, rationale, version_tag}`
via an injected LLM provider, anchors ONE meta-fork batch TX with all
verdicts (Merkle-batched per INV-Syn-12), then anchors ONE follow-up
ScoredByPatch TX so OracleCoverage sees the updated A.6 numerator.

Per-pass cap (default 200, most-recent first) bounds LLM cost per dream
window. Older items wait for the next dream cycle.

`version_tag` = `{model_id}|{prompt_sha256[:12]}` recorded on every
verdict so future scoring-drift audits can re-evaluate outcomes against
historical judges (per rFP §20 "Tier-1-LLM scoring drift" failure mode).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Defaults (overridable from titan_params.toml [synthesis.skill]) ────

DEFAULT_PER_PASS_CAP: int = 200
DEFAULT_TIMEOUT_S: float = 30.0


VALID_VERDICTS = ("full", "partial", "failure")


# Stable prompt — small text per scoring drift mitigation. Any prompt change
# moves the version_tag suffix so historical verdicts stay attributable to
# their judge generation.
JUDGE_PROMPT_TEMPLATE = (
    "You are a Tier-1-LLM fallback judge for Titan's procedural memory. "
    "Given a tool-call record, decide whether it succeeded relative to its goal. "
    "Output STRICT JSON with three fields: "
    'verdict (one of: full, partial, failure), '
    'rationale (one-sentence explanation, string), '
    'confidence (float 0.0..1.0). '
    "No prose outside JSON.\n\n"
    "Tool call:\n{record}\n"
)


def _prompt_version_tag(model_id: str) -> str:
    """Stable per-judge-generation tag for drift tracking."""
    prompt_hash = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()[:12]
    return f"{model_id}|{prompt_hash}"


def _render_record(tx: dict) -> str:
    """Compact record string for the judge prompt. Keep small — the prompt
    pays per tool-call TX."""
    content = tx.get("content") or {}
    summary = {
        "tool_id": content.get("tool_id") or content.get("tool"),
        "args_keys": sorted(list((content.get("args") or {}).keys())) if isinstance(content.get("args"), dict) else [],
        "success": bool(content.get("success", False)),
        "result_summary": (content.get("result_summary") or "")[:240],
        "exception": (content.get("exception") or "")[:120],
        "latency_ms": int(content.get("latency_ms") or 0),
        "parent_goal": (content.get("parent_goal") or "")[:120],
    }
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))


def _parse_verdict(raw: str) -> Optional[dict]:
    """Parse the LLM's JSON. Returns None on malformed / unknown verdict."""
    if not raw:
        return None
    try:
        # Greedy: take the first {...} block in case the model wraps in prose.
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        obj = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    verdict = obj.get("verdict")
    if verdict not in VALID_VERDICTS:
        return None
    rationale = obj.get("rationale") or ""
    if not isinstance(rationale, str):
        rationale = str(rationale)
    confidence = obj.get("confidence")
    try:
        conf_val = max(0.0, min(1.0, float(confidence))) if confidence is not None else None
    except (TypeError, ValueError):
        conf_val = None
    return {
        "verdict": verdict,
        "rationale": rationale[:512],
        "confidence": conf_val,
    }


def _merkle_root(leaves: list[bytes]) -> str:
    """SHA-256 binary tree root with leaf duplication on odd counts.
    Empty input → sha256(b'') so the proof is provably-non-empty (mirrors
    Phase 5 `synthesis/merkle.py`)."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    layer = [hashlib.sha256(l).digest() for l in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i] + layer[i + 1]).digest() for i in range(0, len(layer), 2)]
    return layer[0].hex()


class LLMJudge:
    """Tier-1-LLM fallback judge. INV-Syn-21.

    Constructor params:
      tool_call_reader: `(since_ts, limit) -> list[dict]` returns tool-call
                        TX dicts in the window. Same shape miner uses.
      llm_provider: `(prompt: str, timeout_s: float) -> str` returns raw text.
                    Tests inject a deterministic mock; production wires the
                    Ollama Cloud provider (mirrors P4 ConsolidationPass).
      model_id: identifier baked into version_tag (e.g. "gemma4:31b").
      outer_memory_writer: anchors the verdict batch + scored_by patch.
      bus_emit: optional `(event, payload)` callable for telemetry.
      clock: time source (overridable for tests).
      per_pass_cap: bound LLM cost per dream window (default 200).
      timeout_s: per-TX LLM timeout (default 30.0).
    """

    def __init__(
        self,
        *,
        tool_call_reader: Callable[[float, int], list[dict]],
        llm_provider: Callable[[str, float], str],
        outer_memory_writer: Any,
        model_id: str = "ollama_cloud_deepseek",
        bus_emit: Optional[Callable[[str, dict], None]] = None,
        clock: Callable[[], float] = time.time,
        per_pass_cap: int = DEFAULT_PER_PASS_CAP,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._reader = tool_call_reader
        self._llm = llm_provider
        self._writer = outer_memory_writer
        self._model_id = model_id
        self._bus_emit = bus_emit
        self._clock = clock
        self._per_pass_cap = max(1, int(per_pass_cap))
        self._timeout_s = float(timeout_s)
        self._version_tag = _prompt_version_tag(model_id)

    # ── Per-TX scoring ────────────────────────────────────────────────

    def score_tx(self, tx: dict) -> Optional[dict]:
        """Return `{verdict, rationale, confidence, version_tag}` or None on
        LLM failure / timeout / unparseable response."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(record=_render_record(tx))
        try:
            raw = self._llm(prompt, self._timeout_s)
        except Exception as e:
            logger.warning("[LLMJudge] provider raised: %s", e)
            return None
        parsed = _parse_verdict(raw)
        if not parsed:
            return None
        parsed["version_tag"] = self._version_tag
        return parsed

    # ── Window scoring (one-call surface) ─────────────────────────────

    def score_window(self, *, since_ts: float) -> dict:
        """Score every null-scored_by tool-call TX in window. Returns summary."""
        now = float(self._clock())
        try:
            txs = list(self._reader(since_ts, 5000))
        except Exception as e:
            logger.warning("[LLMJudge] tool_call_reader raised: %s", e)
            txs = []

        # Filter to scored_by IS NULL. tool_call_reader is allowed to pre-filter;
        # we re-check here so a permissive reader doesn't double-score.
        unscored = [
            tx for tx in txs
            if (tx.get("content") or {}).get("scored_by") in (None, "")
        ]
        # Most-recent first (cap)
        unscored.sort(
            key=lambda t: float((t.get("content") or {}).get("ts") or 0.0),
            reverse=True,
        )
        unscored = unscored[:self._per_pass_cap]

        verdict_entries: list[dict] = []
        scored_pairs: list[dict] = []
        llm_calls = 0
        llm_failures = 0

        for tx in unscored:
            tx_hash = tx.get("tx_hash") or tx.get("hash") or (tx.get("content") or {}).get("tx_hash")
            if not tx_hash:
                llm_failures += 1
                continue
            llm_calls += 1
            parsed = self.score_tx(tx)
            if not parsed:
                llm_failures += 1
                continue
            verdict_entries.append({
                "parent_tool_call_tx": tx_hash,
                "verdict": parsed["verdict"],
                "rationale": parsed["rationale"],
                "confidence": parsed.get("confidence"),
                "version_tag": parsed["version_tag"],
            })
            scored_pairs.append({
                "parent_tool_call_tx": tx_hash,
                "scored_by": "llm",
            })

        # Anchor the two batch TXs even when empty — verifies the pass ran
        # AND produces an auditable record of "no eligible TXs this window".
        # An empty-leaves Merkle root is still well-defined per _merkle_root.
        leaves = [
            (e["parent_tool_call_tx"] + ":" + e["verdict"]).encode("utf-8")
            for e in verdict_entries
        ]
        merkle_root = _merkle_root(leaves)
        batch_tx_hash: Optional[str] = None
        patch_tx_hash: Optional[str] = None
        try:
            batch_tx_hash = self._writer.write_llm_judge_batch(
                merkle_root=merkle_root,
                entries=verdict_entries,
            )
        except Exception as e:
            logger.warning("[LLMJudge] write_llm_judge_batch failed: %s", e)
        if scored_pairs:
            try:
                patch_tx_hash = self._writer.write_scored_by_patch(entries=scored_pairs)
            except Exception as e:
                logger.warning("[LLMJudge] write_scored_by_patch failed: %s", e)
            # G1 (AUDIT §5.3): ALSO anchor a per-call tool_call_score TX on the
            # PROCEDURAL fork for each scored call. Its tags (scored_by + the
            # full 64-hex parent tx) survive v2 batch-sealing, so the procedural
            # tool_call_reader overlays scored_by onto the tool_call records —
            # the meta scored_by_patch above is kept as audit but v2 sealing
            # drops its per-entry content, so the miner can't read it from there.
            for pair in scored_pairs:
                try:
                    self._writer.write_tool_call_score(
                        parent_tool_call_tx=pair["parent_tool_call_tx"],
                        scored_by=pair["scored_by"],
                    )
                except Exception as e:
                    logger.debug("[LLMJudge] write_tool_call_score failed: %s", e)

        summary = {
            "ts": now,
            "window_since_ts": since_ts,
            "tool_calls_in_window": len(txs),
            "unscored_in_window": len([tx for tx in txs if (tx.get("content") or {}).get("scored_by") in (None, "")]),
            "scored_now": len(scored_pairs),
            "llm_calls": llm_calls,
            "llm_failures": llm_failures,
            "model_id": self._model_id,
            "version_tag": self._version_tag,
            "batch_tx_hash": batch_tx_hash,
            "scored_by_patch_tx_hash": patch_tx_hash,
            "merkle_root": merkle_root,
        }
        self._emit("META_LLM_JUDGE_PASS_DONE", summary)
        return summary

    def _emit(self, event: str, payload: dict) -> None:
        if self._bus_emit is None:
            return
        try:
            self._bus_emit(event, payload)
        except Exception as e:
            logger.warning("[LLMJudge] bus_emit %s failed: %s", event, e)

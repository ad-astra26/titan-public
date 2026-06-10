"""ProceduralMiner — dream-time recurrent-sequence miner (Phase 8).

Per `ARCHITECTURE_synthesis_engine.md §8.4` + `PLAN_synthesis_engine_Phase8.md §P8.B`.
Runs inside synthesis_worker on DREAM_STATE_CHANGED (after LLMJudge, before
ForkGC). Queries the last `window_hours` of procedural-fork tool-call TXs with
`scored_by ∈ {oracle, llm}` (INV-Syn-15 closure via INV-Syn-21), canonicalizes
each call as `(tool_id, args_shape_hash[:8])`, groups consecutive sequences by
parent (`parent_chat_tx` falls back to `parent_goal`), slides windows of length
∈ [min_seq_len, max_seq_len], GROUP_BY canonical sequence_tuple, keeps clusters
with `≥ min_occurrences`, splits each cluster into positive (success-terminal)
+ negative (failure-terminal), LLM-abstracts each into a `{nl_description,
executable_spec, preconditions, postconditions}` dict, computes a deterministic
`skill_id` from `(sequence_tuple, kind)`, persists via `ProceduralSkillStore`,
emits META_SKILL_COMPILED, and anchors ONE `skill_mining_pass` meta-fork TX.

The miner is idempotent: same recurrent shape on the next pass produces the
same `skill_id`, so `ProceduralSkillStore.persist_skill` returns the existing
row's preserved counters (success/failure/verified_at carry forward).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Defaults (overridable by titan_params.toml [synthesis.skill]) ─────

DEFAULT_WINDOW_HOURS: int = 168              # rFP §11.4 "last 7d"
DEFAULT_MIN_SEQ_LEN: int = 2                 # rFP §11.4 "len ≥2"
DEFAULT_MIN_OCCURRENCES: int = 3             # rFP §11.4 "≥3 occurrences"
DEFAULT_MAX_SKILLS_PER_PASS: int = 10        # bounds LLM cost per dream window
DEFAULT_MAX_SEQ_LEN: int = 8                 # sliding window upper bound


# ── Canonicalization ───────────────────────────────────────────────────


def canonicalize_call(tx: dict) -> tuple[str, str]:
    """Return `(tool_id, args_shape_hash[:8])`.

    Argument-shape-aware, value-agnostic per Q2: two calls with the same
    arg-keys and same arg-value types collapse onto the same canonical key
    regardless of actual values. The hash is sha256 over a canonical JSON
    of `{arg_key: type_name}` to make the shape stable across runs.
    Returns `("unknown", "00000000")` on a malformed input."""
    try:
        content = tx.get("content") or {}
        tool_id = content.get("tool_id") or content.get("tool") or "unknown"
        args = content.get("args") or {}
        if not isinstance(args, dict):
            args_shape: dict = {}
        else:
            args_shape = {k: type(v).__name__ for k, v in args.items()}
        shape_payload = json.dumps(
            args_shape, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        shape_hash = hashlib.sha256(shape_payload).hexdigest()[:8]
        return (str(tool_id), shape_hash)
    except Exception:
        return ("unknown", "00000000")


def compute_skill_id(sequence_tuple: tuple, kind: str) -> str:
    """Deterministic skill_id from (sequence_tuple, kind).
    Same recurrent shape → same skill_id on every pass (idempotent miner)."""
    payload = json.dumps(
        {"seq": list(sequence_tuple), "kind": kind},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return "skill_" + hashlib.sha256(payload).hexdigest()[:16]


# ── ProceduralMiner ────────────────────────────────────────────────────


# Callable signatures (typed via Callable here to keep tests free of imports).
ToolCallReader = Callable[[float, int], list[dict]]   # (since_ts, limit) -> list of tx dicts
LLMProposer = Callable[[dict, str], Optional[dict]]   # (cluster_meta, kind) -> abstracted dict or None
BusEmit = Callable[[str, dict], None]                 # (event_name, payload) -> None


class ProceduralMiner:
    """Dream-time miner. Single-call API: `mine_pass(now_ts) -> summary`."""

    def __init__(
        self,
        *,
        skill_store: Any,
        tool_call_reader: ToolCallReader,
        llm_proposer: LLMProposer,
        outer_memory_writer: Any,
        bus_emit: Optional[BusEmit] = None,
        clock: Callable[[], float] = time.time,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        max_skills_per_pass: int = DEFAULT_MAX_SKILLS_PER_PASS,
    ):
        self._skill_store = skill_store
        self._tool_call_reader = tool_call_reader
        self._llm_proposer = llm_proposer
        self._outer = outer_memory_writer
        self._bus_emit = bus_emit
        self._clock = clock
        self._window_hours = int(window_hours)
        self._min_seq_len = max(2, int(min_seq_len))
        self._max_seq_len = max(self._min_seq_len, int(max_seq_len))
        self._min_occurrences = max(2, int(min_occurrences))
        self._max_skills_per_pass = max(1, int(max_skills_per_pass))

    # ── Group + cluster ────────────────────────────────────────────────

    def group_by_parent(self, txs: list[dict]) -> dict[str, list[dict]]:
        """Group tool-call TXs by `parent_chat_tx` (or `parent_goal` fallback
        for autonomous tool calls). Returns {parent_id: [tx, ...]} ordered by ts."""
        groups: dict[str, list[dict]] = defaultdict(list)
        for tx in txs:
            content = tx.get("content") or {}
            parent = content.get("parent_chat_tx") or content.get("parent_goal") or "__orphan__"
            groups[parent].append(tx)
        for k in groups:
            groups[k].sort(key=lambda t: float((t.get("content") or {}).get("ts") or 0.0))
        return dict(groups)

    def cluster_sequences(self, groups: dict[str, list[dict]]) -> dict[tuple, list[list[dict]]]:
        """Slide windows of length ∈ [min_seq_len, max_seq_len] across each group,
        emit (sequence_tuple, [member_txs[]] ...) clusters."""
        clusters: dict[tuple, list[list[dict]]] = defaultdict(list)
        for parent, txs in groups.items():
            n = len(txs)
            if n < self._min_seq_len:
                continue
            canonical = [canonicalize_call(t) for t in txs]
            # Sliding windows
            for length in range(self._min_seq_len, min(self._max_seq_len, n) + 1):
                for start in range(0, n - length + 1):
                    seq = tuple(canonical[start:start + length])
                    members = txs[start:start + length]
                    clusters[seq].append(members)
        return dict(clusters)

    def filter_recurrent(self, clusters: dict[tuple, list[list[dict]]]) -> list[dict]:
        """Keep clusters with ≥ min_occurrences. Returns list of cluster dicts."""
        out = []
        for seq, member_groups in clusters.items():
            if len(member_groups) < self._min_occurrences:
                continue
            out.append({
                "sequence": seq,
                "occurrence_count": len(member_groups),
                "members": member_groups,
            })
        # Longest sequences first (richer skills) then highest count
        out.sort(key=lambda c: (-len(c["sequence"]), -c["occurrence_count"]))
        return out

    # ── Success/failure split ──────────────────────────────────────────

    def split_success_failure(self, cluster: dict) -> dict[str, list[list[dict]]]:
        """rFP §11.3: compile failure too. For each member sequence, classify
        by the LAST tx's outcome:

          positive  ← success=True AND verdict-non-failure (scored_by ∈ {oracle, llm}
                      AND a 'failure' tag was NOT set)
          negative  ← all other terminal outcomes (exception, success=False, or
                      verdict==failure in the LLM judge's eyes)

        Returns {'positive': [member_seq, ...], 'negative': [member_seq, ...]}.
        Clusters where either bucket has < min_occurrences are filtered out by
        the caller's secondary check (so we may return empty lists here)."""
        positive: list[list[dict]] = []
        negative: list[list[dict]] = []
        for member_seq in cluster.get("members") or []:
            if not member_seq:
                continue
            last = member_seq[-1]
            content = last.get("content") or {}
            success = bool(content.get("success", False))
            scored_by = content.get("scored_by")
            # Negative path: success False, or a failure-tagged verdict was attached
            terminal_failure = (not success) or _has_failure_tag(last)
            # Only count items the miner is allowed to learn from (Tier-1+).
            if scored_by not in ("oracle", "llm"):
                continue
            if terminal_failure:
                negative.append(member_seq)
            else:
                positive.append(member_seq)
        return {"positive": positive, "negative": negative}

    # ── LLM abstraction ────────────────────────────────────────────────

    def abstract_cluster(self, cluster: dict, members: list[list[dict]], kind: str) -> Optional[dict]:
        """Invoke the LLM proposer to abstract a cluster.
        Returns a dict with `{nl_description, executable_spec, preconditions,
        postconditions, compiled_from}` or None on failure / unparseable."""
        if not members:
            return None
        cluster_meta = {
            "sequence": list(cluster["sequence"]),
            "occurrence_count": len(members),
            "kind": kind,
            "members_summary": _members_summary(members),
        }
        try:
            proposal = self._llm_proposer(cluster_meta, kind)
        except Exception as e:
            logger.warning("[ProceduralMiner] llm_proposer raised: %s", e)
            return None
        if not isinstance(proposal, dict):
            return None
        nl = (proposal.get("nl_description") or "").strip()
        spec = proposal.get("executable_spec") or {}
        pre = proposal.get("preconditions") or []
        post = proposal.get("postconditions") or []
        if not nl or not isinstance(spec, dict):
            return None
        # compiled_from = unique source tx_hashes across all member sequences
        compiled_from: list[str] = []
        seen: set[str] = set()
        for seq in members:
            for tx in seq:
                h = tx.get("tx_hash") or tx.get("hash") or (tx.get("content") or {}).get("tx_hash")
                if h and h not in seen:
                    seen.add(h)
                    compiled_from.append(h)
        return {
            "nl_description": nl,
            "executable_spec": spec,
            "preconditions": list(pre) if isinstance(pre, list) else [],
            "postconditions": list(post) if isinstance(post, list) else [],
            "compiled_from": compiled_from,
        }

    # ── mine_pass (the one-call surface) ───────────────────────────────

    def mine_pass(self, *, dream_pass_id: Optional[str] = None) -> dict:
        """Run one full mining pass. Returns a summary dict suitable for the
        consolidation_pass meta-fork anchor.

        Steps (rFP §11.4 / arch §8.4):
        1. Fetch tool-call TXs from window (scored_by ∈ {oracle, llm}).
        2. Group by parent.
        3. Cluster sliding windows.
        4. Filter to recurrent clusters (≥ min_occurrences).
        5. Split each cluster into positive/negative sub-clusters.
        6. LLM-abstract each side (capped at max_skills_per_pass total).
        7. persist_skill (idempotent — skill_id is deterministic).
        8. Emit META_SKILL_COMPILED per skill.
        9. Anchor ONE skill_mining_pass meta-fork TX summarizing the pass.
        """
        now = float(self._clock())
        since_ts = now - self._window_hours * 3600.0
        try:
            txs = list(self._tool_call_reader(since_ts, 5000))
        except Exception as e:
            logger.warning("[ProceduralMiner] tool_call_reader raised: %s", e)
            txs = []

        groups = self.group_by_parent(txs)
        clusters_all = self.cluster_sequences(groups)
        recurrent = self.filter_recurrent(clusters_all)
        logger.info(
            "[ProceduralMiner] mine_pass: txs=%d groups=%d clusters=%d "
            "recurrent=%d (each recurrent cluster → up to 2 LLM abstractions)",
            len(txs), len(groups), len(clusters_all), len(recurrent))

        positive_compiled = 0
        negative_compiled = 0
        llm_calls = 0
        llm_failures = 0
        compiled_ids: list[str] = []
        total_skills_compiled = 0

        for cluster in recurrent:
            if total_skills_compiled >= self._max_skills_per_pass:
                break
            split = self.split_success_failure(cluster)
            # EEL B1 (D-SPEC-153) — the miner is NEGATIVE-ONLY: positive skills now
            # form per oracle-verified use (the skill_score_events queue), so the
            # miner only compiles recurrent FAILURE shapes (INV-5 / B2 replay fuel).
            for kind in ("negative",):
                if total_skills_compiled >= self._max_skills_per_pass:
                    break
                members = split[kind]
                if len(members) < self._min_occurrences:
                    continue
                llm_calls += 1
                logger.info(
                    "[ProceduralMiner] abstracting cluster (seq_len=%d, %s, "
                    "members=%d) — LLM call %d",
                    len(cluster["sequence"]), kind, len(members), llm_calls)
                abstracted = self.abstract_cluster(cluster, members, kind)
                if not abstracted:
                    llm_failures += 1
                    continue
                # EEL B1 — a recurrent failure shape → a NEGATIVE cell on an
                # outcome derived from the abstraction (goal_class) + the recurrent
                # tool-path (task_shape). Never delegated (INV-EEL-5 polarity guard);
                # serves avoidance + the B2 replay queue. oracle_id sentinel marks
                # the recurrence origin (distinct from the per-use oracle outcomes).
                from titan_hcl.synthesis.goal_class import (
                    goal_class as _derive_goal_class,
                    make_task_shape as _derive_task_shape,
                )
                base_name = abstracted["nl_description"].split(".")[0][:64] or "compiled_skill"
                full_name = f"[negative] {base_name}"
                _gclass = _derive_goal_class(abstracted["nl_description"])
                _seq = cluster.get("sequence")
                _tool_sig = ("+".join(str(s) for s in _seq)
                             if isinstance(_seq, (list, tuple)) else str(_seq))
                _task_shape = _derive_task_shape("procedural", _tool_sig[:80], "")
                try:
                    skill_id = self._skill_store.persist_negative_skill(
                        oracle_id="miner_recurrence",
                        goal_class=_gclass,
                        task_shape=_task_shape,
                        name=full_name,
                        nl_description=abstracted["nl_description"],
                        compiled_from=abstracted["compiled_from"],
                        executable_spec=abstracted["executable_spec"],
                        preconditions=abstracted["preconditions"],
                        postconditions=abstracted["postconditions"],
                        ts=now,
                    )
                except Exception as e:
                    logger.warning(
                        "[ProceduralMiner] persist_negative_skill failed: %s", e)
                    llm_failures += 1
                    continue
                compiled_ids.append(skill_id)
                total_skills_compiled += 1
                if kind == "positive":
                    positive_compiled += 1
                else:
                    negative_compiled += 1
                self._emit("META_SKILL_COMPILED", {
                    "skill_id": skill_id,
                    "kind": kind,
                    "sequence_len": len(cluster["sequence"]),
                    "occurrence_count": len(members),
                    "ts": now,
                })

        summary = {
            "dream_pass_id": dream_pass_id,
            "ts": now,
            "txs_scanned": len(txs),
            "groups_built": len(groups),
            "clusters_total": len(clusters_all),
            "clusters_recurrent": len(recurrent),
            "positive_skills_compiled": positive_compiled,
            "negative_skills_compiled": negative_compiled,
            "llm_calls": llm_calls,
            "llm_failures": llm_failures,
            "compiled_ids": compiled_ids,
        }

        self._anchor_mining_pass(summary)
        return summary

    # ── Internals ──────────────────────────────────────────────────────

    def _emit(self, event: str, payload: dict) -> None:
        if self._bus_emit is None:
            return
        try:
            self._bus_emit(event, payload)
        except Exception as e:
            logger.warning("[ProceduralMiner] bus_emit %s failed: %s", event, e)

    def _anchor_mining_pass(self, summary: dict) -> None:
        """Anchor ONE meta-fork TX per pass (mirrors P4 ConsolidationPass)."""
        if self._outer is None:
            return
        try:
            payload = dict(summary)
            # Drop the list of compiled IDs from the anchored content if it grows large
            # (keep it on the bus event instead). Cap at 20.
            ids = payload.get("compiled_ids") or []
            if len(ids) > 20:
                payload["compiled_ids"] = ids[:20]
                payload["compiled_ids_truncated"] = True
            # Many OuterMemoryWriter codepaths use write_consolidation_pass; if not present
            # fall back to a generic emit. Keep this defensive.
            writer_fn = getattr(self._outer, "write_skill_mining_pass", None)
            if writer_fn is None:
                writer_fn = getattr(self._outer, "write_consolidation_pass", None)
            if writer_fn is None:
                # Last-resort: skip the anchor; the bus events still carry the summary.
                logger.debug(
                    "[ProceduralMiner] outer_memory_writer has no mining-pass anchor method; "
                    "summary published via bus only"
                )
                return
            writer_fn(payload)
        except Exception as e:
            logger.warning("[ProceduralMiner] anchor_mining_pass failed: %s", e)


# ── Helpers ────────────────────────────────────────────────────────────


def _has_failure_tag(tx: dict) -> bool:
    """Look for a 'failure' tag or a verdict==failure in the content."""
    tags = tx.get("tags") or []
    if any(isinstance(t, str) and t.lower().startswith("failure") for t in tags):
        return True
    content = tx.get("content") or {}
    if content.get("verdict") == "failure":
        return True
    if content.get("exception"):
        return True
    return False


def _members_summary(members: list[list[dict]]) -> list[dict]:
    """Compact, LLM-friendly representation of cluster members.
    Per-sequence: just the tool_ids + arg-keys + ts. Cap at first 5 sequences
    × first 8 steps each, so prompt size stays bounded."""
    out = []
    for seq in members[:5]:
        rendered_seq = []
        for tx in seq[:8]:
            content = tx.get("content") or {}
            args = content.get("args") or {}
            rendered_seq.append({
                "tool": content.get("tool_id") or content.get("tool") or "unknown",
                "arg_keys": sorted(list(args.keys())) if isinstance(args, dict) else [],
                "success": bool(content.get("success", False)),
                "scored_by": content.get("scored_by"),
                "ts": float(content.get("ts") or 0.0),
            })
        out.append({"steps": rendered_seq})
    return out

"""Production-grade default implementations of the ConsolidationPass
mine + LLM-propose callables (§P4.G).

`titan_hcl/synthesis/consolidation.py` holds the pure orchestration; this
module provides the two REAL collaborators production wires at synthesis
worker boot. Tests use fakes (see test_phase4_consolidation.py).

- `default_mine_recent_thoughts(since_ts, exclude_forks)` — reads recent
  PROMOTED THOUGHTS from the lock-free content sidecar (`data/thought_sidecar.db`),
  keyed by their per-TX promotion hash. This is the crux fix (RFP
  `_synthesis_spine_reads_real_data` §7.D / D1): the old `default_mine_recent_txs`
  mined the chain `block_index` ENVELOPE (keyed by block_hash, no content) →
  embeddings double-missed + the proposer saw no content → 0 real concepts. The
  sidecar key matches the recall deref + the `conversation` FAISS shard, so
  `synthesis_worker._mine_with_embeddings` fetches each candidate's embedding from
  that shard (cosine clustering) and the proposer sees real content. One
  TxCandidate per thought; `embedding` filled by the worker (None here).

- `default_llm_propose(cluster, provider)` — calls the inference module's
  `complete()` to ask the model whether the cluster represents a new
  concept, a version-bump of an existing one, or nothing coherent. The cluster
  prompt now carries a bounded sample of REAL thought content (RFP §7.D / D3) so
  the proposed concept name + memory_type reflect the actual thoughts.
  Output is parsed via a strict line-prefix protocol so a verbose LLM
  response cannot break clustering. Provider failure / parse error →
  LLMProposal(action="reject", reason=<diagnostic>) — never raises.

Both functions are sync; the LLM call is bridged via `asyncio.run` since
synthesis_worker's bus loop is thread-based (no surrounding event loop).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from titan_hcl.synthesis.consolidation import (
    Cluster,
    LLMProposal,
    TxCandidate,
)

logger = logging.getLogger(__name__)

# Hard-cap rows pulled per mine call. The dream window is bounded but
# autonomous tool-call volume can be large; capping protects the dream-
# boundary budget. Per arch §16.1 tiered anchoring caps individual-anchor
# growth so the absolute number stays moderate; 500 is comfortably above
# any organic-traffic 24h window seen in P3 fleet verify.
_MINE_ROW_CAP = 500


# ── §7.F advisory domain_hint — normalization + one-time content backfill ──


def _normalize_domain_hint(raw: str) -> str:
    """Fold a free-text domain hint to its stored form: lowercased + stripped;
    placeholder / non-answer tokens → "" (§7.F — advisory, mutable, never
    gates). The SINGLE source of this rule, shared by the live LLM-`DOMAIN:`
    parse (`_parse_llm_response`) and the one-time content backfill
    (`derive_domain_hint`) so both produce identical stored values."""
    h = (raw or "").strip().lower()
    if h.startswith("<") or h in ("empty", "unclear", "none"):
        return ""
    return h


# Ordered keyword → domain rules for the one-time content backfill of pre-
# Phase-F Engrams whose `domain_hint` is "" (BUG-ENGRAM-DOMAIN-HINT-NOT-
# BACKFILLED). The §7.F fix-plan sanctions "a cheap classifier, normalized per
# consolidation_defaults" as the alternative to re-running the consolidation
# LLM. First-match-wins, most-specific → general. The vocabulary is the LIVE
# observed set (self / philosophy_of_mind / philosophy / sociology /
# neuroscience) plus the two unambiguous domains present in the pre-F blanks
# (security, coding). A name matching nothing confident → "" (left blank — a
# future re-consolidation fills it; never fabricate a low-confidence label).
_DOMAIN_BACKFILL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("consciousness", "phenomenolog", "pre-conceptual", "pre conceptual",
      "pre-label", "pre label", "qualia", "sentience"), "philosophy_of_mind"),
    (("neuroscience", "neural correlate", "neuro"), "neuroscience"),
    (("adversarial", "social engineering", "prompt attack", "prompt pattern",
      "prompting pattern", "jailbreak", "injection"), "security"),
    (("coding", "sandbox", "code verification", "compiler", "programming"),
     "coding"),
    (("sociolog",), "sociology"),
    # The dominant introspective / interpersonal / social bucket — matches how
    # the live LLM labeled this content ("Seaside Philosophical Dialogue",
    # "Interpersonal Dialogue Fragments", "User Interpersonal Dynamics" → self).
    (("interpersonal", "social", "dialogue", "human", "self", "introspect",
      "reflection", "presence", "stillness", "contemplation", "resonance",
      "bond", "emotional", "musician", "seaside", "wisdom",
      "dream consolidation", "interaction"), "self"),
    # Remaining philosophical content with no social/dialogue/reflection cue.
    (("philosoph", "metaphysic", "epistemolog", "ontolog"), "philosophy"),
)

# Names that are test pollution, not real Engrams — never label these (they
# stay blank + are surfaced for separate hygiene cleanup).
_DOMAIN_BACKFILL_SKIP = ("e2e_test", "newconcept_", "test_newconcept",
                         "__test", "pytest")


def derive_domain_hint(name: str, memory_type: str = "",
                       member_text: str = "") -> str:
    """Cheap deterministic content→domain classifier for the one-time backfill
    of pre-Phase-F Engrams whose `domain_hint` is "" (Phase F set it only at
    consolidation time; existing Engrams predate it). Derives a coarse advisory
    hint from the Engram NAME (+ optional `member_text` where available — pre-F
    Engrams have no persisted membership, so the descriptive name is the
    practical signal). Returns a label from the live vocabulary or "" (leave
    blank — never fabricate). Output is `_normalize_domain_hint`-folded so it is
    byte-identical to what the LLM path stores. Advisory / mutable / never gates
    (§7.F).

    NOT the consolidation path: new Engrams still get `domain_hint` from the LLM
    `DOMAIN:` line (`_parse_llm_response`). This retro-labels existing blanks
    only (BUG-ENGRAM-DOMAIN-HINT-NOT-BACKFILLED; precedent = the `axis_used`
    one-time backfill)."""
    blob = " ".join(p for p in (name, member_text) if p).strip().lower()
    if not blob:
        return ""
    if any(tok in blob for tok in _DOMAIN_BACKFILL_SKIP):
        return ""  # test pollution — leave blank, flag for hygiene cleanup
    for keywords, domain in _DOMAIN_BACKFILL_RULES:
        if any(kw in blob for kw in keywords):
            return _normalize_domain_hint(domain)
    return ""


# ── Default mine ────────────────────────────────────────────────────


def default_mine_recent_thoughts(
    *,
    since_ts: float,
    exclude_forks: set[str],
    data_dir: str = "data",
    row_cap: int = _MINE_ROW_CAP,
) -> list[TxCandidate]:
    """Mine recent PROMOTED THOUGHTS from the lock-free content sidecar (RFP
    `_synthesis_spine_reads_real_data` §7.D / D1) — the REAL thoughts, keyed by
    their per-TX promotion hash.

    This replaces the old `default_mine_recent_txs`, which mined the chain
    `block_index` ENVELOPE (keyed by `block_hash`, no thought content): there the
    embedding fetch double-missed (wrong shard + wrong key) → tag-only clustering
    on generic envelope tags → 0 real concepts, and the proposer saw no content.

    The sidecar key is the SAME hash the recall deref and the `conversation` FAISS
    shard use, so `synthesis_worker._mine_with_embeddings` can fetch each
    candidate's embedding from the conversation shard (cosine clustering) and the
    proposer sees the real content. One TxCandidate per thought; `embedding` is
    filled by the worker (None here). Read-only; soft-fail → []."""
    out: list[TxCandidate] = []
    try:
        from titan_hcl.synthesis.thought_sidecar import ThoughtSidecarReader
        reader = ThoughtSidecarReader(data_dir)
    except Exception as e:
        logger.warning(
            "[consolidation_defaults] mine_thoughts: sidecar reader init "
            "failed: %s", e)
        return out
    try:
        rows = reader.iter_since(since_ts=float(since_ts), limit=int(row_cap))
        for row in rows:
            txh = row.get("tx_hash")
            if not txh:
                continue
            # The chain fork (declarative/episodic) is the SEMANTIC anchor and is
            # decoupled from the embed shard (the worker fetches embeddings from the
            # `conversation` shard). The sidecar holds ONLY promoted memory thoughts
            # — there is no chat-turn `conversation`-fork or `meta` noise to filter
            # — but honor exclude_forks defensively anyway.
            fork = str(row.get("fork") or "")
            if fork in exclude_forks:
                continue
            prompt = row.get("user_prompt") or ""
            response = row.get("agent_response") or ""
            content = (prompt + "\n" + response).strip()
            out.append(TxCandidate(
                tx_hash=str(txh),
                fork=fork or "episodic",
                tags=(),  # sidecar carries none → cosine-primary clustering
                embedding=None,  # filled from the conversation FAISS shard (D2)
                content_summary=content,
                felt=row.get("felt"),  # felt-at-lived-time JSON (§7.C)
            ))
    except Exception as e:
        logger.warning(
            "[consolidation_defaults] mine_thoughts: failed (%s) — %d rows",
            e, len(out))
    finally:
        try:
            reader.close()
        except Exception:
            pass

    return out


# ── Default LLM propose ────────────────────────────────────────────


_LLM_SYSTEM_PROMPT = (
    "You are part of Titan's synthesis engine. You will receive a cluster "
    "of related thoughts (chain TXs) that may represent ONE coherent "
    "concept in Titan's experience.\n\n"
    "Decide whether the cluster is:\n"
    "  - a NEW concept that should be materialized (give it a short, "
    "human-readable concept_id like 'linux_terminal' or "
    "'metaplex_nft_minting'),\n"
    "  - a VERSION_BUMP of an existing concept (you'll be told which "
    "concept_ids already exist in the cluster's tag set),\n"
    "  - or REJECT (clusters that are noise or already represented).\n\n"
    "Respond in this EXACT format (one field per line):\n"
    "ACTION: new_concept | version_bump | reject\n"
    "CONCEPT_ID: <id_with_underscores>\n"
    "NAME: <human-readable name, empty for reject>\n"
    "MEMORY_TYPE: declarative | procedural | episodic | meta\n"
    "DOMAIN: <broad knowledge domain, e.g. biology | mathematics | music | "
    "self; empty if unclear>\n"
    "REASON: <one short sentence>\n"
)


def _build_cluster_prompt(cluster: Cluster, max_tags: int = 30,
                          max_samples: int = 5, max_chars: int = 240) -> str:
    """Compact prompt describing a cluster — a bounded sample of REAL thought
    content, fork distribution, top tags. The content sample (RFP
    `_synthesis_spine_reads_real_data` §7.D / D3) is what lets the LLM name a real
    concept; previously the proposer saw only hash prefixes + tags and could not."""
    n = len(cluster.members)
    tag_counts: dict[str, int] = {}
    fork_counts: dict[str, int] = {}
    for tx in cluster.members:
        fork_counts[tx.fork] = fork_counts.get(tx.fork, 0) + 1
        for t in tx.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(
        tag_counts.items(), key=lambda kv: -kv[1],
    )[:max_tags]
    top_tag_str = ", ".join(f"{t}×{c}" for t, c in top_tags)
    forks_str = ", ".join(f"{f}×{c}" for f, c in fork_counts.items())
    # Bounded REAL-content sample — the signal the proposer names the concept from.
    samples = []
    for m in cluster.members[:max_samples]:
        text = (m.content_summary or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        samples.append(f"  - {text}")
    samples_str = "\n".join(samples) if samples else "  (no content available)"
    return (
        f"Cluster size: {n} thought(s)\n"
        f"Forks: {forks_str}\n"
        f"Top tags: {top_tag_str or '(none)'}\n"
        f"Sample thoughts:\n{samples_str}\n"
    )


def _parse_llm_response(text: str) -> LLMProposal:
    """Parse the LLM's line-prefix response. Lenient: missing fields →
    safe defaults that still produce a valid LLMProposal. Unknown ACTION
    → reject. Never raises."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        if key in ("ACTION", "CONCEPT_ID", "NAME", "MEMORY_TYPE", "DOMAIN",
                   "REASON"):
            fields[key] = val.strip()

    action_raw = fields.get("ACTION", "reject").lower()
    if action_raw not in ("new_concept", "version_bump", "reject"):
        action_raw = "reject"

    if action_raw == "reject":
        return LLMProposal(
            action="reject",
            reason=fields.get("REASON", ""),
        )

    concept_id = fields.get("CONCEPT_ID", "").strip()
    if not concept_id:
        return LLMProposal(
            action="reject",
            reason="llm_returned_empty_concept_id",
        )
    # Sanitize concept_id: lowercase + underscores; the chain tag space
    # treats concept_ids as identifiers, not freeform names.
    concept_id = concept_id.lower().replace(" ", "_").replace("-", "_")

    memory_type = fields.get("MEMORY_TYPE", "meta").lower()
    if memory_type not in ("declarative", "procedural", "episodic", "meta"):
        memory_type = "meta"

    # §7.F — advisory domain_hint (free-text, normalized; "" if the LLM omitted
    # it or returned the placeholder). Never gates behaviour; mutable.
    domain_hint = _normalize_domain_hint(fields.get("DOMAIN", ""))

    return LLMProposal(
        action=action_raw,  # type: ignore[arg-type]
        concept_id=concept_id,
        proposed_name=fields.get("NAME", concept_id) or concept_id,
        memory_type=memory_type,
        base_concept_refs=(),
        reason=fields.get("REASON", ""),
        domain_hint=domain_hint,
    )


def make_default_llm_propose(provider: Any) -> Any:
    """Bind a provider into a sync `llm_propose_fn(cluster) -> LLMProposal`
    callable. The provider must conform to the
    `titan_hcl.inference.base.InferenceProvider` async interface — we
    bridge via asyncio.run().

    Provider failure / parse error → REJECT with a diagnostic reason
    (never raises)."""

    def _propose(cluster: Cluster) -> LLMProposal:
        prompt = _build_cluster_prompt(cluster)
        try:
            text = asyncio.run(provider.complete(
                prompt=prompt,
                system=_LLM_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=300,
                timeout=45.0,
            ))
        except Exception as e:
            logger.warning(
                "[consolidation_defaults] llm_propose provider failed: %s", e,
            )
            return LLMProposal(
                action="reject",
                reason=f"llm_provider_error: {type(e).__name__}",
            )

        return _parse_llm_response(text or "")

    return _propose


# ── Default decompose (Inner↔Outer Felt-Teaching Bridge §7.1) ──────


def make_default_decompose(provider: Any, *, max_objects: int = 8) -> Any:
    """Bind a provider into a sync `decompose_fn(name, sample_content) -> list[str]`
    that decomposes an Engram(Idea) into its constituent Object labels
    (RFP_inner_outer_felt_teaching_bridge §7.1). Mirrors `make_default_llm_propose`:
    `LanguageTeacher` builds the prompt + parses (it never invokes the LLM itself);
    this bridges the provider via `asyncio.run` (synthesis_worker's bus loop is
    thread-based, no surrounding event loop).

    Provider failure / parse error → `[]` (FeltBridge does NOT cache an empty result
    → it retries next touch; a transient failure must never be frozen as "no
    Objects"). Never raises."""
    from titan_hcl.logic.language_teacher import LanguageTeacher

    def _decompose(name: str, sample_content: str = "") -> list[str]:
        spec = LanguageTeacher.build_decompose_prompt(
            name, sample_content, max_objects=max_objects)
        try:
            text = asyncio.run(provider.complete(
                prompt=spec["prompt"],
                system=spec["system"],
                temperature=0.2,
                max_tokens=spec.get("max_tokens", 200),
                timeout=45.0,
            ))
        except Exception as e:
            logger.warning(
                "[consolidation_defaults] decompose provider failed: %s", e)
            return []

        return LanguageTeacher.parse_decompose_objects(
            text or "", max_objects=max_objects)

    return _decompose


__all__ = (
    "default_mine_recent_thoughts",
    "make_default_llm_propose",
    "make_default_decompose",
    "_parse_llm_response",
    "_build_cluster_prompt",
)

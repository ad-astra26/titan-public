"""titan_hcl.synthesis.prompt_signature — §7.E (E.2) self-verifying prompt→solution cache.

The TOP tier of the decide-time cache hierarchy: an identical, durable, non-stale
prompt is answered from a STORED verified answer with ZERO LLM/oracle/sandbox — even
cheaper than E.1's recipe replay (which still runs the sandbox). Below E.2 sits E.1
(replay the recipe for a *similar* prompt, different params); below that, the normal
route.

Two halves:
  • PromptSignatureStore — sole-writer (SynthesisWriter), synthesis-side: persists a
    verified (prompt → answer) as a `PromptSignature` (DuckDB scalars + faiss signature
    + lock-free snapshot). Mutate-not-update (INV-OML-5): a CONFLICTING answer for the
    same prompt mints a successor `::v{n}`, never overwrites.
  • PromptSignatureReader — lock-free (faiss FILE + snapshot), agno-side: SC-search the
    live prompt → serve the literal answer ONLY when durable + params identical + not
    stale. Else None → fall through to E.1.

🔒 Durability is the DK.3-shared canonical mechanic (`research_volatility` + the emergent-
epoch `ConsciousnessAgeReader`), NOT a divergent decay scheme (the Maker's shared-cache
lock, 2026-06-13). Params = numeric tokens (`recipe_template.extract_numeric_params`,
consistent with E.1) — a durable conceptual ask has no numeric params → [] → literal on
a semantic match; a compute re-ask with different numbers → params differ → not literal
→ E.1 replays. Scope = the verifiable lane (the verified `literal_answer`); research /
durable-knowledge recall stays Phase D's live DK.4 concept path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

import numpy as np

from titan_hcl.synthesis.recipe_template import extract_numeric_params

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 384
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_prompt(prompt: str) -> str:
    """Lowercased, whitespace-collapsed prompt — the literal-cache identity key
    (params INCLUDED: distinct param-sets are distinct cache entries, each with its
    own verified answer; E.1 handles param-variation by replay)."""
    return _WS_RE.sub(" ", (prompt or "").strip().lower())


def prompt_template(prompt: str) -> str:
    """The param-slotted prompt (numbers → {pN}) — for display/provenance; the
    matching uses the embedding + param-identity, not this string."""
    out = prompt or ""
    for i, val in enumerate(extract_numeric_params(prompt)):
        out = re.sub(r"(?<![\w.])" + re.escape(val) + r"(?![\w.])",
                     "{p%d}" % i, out, count=1)
    return out


def signature_id_for(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
class PromptSignatureStore:
    """Synthesis-side writer for the E.2 prompt→solution cache. Single-writer via
    the injected SynthesisWriter (INV-Syn-19/28); owns its own duckdb table, faiss
    file, and snapshot. Mirrors ReasoningStore's faiss/snapshot mechanics."""

    def __init__(self, conn: Any, *, faiss_path: str, snapshot_path: str = "",
                 embedder: Optional[Callable[[str], Any]] = None, writer: Any = None,
                 graph: Any = None):
        from titan_hcl.synthesis.writer import resolve_writer
        self._db = conn
        self._faiss_path = str(faiss_path)
        self._snapshot_path = str(snapshot_path) if snapshot_path else os.path.join(
            os.path.dirname(self._faiss_path) or ".", "prompt_signature_snapshot.json")
        self._embedder = embedder
        self._writer = resolve_writer(writer)
        self._graph = graph
        self._faiss = None
        self._dim = EMBEDDING_DIM
        self.written = 0
        self._clock = time.time
        self._init_schema()
        # §7.E — write the (initially empty) snapshot at construction so its mere
        # EXISTENCE after boot proves the store wired (constructor reached the end →
        # prompt_signature_store ≠ None). A clean observability signal on a fleet
        # whose module logs are not readily reachable.
        try:
            self.snapshot_export()
        except Exception:  # noqa: BLE001
            pass

    # ── schema ──
    def _init_schema(self) -> None:
        def _create() -> None:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS prompt_signatures ("
                "  signature_id   TEXT    PRIMARY KEY,"   # hash(normalized prompt) [+ ::v{n} successor]
                "  prompt_template TEXT,"                 # param-slotted prompt (provenance)
                "  param_values_json TEXT,"               # the numeric params that produced the answer
                "  literal_answer TEXT,"                  # the verified answer (served on an identical hit)
                "  solved_by      TEXT,"                  # the Reasoning reasoning_id that solved it (deref)
                "  durability     TEXT,"                  # 'durable' | 'volatile' (research_volatility)
                "  created_epoch  DOUBLE  DEFAULT 0,"     # emergent epoch at write (TTL base)
                "  embedding_id   INTEGER DEFAULT -1,"
                "  lineage        TEXT    DEFAULT '',"    # predecessor signature_id (mutate-not-update)
                "  created_at     DOUBLE  NOT NULL"
                ")")
        try:
            self._writer.submit_sync(_create)
        except Exception as e:  # noqa: BLE001
            logger.warning("[PromptSignatureStore] schema init failed: %s", e)

    # ── faiss ──
    def _ensure_faiss(self) -> None:
        if self._faiss is not None:
            return
        import faiss
        if os.path.exists(self._faiss_path):
            try:
                self._faiss = faiss.read_index(self._faiss_path)
                if self._faiss.d != self._dim:
                    self._faiss = faiss.IndexFlatL2(self._dim)
            except Exception:
                self._faiss = faiss.IndexFlatL2(self._dim)
        else:
            self._faiss = faiss.IndexFlatL2(self._dim)

    def _save_faiss(self) -> None:
        if self._faiss is None:
            return
        import faiss
        try:
            os.makedirs(os.path.dirname(self._faiss_path) or ".", exist_ok=True)
            tmp = self._faiss_path + ".tmp"
            faiss.write_index(self._faiss, tmp)
            os.replace(tmp, self._faiss_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("[PromptSignatureStore] faiss save failed: %s", e)

    def _embed_vec(self, text: str):
        if self._embedder is None or not text:
            return None
        try:
            vec = self._embedder(text)
            vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            if vec.shape[1] != self._dim:
                return None
            return vec
        except Exception:
            return None

    # ── write ──
    def write_signature(self, *, prompt: str, literal_answer: str, solved_by: str,
                        durability: str, created_epoch: float) -> bool:
        """Persist a verified (prompt → answer). Idempotent on an identical
        re-verify; a CONFLICTING answer for the same prompt mints a successor
        (mutate-not-update, INV-OML-5). Soft-fail."""
        if not prompt or not literal_answer:
            return False
        params = extract_numeric_params(prompt)
        base_id = signature_id_for(prompt)
        vec = self._embed_vec(prompt)

        def _do() -> bool:
            # mutate-not-update: if the base exists with a DIFFERENT answer, mint a
            # successor; identical answer → idempotent no-op.
            try:
                existing = self._db.execute(
                    "SELECT literal_answer FROM prompt_signatures "
                    "WHERE signature_id = ?", [base_id]).fetchone()
            except Exception:
                existing = None
            sig_id, lineage = base_id, ""
            if existing is not None:
                if str(existing[0]) == str(literal_answer):
                    return True  # identical re-verify → no-op
                # successor ::v{n}
                try:
                    n = self._db.execute(
                        "SELECT COUNT(*) FROM prompt_signatures "
                        "WHERE signature_id LIKE ?", [base_id + "%"]).fetchone()[0]
                except Exception:
                    n = 1
                sig_id, lineage = "%s::v%d" % (base_id, int(n) + 1), base_id
            emb_id = -1
            if vec is not None:
                try:
                    self._ensure_faiss()
                    emb_id = int(self._faiss.ntotal)
                    self._faiss.add(vec)
                    self._save_faiss()
                except Exception:
                    emb_id = -1
            try:
                self._db.execute(
                    "INSERT INTO prompt_signatures (signature_id, prompt_template, "
                    "param_values_json, literal_answer, solved_by, durability, "
                    "created_epoch, embedding_id, lineage, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT (signature_id) DO NOTHING",
                    [sig_id, prompt_template(prompt), json.dumps(params),
                     str(literal_answer), str(solved_by or ""), str(durability or "durable"),
                     float(created_epoch or 0.0), int(emb_id), lineage,
                     float(self._clock())])
            except Exception as e:  # noqa: BLE001
                logger.warning("[PromptSignatureStore] insert failed: %s", e)
                return False
            if self._graph is not None and hasattr(
                    self._graph, "spine_create_prompt_signature"):
                try:
                    self._graph.spine_create_prompt_signature(
                        signature_id=sig_id, solved_by=str(solved_by or ""),
                        durability=str(durability or "durable"),
                        created_at=float(self._clock()))
                except Exception as e:  # noqa: BLE001
                    logger.debug("[PromptSignatureStore] kuzu soft-fail: %s", e)
            self.written += 1
            return True

        try:
            ok = bool(self._writer.submit_sync(_do))
            if ok:
                self.snapshot_export()
            return ok
        except Exception as e:  # noqa: BLE001
            logger.warning("[PromptSignatureStore] write failed: %s", e)
            return False

    # ── snapshot (lock-free read surface) ──
    def _snapshot_payload(self) -> dict:
        try:
            rows = self._db.execute(
                "SELECT signature_id, embedding_id, param_values_json, literal_answer, "
                "solved_by, durability, created_epoch FROM prompt_signatures "
                "WHERE embedding_id >= 0").fetchall()
            return {"version": 1, "ts": float(self._clock()),
                    "signatures": [
                        {"signature_id": r[0], "embedding_id": int(r[1]),
                         "param_values": json.loads(r[2]) if r[2] else [],
                         "literal_answer": r[3], "solved_by": r[4],
                         "durability": r[5], "created_epoch": float(r[6] or 0.0)}
                        for r in rows]}
        except Exception as e:  # noqa: BLE001
            logger.debug("[PromptSignatureStore] snapshot payload failed: %s", e)
            return {"version": 1, "signatures": []}

    def snapshot_export(self) -> str:
        def _export() -> str:
            payload = self._snapshot_payload()
            try:
                os.makedirs(os.path.dirname(self._snapshot_path) or ".", exist_ok=True)
                tmp = self._snapshot_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                os.replace(tmp, self._snapshot_path)
            except Exception as e:  # noqa: BLE001
                logger.debug("[PromptSignatureStore] snapshot write failed: %s", e)
            return self._snapshot_path
        try:
            return str(self._writer.submit_sync(_export))
        except Exception:  # noqa: BLE001
            return self._snapshot_path

    def count(self) -> int:
        try:
            return int(self._db.execute(
                "SELECT COUNT(*) FROM prompt_signatures").fetchone()[0])
        except Exception:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
class PromptSignatureReader:
    """Lock-free agno-side reader for the E.2 cache (faiss FILE + snapshot; the
    SPEC-canonical cross-process pattern — no DuckDB open in agno). Serves a literal
    answer ONLY when the match is durable + params identical + not stale."""

    def __init__(self, faiss_path: str, snapshot_path: str, refresh_s: float = 60.0,
                 sim_floor: float = 0.93):
        self._faiss_path = str(faiss_path)
        self._snapshot_path = str(snapshot_path)
        self._refresh_s = float(refresh_s)
        self._sim_floor = float(sim_floor)
        self._index = None
        self._sigs: dict = {}    # embedding_id -> signature meta
        self._next_refresh = 0.0

    def _refresh(self, now: float) -> None:
        if self._index is not None and now < self._next_refresh:
            return
        self._next_refresh = now + self._refresh_s
        try:
            import faiss
            if os.path.exists(self._faiss_path):
                self._index = faiss.read_index(self._faiss_path)
        except Exception:
            self._index = None
        sigs: dict = {}
        try:
            if os.path.exists(self._snapshot_path):
                with open(self._snapshot_path) as f:
                    snap = json.load(f)
                for s in (snap.get("signatures") or []):
                    eid = int(s.get("embedding_id", -1))
                    if eid >= 0:
                        sigs[eid] = s
        except Exception:
            sigs = {}
        self._sigs = sigs

    def lookup(self, prompt: str, prompt_vec,
               now: Optional[float] = None) -> Optional[dict]:
        """Return `{literal_answer, solved_by, signature_id}` when the live prompt
        hits a DURABLE, params-IDENTICAL, semantically-near cached signature; else
        None (→ caller falls through to E.1). Reuses the agno embed-once `prompt_vec`.

        🔒 Durability is the DK.3-shared canonical mechanic (`classify_volatility`,
        stamped at write): a DURABLE fact is evergreen → served literally; a VOLATILE
        fact is NEVER served literally (its value may have moved → E.3 re-fetches it
        fresh). The epoch-TTL/`is_stale` is the VOLATILE class's decay (DK.3 idle pass +
        E.3 re-fetch), so it does NOT gate the durable literal serve — applying the
        volatile 417-epoch lifetime to a durable fact would wrongly expire evergreen
        answers. The shared seam E.2 reuses = the detector + the param-identity rule."""
        if now is None:
            now = time.time()
        self._refresh(now)
        if (self._index is None or not self._sigs
                or getattr(self._index, "ntotal", 0) == 0 or prompt_vec is None):
            return None
        try:
            v = np.asarray(prompt_vec, dtype=np.float32).reshape(1, -1)
            if v.shape[1] != self._index.d:
                return None
            dists, ids = self._index.search(v, 1)
            eid = int(ids[0][0])
            sig = self._sigs.get(eid)
            if sig is None:
                return None
            cos = 1.0 - float(dists[0][0]) / 2.0      # normalized embedder → cos
            if cos < self._sim_floor:
                return None                            # not the same prompt
            if str(sig.get("durability", "durable")) != "durable":
                return None                            # volatile → never literal (value may have moved → E.3)
            # params must match exactly (a compute re-ask with new numbers → E.1)
            if [str(p) for p in (sig.get("param_values") or [])] != \
                    [str(p) for p in extract_numeric_params(prompt)]:
                return None
            return {"literal_answer": sig.get("literal_answer", ""),
                    "solved_by": sig.get("solved_by", ""),
                    "signature_id": sig.get("signature_id", "")}
        except Exception:
            return None

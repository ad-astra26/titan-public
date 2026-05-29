"""titan_hcl/health/timechain_guardian.py — Phase 14 (§3K.7) Timechain Guardian.

The timechain analogue of GuardianHCL: a periodic in-process verifier so
structural divergences are detected automatically, not by chance review. (Had
this existed, the T2/T3 `conversation`-fork divergence — BUG-FORK-CONVERSATION-
MISSING-T2T3 — would have alerted at the first cycle.)

Three layers per pass, all READ-ONLY against `data/timechain/index.db`
(NEVER constructs a TimeChain — that would trigger the idempotent reseed write
and violate the side-effect-free check() contract):

  Layer "fork_completeness" (auto-healable) — every chain must have all 6
    primary forks (main/declarative/procedural/episodic/meta/conversation) and
    each must resolve BY NAME via fork_registry to a real row (directly enforces
    INV-Syn-26 + catches the T2/T3 class). status OK / DEGRADED. On a missing
    primary, heal_recommended=True → HEAL_REQUEST(action="reseed_primary_fork",
    dst="timechain_worker") → the idempotent additive reseed (the ONE self-heal
    we trust — purely additive, never touches existing data).

  Layer "chain_integrity" (ALERT-ONLY) — bounded tip + structural consistency:
    every fork_registry row claiming tip_height >= 0 must have a chain file that
    exists and is larger than one header. Detects truncation / orphaned tips.
    NEVER auto-mutates sovereign chain data; deep prev-hash/merkle/PoT walks are
    delegated to the on-demand `/v6/timechain/verify` + `ChainIntegrity` repair
    engine (those exceed the 30s check() budget on large forks). status OK/DOWN.

  Layer "block_inclusion" (ALERT-ONLY) — fork-registry hygiene: no duplicate
    fork_names (would break name-resolution), no primary fork sharing an id.
    status OK / DEGRADED.

SPEC anchor: §9.B health_monitor_worker block (timechain_guardian) + INV-Syn-26
+ synthesis arch §5.1/§7/§13.2 (fork model).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from titan_hcl.health import (
    HealthCheckPlugin,
    HealthResult,
)

logger = logging.getLogger("health.timechain_guardian")

# The 6 reserved primary forks every chain MUST carry, resolvable by NAME.
PRIMARY_FORK_NAMES = ("main", "declarative", "procedural", "episodic", "meta",
                      "conversation")
_DEFAULT_TIMECHAIN_DIR = "data/timechain"
_HEADER_SIZE = 128  # mirrors timechain.HEADER_SIZE (read-only constant)


class TimechainGuardianHealthCheck(HealthCheckPlugin):
    """Continual verification daemon for the TimeChain structural invariants."""

    name = "timechain_guardian"
    applies_on = "all"
    # The worker that owns the TimeChain instance + executes the additive
    # reseed on a HEAL_REQUEST. MUST match the bus subscriber name.
    owning_worker = "timechain_worker"

    # Low cadence — dream-boundary scale, off the hot path. A divergence is
    # structural (only changes on a chain rebuild / genesis-era artifact), so
    # hourly is ample. Overridable via [health_monitor.timechain_guardian].
    cadence_s = 3600.0

    # Reseed is idempotent + additive, but still gate to avoid log spam on a
    # genuinely un-healable state (e.g. read-only filesystem).
    max_heal_attempts_per_24h = 4
    heal_cooldown_after_success_s = 3600.0
    heal_cooldown_after_failure_s = 1800.0

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        db_path = (self.config.get("index_db_path")
                   or _DEFAULT_TIMECHAIN_DIR + "/index.db")
        if not Path(db_path).is_absolute():
            db_path = str((Path(__file__).resolve().parents[2] / db_path))
        self._db_path = db_path
        self._data_dir = Path(self._db_path).parent

    # ── check() ───────────────────────────────────────────────────────

    def check(self) -> list[HealthResult]:
        if not Path(self._db_path).exists():
            # No chain yet (pre-genesis boot) — DEGRADED, not healable here.
            return [HealthResult(
                plugin=self.name, layer="fork_completeness", status="DEGRADED",
                reason="index_db_missing", details={"db_path": self._db_path},
                heal_recommended=False)]
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            try:
                rows = conn.execute(
                    "SELECT fork_id, fork_name, fork_type, tip_height "
                    "FROM fork_registry"
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error as e:
            return [HealthResult(
                plugin=self.name, layer="chain_integrity", status="DOWN",
                reason=f"index_db_error:{type(e).__name__}",
                details={"exception": str(e)[:200]}, heal_recommended=False)]

        return [
            self._check_fork_completeness(rows),
            self._check_chain_integrity(rows),
            self._check_block_inclusion(rows),
        ]

    def heal(self, last_result: HealthResult) -> tuple[str | None, dict]:
        # Only fork_completeness is auto-healable, and only when a primary is
        # actually missing. Integrity / inclusion failures are ALERT-ONLY —
        # sovereign chain data is never auto-mutated (directive_memory_preservation).
        if (last_result.layer == "fork_completeness"
                and last_result.heal_recommended):
            return "reseed_primary_fork", {
                "trigger": "missing_primary_fork",
                "missing": last_result.details.get("missing", []),
                "reason": last_result.reason,
            }
        return None, {}

    # ── Layers ──────────────────────────────────────────────────────────

    def _check_fork_completeness(self, rows) -> HealthResult:
        name_to_id: dict[str, int] = {r[1]: r[0] for r in rows}
        missing = [n for n in PRIMARY_FORK_NAMES if n not in name_to_id]
        ok = not missing
        return HealthResult(
            plugin=self.name, layer="fork_completeness",
            status="OK" if ok else "DEGRADED",
            reason=("all_6_primaries_name_resolvable" if ok
                    else f"missing_primaries={missing}"),
            details={
                "missing": missing,
                "resolved": {n: name_to_id.get(n) for n in PRIMARY_FORK_NAMES},
            },
            # Missing primary → reseed (additive self-heal). This is the layer
            # that would have FLAGGED + auto-fixed the T2/T3 conversation fork.
            heal_recommended=not ok,
        )

    def _check_chain_integrity(self, rows) -> HealthResult:
        """Bounded tip/structural consistency (alert-only). Deep merkle/PoT
        verification is delegated to /v6/timechain/verify + ChainIntegrity."""
        truncated = []
        for fork_id, fork_name, _ftype, tip_height in rows:
            if tip_height is None or tip_height < 0:
                continue  # no blocks committed yet — nothing to verify
            path = self._chain_file(fork_id, fork_name)
            try:
                if not path.exists() or path.stat().st_size <= _HEADER_SIZE:
                    truncated.append({"fork_id": fork_id, "fork_name": fork_name,
                                      "tip_height": tip_height,
                                      "file": path.name})
            except OSError:
                truncated.append({"fork_id": fork_id, "fork_name": fork_name,
                                  "tip_height": tip_height, "file": path.name})
        ok = not truncated
        return HealthResult(
            plugin=self.name, layer="chain_integrity",
            status="OK" if ok else "DOWN",
            reason=("tips_consistent_with_files" if ok
                    else f"truncated_or_missing_chain_files={len(truncated)}"),
            details={"offenders": truncated,
                     "note": "deep merkle/PoT walk delegated to "
                             "/v6/timechain/verify (exceeds 30s check budget)"},
            heal_recommended=False,  # ALERT-ONLY — never auto-mutate chain data
        )

    def _check_block_inclusion(self, rows) -> HealthResult:
        """Fork-registry hygiene (alert-only): unique names, no shared ids."""
        names = [r[1] for r in rows]
        ids = [r[0] for r in rows]
        dup_names = sorted({n for n in names if names.count(n) > 1})
        dup_ids = sorted({i for i in ids if ids.count(i) > 1})
        ok = not dup_names and not dup_ids
        return HealthResult(
            plugin=self.name, layer="block_inclusion",
            status="OK" if ok else "DEGRADED",
            reason=("registry_consistent" if ok
                    else f"duplicate_names={dup_names} duplicate_ids={dup_ids}"),
            details={"duplicate_fork_names": dup_names,
                     "duplicate_fork_ids": dup_ids},
            heal_recommended=False,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _chain_file(self, fork_id: int, fork_name: str) -> Path:
        """EXACT mirror of TimeChain._get_chain_file_path so the integrity layer
        probes the real on-disk file (never false-alarms).

        That function keys the chain-file name off the canonical FORK_NAMES
        id→name map, NOT the registry name: a primary at its canonical id →
        chain_<name>.bin; anything else (incl. a `conversation` reseeded at a
        relocated id) → sidechains/sc_NNNN.bin. We import FORK_NAMES (a plain
        constant — no TimeChain construction, so check() stays side-effect-free).
        """
        from titan_hcl.logic.timechain import FORK_NAMES
        canonical_name = FORK_NAMES.get(fork_id)
        if canonical_name:
            return self._data_dir / f"chain_{canonical_name}.bin"
        return self._data_dir / "sidechains" / f"sc_{fork_id:04d}.bin"


__all__ = ("TimechainGuardianHealthCheck",)

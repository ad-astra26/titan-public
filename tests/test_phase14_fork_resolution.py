"""Phase 14 / INV-Syn-26 — chain-local fork resolution + idempotent reseed.

Covers:
  1. Idempotent additive reseed materializes a missing primary at the next
     free id when its canonical id is occupied by a non-primary, WITHOUT
     clobbering the occupant (the T2/T3 `conversation` fork class of bug).
  2. resolve_fork_id is a verified no-op on a fresh chain (T1 regression guard
     — returns exactly the canonical FORK_* ids).
  3. All 6 primaries are name-resolvable on every chain (the §3K.6 invariant).
  4. Grep guard: no fork-int resolution by hardcoded map outside the resolver
     in the choke-point modules.

Run: python -m pytest tests/test_phase14_fork_resolution.py -v -p no:anchorpy
"""
import sqlite3
import pathlib

import pytest

from titan_hcl.logic.timechain import (
    TimeChain, FORK_NAMES, FORK_CONVERSATION, FORK_SIDECHAIN_START,
    FORK_MAIN, FORK_DECLARATIVE, FORK_PROCEDURAL, FORK_EPISODIC, FORK_META,
)

PRIMARY_NAMES = ["main", "declarative", "procedural", "episodic", "meta",
                 "conversation"]


def _new_chain(tmp_path, titan_id="TZ"):
    tc = TimeChain(data_dir=str(tmp_path), titan_id=titan_id)
    tc.create_genesis({"maker_pubkey": "test", "soul_hash": "0" * 64})
    return tc


def test_fresh_chain_is_canonical_noop(tmp_path):
    """T1 regression guard — a freshly-born chain resolves every primary to its
    canonical FORK_* id. The resolver MUST be a no-op where ids already match."""
    tc = _new_chain(tmp_path)
    assert tc.resolve_fork_id("main") == FORK_MAIN == 0
    assert tc.resolve_fork_id("declarative") == FORK_DECLARATIVE == 1
    assert tc.resolve_fork_id("procedural") == FORK_PROCEDURAL == 2
    assert tc.resolve_fork_id("episodic") == FORK_EPISODIC == 3
    assert tc.resolve_fork_id("meta") == FORK_META == 4
    assert tc.resolve_fork_id("conversation") == FORK_CONVERSATION == 5
    # Reverse direction agrees.
    assert tc.resolve_fork_name(5) == "conversation"
    assert tc.resolve_fork_name(0) == "main"
    # Unknown name → None (caller decides; NEVER a hardcoded substitute).
    assert tc.resolve_fork_id("does_not_exist") is None


def test_all_six_primaries_name_resolvable(tmp_path):
    tc = _new_chain(tmp_path)
    for name in PRIMARY_NAMES:
        assert tc.resolve_fork_id(name) is not None, f"{name} not resolvable"


def test_reseed_relocates_when_canonical_id_occupied(tmp_path):
    """Simulate a T2/T3 chain born before FORK_CONVERSATION=5 existed: a
    sidechain (`topic:expression`) holds id 5 and there is no `conversation`
    fork. Re-opening the chain must (a) materialize `conversation` at the next
    FREE id, (b) leave `topic:expression` untouched at id 5 (sovereign data)."""
    tc = _new_chain(tmp_path)
    db = pathlib.Path(tmp_path) / "index.db"

    # Recreate the genesis-era divergence: drop the conversation primary and
    # seat a topic sidechain at id 5.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM fork_registry WHERE fork_id = 5")
    conn.execute(
        "INSERT INTO fork_registry (fork_id, fork_name, fork_type, parent_fork, "
        "parent_block, created_at, tip_height, tip_hash, topic, compacted) "
        "VALUES (5, 'topic:expression', 'topic_sidechain', 3, 0, 0, -1, NULL, "
        "'expression', 0)"
    )
    conn.commit()
    conn.close()

    # Re-open — __init__ runs the idempotent additive reseed.
    tc2 = TimeChain(data_dir=str(tmp_path), titan_id="TZ")

    # (a) conversation now exists, at a free id that is NOT 5.
    conv_id = tc2.resolve_fork_id("conversation")
    assert conv_id is not None
    assert conv_id != 5
    assert conv_id >= FORK_SIDECHAIN_START

    # (b) the occupant is untouched — id 5 is still topic:expression.
    assert tc2.resolve_fork_name(5) == "topic:expression"
    assert tc2.resolve_fork_id("topic:expression") == 5

    # All six primaries resolvable post-reseed.
    for name in PRIMARY_NAMES:
        assert tc2.resolve_fork_id(name) is not None, f"{name} missing post-reseed"

    # Idempotent: a third open does not move conversation again.
    tc3 = TimeChain(data_dir=str(tmp_path), titan_id="TZ")
    assert tc3.resolve_fork_id("conversation") == conv_id


def test_reseed_is_purely_additive_no_occupant_mutation(tmp_path):
    """The reseed must never UPDATE/DELETE an existing fork_registry row."""
    tc = _new_chain(tmp_path)
    db = pathlib.Path(tmp_path) / "index.db"
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM fork_registry WHERE fork_id = 5")
    conn.execute(
        "INSERT INTO fork_registry (fork_id, fork_name, fork_type, parent_fork, "
        "parent_block, created_at, tip_height, tip_hash, topic, compacted) "
        "VALUES (5, 'topic:expression', 'topic_sidechain', 3, 7, 123.0, 42, NULL, "
        "'expression', 0)"
    )
    conn.commit()
    before = conn.execute(
        "SELECT fork_id, fork_name, fork_type, parent_block, created_at, "
        "tip_height FROM fork_registry WHERE fork_id = 5"
    ).fetchone()
    conn.close()

    TimeChain(data_dir=str(tmp_path), titan_id="TZ")

    conn = sqlite3.connect(str(db))
    after = conn.execute(
        "SELECT fork_id, fork_name, fork_type, parent_block, created_at, "
        "tip_height FROM fork_registry WHERE fork_id = 5"
    ).fetchone()
    conn.close()
    assert before == after, "reseed mutated the id-5 occupant — sovereign-data violation"


def test_no_hardcoded_fork_resolution_outside_resolver():
    """Grep guard (§3K.3.D) — the choke-point modules must not resolve a fork
    id via the static FORK_IDS / fork_id_map / FORK_CONVERSATION constant."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    forbidden = {
        "titan_hcl/modules/timechain_worker.py": ["fork_id_map", "FORK_CONVERSATION", "FORK_IDS"],
        "titan_hcl/api/dashboard.py": ["FORK_CONVERSATION"],
    }
    offenders = []
    for rel, patterns in forbidden.items():
        text = (repo / rel).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat in patterns:
                if pat in line:
                    offenders.append(f"{rel}: {stripped}")
    assert not offenders, "hardcoded fork resolution remains:\n" + "\n".join(offenders)


def test_timechain_v2_fork_ids_is_seed_only():
    """timechain_v2.FORK_IDS may exist only as a seed-default definition — never
    referenced for resolution (resolution goes through _resolve_fork_id)."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "titan_hcl/logic/timechain_v2.py").read_text()
    bad = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("FORK_IDS = {") or s.startswith('"main":'):
            continue
        # Any FORK_IDS[...] or FORK_IDS.get(...) read is forbidden.
        if "FORK_IDS[" in line or "FORK_IDS.get(" in line:
            bad.append(s)
    assert not bad, "timechain_v2 still resolves via FORK_IDS:\n" + "\n".join(bad)


# ── Timechain Guardian (§3K.7) ──────────────────────────────────────────

def _guardian(tmp_path):
    from titan_hcl.health.timechain_guardian import TimechainGuardianHealthCheck
    return TimechainGuardianHealthCheck(
        config={"index_db_path": str(pathlib.Path(tmp_path) / "index.db")})


def test_guardian_clean_chain_all_ok(tmp_path):
    """On a healthy chain (all 6 primaries present) the completeness layer is
    OK and recommends no heal."""
    _new_chain(tmp_path)
    results = _guardian(tmp_path).check()
    by_layer = {r.layer: r for r in results}
    assert by_layer["fork_completeness"].status == "OK"
    assert by_layer["fork_completeness"].heal_recommended is False
    assert by_layer["block_inclusion"].status == "OK"


def test_guardian_flags_missing_conversation_and_heals(tmp_path):
    """Pre-reseed T2/T3 state (no conversation fork, id 5 = topic:expression)
    must FLAG fork_completeness DEGRADED + recommend the reseed heal — proving
    the guardian catches the BUG-FORK-CONVERSATION-MISSING-T2T3 class."""
    _new_chain(tmp_path)
    db = pathlib.Path(tmp_path) / "index.db"
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM fork_registry WHERE fork_id = 5")
    conn.execute(
        "INSERT INTO fork_registry (fork_id, fork_name, fork_type, parent_fork, "
        "parent_block, created_at, tip_height, tip_hash, topic, compacted) "
        "VALUES (5, 'topic:expression', 'topic_sidechain', 3, 0, 0, -1, NULL, "
        "'expression', 0)"
    )
    conn.commit()
    conn.close()

    g = _guardian(tmp_path)
    results = g.check()
    completeness = next(r for r in results if r.layer == "fork_completeness")
    assert completeness.status == "DEGRADED"
    assert completeness.heal_recommended is True
    assert "conversation" in completeness.details["missing"]

    # heal() returns the reseed descriptor targeting timechain_worker.
    action, details = g.heal(completeness)
    assert action == "reseed_primary_fork"
    assert "conversation" in details["missing"]


def test_guardian_is_side_effect_free(tmp_path):
    """check() must NOT mutate the chain (no reseed, no row changes)."""
    _new_chain(tmp_path)
    db = pathlib.Path(tmp_path) / "index.db"
    conn = sqlite3.connect(str(db))
    before = conn.execute("SELECT fork_id, fork_name FROM fork_registry "
                          "ORDER BY fork_id").fetchall()
    conn.close()

    _guardian(tmp_path).check()

    conn = sqlite3.connect(str(db))
    after = conn.execute("SELECT fork_id, fork_name FROM fork_registry "
                         "ORDER BY fork_id").fetchall()
    conn.close()
    assert before == after, "guardian check() mutated fork_registry"


def test_guardian_discovered_by_registry():
    """health_monitor_worker's pkgutil discovery must find the new plugin."""
    import importlib
    import pkgutil
    import titan_hcl.health as health_pkg
    from titan_hcl.health import HealthCheckPlugin
    found = []
    for _f, mod_name, _is in pkgutil.iter_modules(health_pkg.__path__):
        mod = importlib.import_module(f"titan_hcl.health.{mod_name}")
        for attr in vars(mod).values():
            if (isinstance(attr, type) and issubclass(attr, HealthCheckPlugin)
                    and attr is not HealthCheckPlugin):
                found.append(attr.name)
    assert "timechain_guardian" in found

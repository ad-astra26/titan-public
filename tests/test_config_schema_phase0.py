"""Phase 0 acceptance tests for the canonical config schema
(RFP_config_as_shm_state_canonical_integrity §7.0.3 / SPEC §14.A).

Durable invariants (re-checkable any session, no /tmp baseline needed):
  - schema parses
  - params.toml ∩ config.toml = ∅  (disjoint key-spaces, INV-CFG-6)
  - schema covers 100% of keys in BOTH TOMLs; no schema entry without a TOML key
  - every schema `owner` is a real registered module (or "kernel")
  - every stanza has the full {type, constraint, reload, owner, sensitive} shape

The one-time migration-parity check (post-revision merge == pre-revision 4-layer
merge) lives in scripts/config_phase0_parity.py — it was run green at migration
time (baseline is pre-revision and gone after the merge), so it is not a durable
regression test.

No TitanHCL instance is created, so these run fine in a shared pytest process.
"""
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PARAMS_PATH = REPO / "titan_hcl" / "titan_params.toml"
SCHEMA_PATH = REPO / "titan_hcl" / "config_schema.toml"

# config.toml is gitignored/per-box — resolve from the running box's main repo.
_CONFIG_CANDIDATES = [
    REPO / "titan_hcl" / "config.toml",
    Path("/home/youruser/projects/titan/titan_hcl/config.toml"),
    Path("/home/youruser/projects/titan3/titan_hcl/config.toml"),
]
CONFIG_PATH = next((p for p in _CONFIG_CANDIDATES if p.exists()), None)

# Canonical registered modules (titan_hcl/module_catalog.py ModuleSpec names) + kernel.
MODULE_NAMES = {
    "agency_worker", "agno_worker", "api", "backup", "body", "cgn",
    "cognitive_worker", "corrective_events_persistence", "dream_state",
    "emot_cgn", "expression_worker", "felt_teaching", "health_monitor",
    "hormonal_module", "imw", "interface_advisor", "journey_persistence",
    "knowledge", "language", "life_force", "llm", "media", "meditation",
    "memory", "meta_teacher", "metabolism", "mind", "neuromod_module",
    "ns_module", "observatory", "observatory_writer", "outer_interface_worker",
    "pattern_logic",
    "output_verifier", "reflex", "self_learning", "self_reflection_worker",
    "social_graph", "social_worker", "soul_diary", "sovereignty", "studio",
    "synthesis", "timechain", "warning_monitor", "kernel",
}
STANZA_FIELDS = {"type", "constraint", "reload", "owner", "sensitive"}


def _flatten(d, prefix=""):
    """Flatten nested TOML to dotted keys; array-of-tables -> <key>[] (shape once)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list) and v and all(isinstance(e, dict) for e in v):
            for e in v:
                out.update(_flatten(e, f"{key}[]"))
        else:
            out[key] = v
    return out


def _load(p):
    with open(p, "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def schema():
    return _load(SCHEMA_PATH)


@pytest.fixture(scope="module")
def params_keys():
    return set(_flatten(_load(PARAMS_PATH)))


@pytest.fixture(scope="module")
def config_keys():
    if CONFIG_PATH is None:
        pytest.skip("config.toml not found on this box (gitignored/per-box)")
    return set(_flatten(_load(CONFIG_PATH)))


def test_schema_parses(schema):
    assert "config" in schema and "params" in schema
    assert len(schema["config"]) > 100 and len(schema["params"]) > 1000


def test_disjoint_key_spaces(params_keys, config_keys):
    """INV-CFG-6: params.toml and config.toml share NO key."""
    overlap = params_keys & config_keys
    assert not overlap, f"params∩config must be empty; got {sorted(overlap)}"


def test_schema_covers_all_keys(schema, params_keys, config_keys):
    """Every TOML key has a schema entry; no schema entry without a TOML key."""
    sch_params = set(schema["params"])
    sch_config = set(schema["config"])

    # Keys declared in the schema but not present in THIS box's two TOMLs:
    #  - secrets-only keys live in ~/.titan/secrets.toml (no params/config placeholder)
    #  - per-box keys present on other boxes' config.toml (e.g. T2/T3's [genesis])
    # The schema is fleet-wide, so these are legitimately "orphan" relative to one box.
    fleet_extra = {"synthesis.user_id_hash_salt", "genesis.titan_name",
                   # per-Titan synthesis canary flags — only present in canary boxes'
                   # config.toml (folded from the retired microkernel override, §7-Phase-B(6))
                   "synthesis.fork_gc_live", "synthesis.recall.augment_chat"}

    missing_p = params_keys - sch_params
    missing_c = config_keys - sch_config
    orphan_p = sch_params - params_keys - fleet_extra
    orphan_c = sch_config - config_keys - fleet_extra

    assert not missing_p, f"params keys missing a schema entry: {sorted(missing_p)[:20]}"
    assert not missing_c, f"config keys missing a schema entry: {sorted(missing_c)[:20]}"
    assert not orphan_p, f"schema [params] entries with no TOML key: {sorted(orphan_p)[:20]}"
    assert not orphan_c, f"schema [config] entries with no TOML key: {sorted(orphan_c)[:20]}"


def test_every_owner_is_a_real_module(schema):
    bad = {}
    for ns in ("config", "params"):
        for key, stanza in schema[ns].items():
            for owner in stanza["owner"]:
                if owner not in MODULE_NAMES:
                    bad.setdefault(owner, []).append(f"{ns}::{key}")
    assert not bad, f"owners not in the registered-module set: { {k: v[:3] for k, v in bad.items()} }"


def test_every_stanza_has_full_shape(schema):
    for ns in ("config", "params"):
        for key, stanza in schema[ns].items():
            assert set(stanza) == STANZA_FIELDS, f"{ns}::{key} fields = {set(stanza)}"
            assert stanza["reload"] in ("hot", "restart_required"), f"{ns}::{key} reload={stanza['reload']}"
            assert stanza["type"] in ("bool", "int", "float", "str", "list"), f"{ns}::{key} type={stanza['type']}"
            assert isinstance(stanza["owner"], list) and stanza["owner"], f"{ns}::{key} empty owner"
            assert isinstance(stanza["sensitive"], bool), f"{ns}::{key} sensitive not bool"

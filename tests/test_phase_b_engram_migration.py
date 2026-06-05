"""Phase B — Concept→Engram Kuzu copy-migration tests (RFP_synthesis_engram_grounding §7.B / G2).

Builds a graph on the LEGACY (pre-Phase-B) `Concept` schema, runs
`migrate_concept_to_engram`, and asserts: node count preserved, all 4 spine
edge types preserved, the 5 new columns present (axes 0.0 / domain_hint NULL),
`Concept` dropped, `Engram` present, `COMPILED_FROM` (non-spine rel) untouched,
plus idempotency + the fresh-install guard.
"""
import pytest

kuzu = pytest.importorskip("kuzu")

from titan_hcl.synthesis.kuzu_spine_schema import (  # noqa: E402
    _count_nodes,
    _table_exists,
    migrate_concept_to_engram,
)

# The pre-Phase-B Concept spine schema — what an existing Titan has on disk.
_LEGACY_NODES = {
    "Concept": (
        "pk STRING, concept_id STRING, version INT64, name STRING, "
        "memory_type STRING, groundedness DOUBLE, anchor_tx STRING, "
        "created_at DOUBLE, PRIMARY KEY(pk)"
    ),
    "Production": (
        "skill_id STRING, name STRING, utility_score DOUBLE, anchor_tx STRING, "
        "PRIMARY KEY(skill_id)"
    ),
    "ActionChain": (
        "chain_id STRING, shape STRING, success_count INT64, failure_count INT64, "
        "PRIMARY KEY(chain_id)"
    ),
    "HypothesisFork": (
        "fork_id STRING, root_anchor STRING, activation DOUBLE, status STRING, "
        "PRIMARY KEY(fork_id)"
    ),
}
_LEGACY_RELS = (
    ("COMPOSED_FROM", "Concept", "Concept"),
    ("COMPOSED_INTO", "Concept", "Concept"),
    ("USES_SKILL", "Concept", "Production"),
    ("COMPILED_FROM", "Production", "ActionChain"),
    ("EXPLORES", "HypothesisFork", "Concept"),
)


class _RawGraph:
    """Minimal shim exposing `_conn` + `_db_path` (what migrate() reads)."""

    def __init__(self, path):
        self._db_path = path
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)


def _make_legacy_graph(path):
    g = _RawGraph(path)
    for name, schema in _LEGACY_NODES.items():
        g._conn.execute(f"CREATE NODE TABLE {name}({schema})")
    for rel, s, d in _LEGACY_RELS:
        g._conn.execute(f"CREATE REL TABLE {rel}(FROM {s} TO {d})")
    return g


def _insert_concept(conn, cid, ver=1, name="X", mt="declarative", g=0.2):
    pk = f"{cid}:v{ver}"
    conn.execute(
        "CREATE (c:Concept {pk:$pk, concept_id:$cid, version:$ver, name:$name, "
        "memory_type:$mt, groundedness:$g, anchor_tx:$tx, created_at:$ca})",
        {"pk": pk, "cid": cid, "ver": ver, "name": name, "mt": mt, "g": g,
         "tx": "tx_" + cid, "ca": 1.0},
    )
    return pk


def test_migration_preserves_nodes_edges_and_adds_axes(tmp_path):
    g = _make_legacy_graph(str(tmp_path / "spine.kuzu"))
    c = g._conn
    a = _insert_concept(c, "alpha")
    b = _insert_concept(c, "beta")
    c.execute("CREATE (p:Production {skill_id:'sk1', name:'S', utility_score:0.5, anchor_tx:'t'})")
    c.execute("CREATE (f:HypothesisFork {fork_id:'fk1', root_anchor:'r', activation:0.1, status:'open'})")
    c.execute("MATCH (x:Concept {pk:$a}),(y:Concept {pk:$b}) CREATE (x)-[:COMPOSED_FROM]->(y)", {"a": a, "b": b})
    c.execute("MATCH (x:Concept {pk:$a}),(p:Production {skill_id:'sk1'}) CREATE (x)-[:USES_SKILL]->(p)", {"a": a})
    c.execute("MATCH (f:HypothesisFork {fork_id:'fk1'}),(y:Concept {pk:$b}) CREATE (f)-[:EXPLORES]->(y)", {"b": b})

    out = migrate_concept_to_engram(g)

    assert out["migrated"] is True and out["reason"] == "ok"
    assert out["nodes"] == 2
    assert out["edges"] == 3  # COMPOSED_FROM + USES_SKILL + EXPLORES
    # Concept gone, Engram present with no node loss (G2).
    assert not _table_exists(c, "Concept")
    assert _table_exists(c, "Engram")
    assert _count_nodes(c, "Engram") == 2
    # Non-spine rel left untouched.
    assert _table_exists(c, "COMPILED_FROM")
    # New columns: axes default 0.0, domain_hint NULL; legacy data intact.
    qr = c.execute(
        "MATCH (e:Engram {concept_id:'alpha'}) RETURN e.axis_used, e.axis_verified, "
        "e.axis_felt, e.axis_fluent, e.domain_hint, e.name, e.groundedness")
    row = qr.get_next()
    assert row[0] == 0.0 and row[1] == 0.0 and row[2] == 0.0 and row[3] == 0.0
    assert row[4] is None
    assert row[5] == "X" and abs(row[6] - 0.2) < 1e-9
    # Each spine edge type preserved on Engram.
    for q in (
        "MATCH (:Engram)-[r:COMPOSED_FROM]->(:Engram) RETURN COUNT(r)",
        "MATCH (:Engram)-[r:USES_SKILL]->(:Production) RETURN COUNT(r)",
        "MATCH (:HypothesisFork)-[r:EXPLORES]->(:Engram) RETURN COUNT(r)",
    ):
        assert int(c.execute(q).get_next()[0]) == 1


def test_migration_idempotent(tmp_path):
    g = _make_legacy_graph(str(tmp_path / "spine.kuzu"))
    _insert_concept(g._conn, "alpha")
    first = migrate_concept_to_engram(g)
    assert first["migrated"] is True and first["nodes"] == 1
    second = migrate_concept_to_engram(g)
    assert second["migrated"] is False and second["reason"] == "engram_exists"
    assert _count_nodes(g._conn, "Engram") == 1


def test_migration_fresh_install_no_concept_is_noop(tmp_path):
    g = _RawGraph(str(tmp_path / "fresh.kuzu"))  # no Concept table at all
    out = migrate_concept_to_engram(g)
    assert out["migrated"] is False and out["reason"] == "no_concept_fresh"


def test_migration_empty_concept_trivial(tmp_path):
    """knowledge_graph.kuzu case: a Concept table with zero rows migrates to an
    empty Engram (0 nodes, 0 edges) and drops Concept — no snapshot needed."""
    g = _make_legacy_graph(str(tmp_path / "empty.kuzu"))
    out = migrate_concept_to_engram(g)
    assert out["migrated"] is True and out["nodes"] == 0 and out["edges"] == 0
    assert out["snapshot"] is None  # no rows → no snapshot copy
    assert not _table_exists(g._conn, "Concept")
    assert _table_exists(g._conn, "Engram")

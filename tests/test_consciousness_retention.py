"""Consciousness retention (2026-06-18): bound `epochs` (1.23M) + `trinity_journey_gifts`
(676k) — the 2.7GB consciousness.db leak. epochs retention is KEEP-MEANINGFUL: the
recent window + anchored (Solana-committed) + high-curvature epochs survive forever;
only the routine bulk is pruned. gifts keep a recent window (read-windowed only).
"""
import sqlite3

_EPOCHS_PRUNE = (
    "DELETE FROM epochs "
    "WHERE epoch_id <= (SELECT MAX(epoch_id) FROM epochs) - ? "
    "AND (anchored_tx IS NULL OR anchored_tx = '') AND curvature <= 3.0")
_GIFTS_PRUNE = (
    "DELETE FROM trinity_journey_gifts "
    "WHERE gift_id <= (SELECT MAX(gift_id) FROM trinity_journey_gifts) - ?")


def test_epochs_prune_preserves_anchored_and_high_curvature(tmp_path):
    c = sqlite3.connect(str(tmp_path / "c.db"))
    c.execute("CREATE TABLE epochs (epoch_id INTEGER PRIMARY KEY, curvature REAL, "
              "anchored_tx TEXT)")
    for i in range(1, 1001):  # 1000 routine epochs (low curvature, not anchored)
        c.execute("INSERT INTO epochs VALUES (?, ?, ?)", (i, 1.0, ""))
    c.execute("UPDATE epochs SET anchored_tx='sig123' WHERE epoch_id=5")   # old, anchored
    c.execute("UPDATE epochs SET curvature=4.0 WHERE epoch_id=10")          # old, high-curv
    c.commit()

    c.execute(_EPOCHS_PRUNE, (100,))  # keep last 100 + meaningful
    c.commit()
    ids = {r[0] for r in c.execute("SELECT epoch_id FROM epochs").fetchall()}

    assert 5 in ids and 10 in ids          # meaningful OLD epochs preserved
    assert {901, 1000}.issubset(ids)       # recent window kept
    assert 500 not in ids                  # routine old pruned
    assert sum(1 for i in ids if i >= 901) == 100  # exactly the recent window
    c.close()


def test_epochs_prune_noop_when_small(tmp_path):
    c = sqlite3.connect(str(tmp_path / "c.db"))
    c.execute("CREATE TABLE epochs (epoch_id INTEGER PRIMARY KEY, curvature REAL, "
              "anchored_tx TEXT)")
    for i in range(1, 11):
        c.execute("INSERT INTO epochs VALUES (?, ?, ?)", (i, 1.0, ""))
    c.commit()
    c.execute(_EPOCHS_PRUNE, (50_000,))  # cutoff negative → nothing pruned
    c.commit()
    assert c.execute("SELECT COUNT(*) FROM epochs").fetchone()[0] == 10
    c.close()


def test_journey_gifts_prune_keeps_recent_window(tmp_path):
    c = sqlite3.connect(str(tmp_path / "c.db"))
    c.execute("CREATE TABLE trinity_journey_gifts (gift_id INTEGER PRIMARY KEY)")
    for i in range(1, 501):
        c.execute("INSERT INTO trinity_journey_gifts VALUES (?)", (i,))
    c.commit()
    c.execute(_GIFTS_PRUNE, (100,))
    c.commit()
    lo, hi = c.execute("SELECT MIN(gift_id), MAX(gift_id) FROM "
                       "trinity_journey_gifts").fetchone()
    assert (lo, hi) == (401, 500)  # most-recent 100
    c.close()

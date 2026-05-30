"""PROFILING.md — TimeChain read-only instances skip the O(all-blocks) tag scan.

`_load_fork_state` scanned every block's tags (via eval) at every __init__ to
build `_tag_counts`, which is consumed ONLY by `_track_tags_for_sidechain`
(gated on `self._auto_sidechain`). For read-only / auto_sidechain=False
instances (e.g. the api's cached TimeChain, rebuilt every 60s) that scan built a
dict that was never read — it measured ~34% of one core in titan_hcl_api.

Verifies:
  1. auto_sidechain=True still builds _tag_counts (write instance unaffected).
  2. auto_sidechain=False skips the scan (_tag_counts empty) BUT still builds
     the _fork_name_to_id index that read queries depend on.
  3. The re-scan parses the stored `str(list)` tags via ast.literal_eval
     (no eval), so the counts round-trip correctly.
"""
from titan_hcl.logic.timechain import (
    BlockPayload, TimeChain, FORK_DECLARATIVE,
)


def _payload(tags):
    return BlockPayload(
        thought_type="declarative", source="teacher",
        content={"word": "test", "confidence": 0.8},
        significance=0.5, confidence=0.7, tags=tags, db_ref="vocabulary:42")


def _neuromods():
    return {"DA": 0.5, "ACh": 0.5, "NE": 0.5, "5HT": 0.5, "GABA": 0.3,
            "endorphin": 0.5}


def _commit(tc, tags, epoch):
    tc.commit_block(fork_id=FORK_DECLARATIVE, epoch_id=epoch,
                    payload=_payload(tags), pot_nonce=1, chi_spent=0.005,
                    neuromod_state=_neuromods())


def test_readonly_skips_tag_scan_keeps_fork_index(tmp_path):
    dd = str(tmp_path / "tc")

    # 1. write instance — adds tagged blocks → _tag_counts populated.
    tc = TimeChain(data_dir=dd, titan_id="T_TEST", auto_sidechain=True)
    _commit(tc, ["alpha", "beta"], 100)
    _commit(tc, ["alpha"], 101)
    assert tc._tag_counts.get(FORK_DECLARATIVE, {}).get("alpha", 0) >= 1

    # 2. read-only instance on the SAME data — scan skipped, but fork index
    #    (which read queries / resolve_fork_id need) is still built.
    ro = TimeChain(data_dir=dd, titan_id="T_TEST", auto_sidechain=False)
    assert ro._tag_counts == {}, "read-only must skip the tag-count scan"
    assert ro._fork_name_to_id, "read-only must still build the fork name index"
    assert ro._total_blocks >= 2, "read-only must still count blocks"

    # 3. re-open as a write instance — the scan re-runs and ast.literal_eval
    #    parses the stored str(list) tags back into the counts.
    tc2 = TimeChain(data_dir=dd, titan_id="T_TEST", auto_sidechain=True)
    assert tc2._tag_counts.get(FORK_DECLARATIVE, {}).get("alpha", 0) >= 2
    assert tc2._tag_counts.get(FORK_DECLARATIVE, {}).get("beta", 0) >= 1

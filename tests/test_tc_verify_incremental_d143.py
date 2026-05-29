"""Incremental TimeChain verify (D-SPEC-143 F2) — verify_fork_incremental.

The tc-verify warmer was the api's #1 CPU consumer: it re-hashed the entire
~200 MB chain (verify_all → sha256 every fork) every 60s. F2 makes it
incremental: resume from the previously-verified tip, hash only appended
blocks. These tests pin: (1) incremental == full validity; (2) resume advances
the tip and only covers new blocks; (3) unchanged fork is cheap + still valid;
(4) full verify still catches tamper; (5) shrink/rewrite → valid=None fallback;
(6) verify_fork delegates unchanged.
"""
import os
import shutil
import tempfile

import pytest

from titan_hcl.logic.timechain import (
    TimeChain,
    FORK_DECLARATIVE,
    FORK_EPISODIC,
)
from titan_hcl.logic.timechain import BlockPayload


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="tc_incr_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def chain(tmp_dir):
    tc = TimeChain(data_dir=os.path.join(tmp_dir, "timechain"), titan_id="T_TEST")
    tc.create_genesis(
        genesis_content={"maker_pubkey": "mk", "soul_hash": "sh",
                         "prime_directives": ["X"], "born": "2026-04-05"},
        birth_timestamp=1712345678.0,
    )
    return tc


def _neuromods():
    return {"DA": 0.5, "ACh": 0.5, "NE": 0.5, "5HT": 0.5, "GABA": 0.2, "endorphin": 0.3}


def _payload(i):
    return BlockPayload(
        thought_type="declarative", source="teacher",
        content={"word": f"w{i}", "i": i}, significance=0.5,
        confidence=0.7, tags=[f"w{i}"], db_ref="vocabulary:42")


def _commit(chain, fork, n, start=0):
    for i in range(start, start + n):
        chain.commit_block(fork_id=fork, epoch_id=1000 + i, payload=_payload(i),
                            pot_nonce=1, chi_spent=0.005, neuromod_state=_neuromods())


def test_incremental_full_matches_verify_fork(chain):
    _commit(chain, FORK_DECLARATIVE, 5)
    full_valid, full_msg = chain.verify_fork(FORK_DECLARATIVE)
    inc_valid, inc_msg, resume = chain.verify_fork_incremental(FORK_DECLARATIVE, None)
    assert full_valid is True and inc_valid is True
    assert full_msg == inc_msg  # delegation is byte-identical
    assert resume is not None and resume["height"] == 5
    assert resume["offset"] > 0


def test_resume_only_covers_new_blocks(chain):
    _commit(chain, FORK_DECLARATIVE, 3)
    v1, _m1, r1 = chain.verify_fork_incremental(FORK_DECLARATIVE, None)
    assert v1 and r1["height"] == 3
    off1 = r1["offset"]
    # append 4 more
    _commit(chain, FORK_DECLARATIVE, 4, start=3)
    v2, _m2, r2 = chain.verify_fork_incremental(FORK_DECLARATIVE, r1)
    assert v2 is True
    assert r2["height"] == 7          # resumed from 3, added 4
    assert r2["offset"] > off1        # tip advanced
    # and it agrees with a from-scratch full verify
    vf, _mf, rf = chain.verify_fork_incremental(FORK_DECLARATIVE, None)
    assert vf is True and rf["height"] == 7 and rf["offset"] == r2["offset"]


def test_unchanged_fork_resume_is_valid_and_stable(chain):
    _commit(chain, FORK_DECLARATIVE, 3)
    _v, _m, r1 = chain.verify_fork_incremental(FORK_DECLARATIVE, None)
    # no new blocks → resume returns valid, same height + offset
    v2, _m2, r2 = chain.verify_fork_incremental(FORK_DECLARATIVE, r1)
    assert v2 is True
    assert r2["height"] == r1["height"]
    assert r2["offset"] == r1["offset"]


def test_full_verify_detects_tamper(chain):
    _commit(chain, FORK_EPISODIC, 4)
    path = chain._get_chain_file_path(FORK_EPISODIC)
    data = bytearray(open(path, "rb").read())
    data[-10] ^= 0xFF  # flip a byte in the last block's payload region
    open(path, "wb").write(bytes(data))
    valid, msg = chain.verify_fork(FORK_EPISODIC)
    assert valid is False
    assert "tamper" in msg.lower() or "mismatch" in msg.lower() or "break" in msg.lower()


def test_resume_offset_past_eof_signals_full_reverify(chain):
    _commit(chain, FORK_DECLARATIVE, 4)
    _v, _m, r = chain.verify_fork_incremental(FORK_DECLARATIVE, None)
    # simulate a file shrink/rewrite (reorg/compaction): truncate below tip
    path = chain._get_chain_file_path(FORK_DECLARATIVE)
    size = path.stat().st_size
    with open(path, "r+b") as f:
        f.truncate(size // 2)
    valid, msg, new_resume = chain.verify_fork_incremental(FORK_DECLARATIVE, r)
    assert valid is None  # NOT a tamper verdict — caller must full re-verify
    assert new_resume is None
    assert "re-verify" in msg.lower() or "eof" in msg.lower()


def test_malformed_resume_signals_full_reverify(chain):
    _commit(chain, FORK_DECLARATIVE, 2)
    valid, msg, _ = chain.verify_fork_incremental(
        FORK_DECLARATIVE, {"offset": "nope", "height": 1, "prev_hash": b"x"})
    assert valid is None


def test_empty_fork_resume_none(chain):
    # A fork with no blocks committed → True, no resume.
    valid, _msg, resume = chain.verify_fork_incremental(FORK_EPISODIC, None)
    assert valid is True
    assert resume is None  # file does not exist → nothing to resume

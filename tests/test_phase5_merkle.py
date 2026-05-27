"""Phase 5 — Merkle root helper tests.

Covers `titan_hcl/synthesis/merkle.py` against §P5.G + arch §11.2:

- empty input returns SHA-256(b"") (deterministic, proves "set was empty")
- single leaf duplicates → SHA-256(leaf || leaf) (keeps tree-walk uniform)
- two leaves: SHA-256(L0 || L1)
- three leaves: odd-count duplication on right
- four leaves: balanced tree
- determinism: same leaves → same root
- order-sensitivity: reorder → different root
- hex output: 64 chars, lowercase
"""
from __future__ import annotations

import hashlib

import pytest

from titan_hcl.synthesis.merkle import merkle_root_hex


def _hh(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()


def test_empty_input_is_sha256_of_empty_string():
    r = merkle_root_hex([])
    assert r == _hh(b"")
    assert len(r) == 64


def test_single_leaf_duplicates_for_uniform_tree_walk():
    """Standard convention: a 1-leaf tree's root is SHA-256(leaf || leaf),
    not the leaf itself — so a verifier's tree walk is uniform regardless
    of leaf count."""
    leaf = "a" * 64
    leaf_bytes = bytes.fromhex(leaf)
    expected = _hh(leaf_bytes + leaf_bytes)
    assert merkle_root_hex([leaf]) == expected


def test_two_leaves():
    l0, l1 = "00" * 32, "ff" * 32
    expected = _hh(bytes.fromhex(l0) + bytes.fromhex(l1))
    assert merkle_root_hex([l0, l1]) == expected


def test_three_leaves_duplicates_right_on_odd_count():
    """Bitcoin-style odd-count duplication on the right side of each level."""
    l0 = "00" * 32
    l1 = "11" * 32
    l2 = "22" * 32
    # Level 1: pair (L0, L1) → H01; pair (L2, L2) → H22.
    h01 = hashlib.sha256(bytes.fromhex(l0) + bytes.fromhex(l1)).digest()
    h22 = hashlib.sha256(bytes.fromhex(l2) + bytes.fromhex(l2)).digest()
    # Level 0: pair (H01, H22) → root.
    expected = hashlib.sha256(h01 + h22).hexdigest()
    assert merkle_root_hex([l0, l1, l2]) == expected


def test_four_leaves_balanced_tree():
    leaves = [f"{i:064x}" for i in range(4)]
    h01 = hashlib.sha256(bytes.fromhex(leaves[0]) + bytes.fromhex(leaves[1])).digest()
    h23 = hashlib.sha256(bytes.fromhex(leaves[2]) + bytes.fromhex(leaves[3])).digest()
    expected = hashlib.sha256(h01 + h23).hexdigest()
    assert merkle_root_hex(leaves) == expected


def test_determinism():
    leaves = [f"{i:064x}" for i in range(7)]
    a = merkle_root_hex(leaves)
    b = merkle_root_hex(list(leaves))
    assert a == b


def test_order_sensitivity():
    leaves = [f"{i:064x}" for i in range(4)]
    a = merkle_root_hex(leaves)
    b = merkle_root_hex(list(reversed(leaves)))
    assert a != b


def test_output_format_is_lowercase_hex_64_chars():
    for n in (0, 1, 3, 5, 8, 13):
        leaves = [f"{i:064X}" for i in range(n)]  # uppercase input
        r = merkle_root_hex(leaves)
        assert len(r) == 64
        assert r == r.lower()
        # All chars are hex.
        int(r, 16)

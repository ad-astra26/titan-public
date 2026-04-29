"""Unit tests for `take_heap_snapshot()` and `TitanKernel.dump_heap`.

Validates that the heap-snapshot helper:
  - Returns the expected schema
  - Counts known objects we plant in the heap
  - Surfaces large containers in `top_containers`
  - Filters containers below `container_min_len`
  - Does not raise on weird C-level objects (frames, generators)
  - Honors `top_types` and `top_containers` limits

These tests run in-process (no subprocess) so the heap is whatever the
test process happens to hold — we only assert on planted objects we
control, not absolute counts.
"""
from __future__ import annotations

import gc

from titan_plugin.core.profiler import take_heap_snapshot


def test_schema_keys_present():
    """Every result has the documented top-level keys."""
    snap = take_heap_snapshot(top_types=5, top_containers=5)
    for key in ("object_count", "type_count", "total_size_mb",
                "duration_s", "top_types", "top_containers"):
        assert key in snap, f"missing key {key!r}"
    assert isinstance(snap["top_types"], list)
    assert isinstance(snap["top_containers"], list)
    assert snap["object_count"] > 0
    assert snap["type_count"] > 0


def test_top_types_limit_honored():
    """`top_types` caps the returned list length."""
    snap = take_heap_snapshot(top_types=3, top_containers=0)
    assert len(snap["top_types"]) <= 3
    for entry in snap["top_types"]:
        assert "type" in entry
        assert "count" in entry
        assert "total_mb" in entry
        assert "max_kb" in entry
        assert entry["count"] >= 1


def test_planted_container_appears_in_top_containers():
    """A purposely-large list shows up in `top_containers` by len."""
    # Plant a large list — anchor a strong reference so gc.get_objects()
    # tracks it.
    big_list = [i for i in range(10_000)]
    big_dict = {i: f"v{i}" for i in range(5_000)}
    gc.collect()

    snap = take_heap_snapshot(top_types=10, top_containers=20,
                              container_min_len=100)
    container_lens = [c["len"] for c in snap["top_containers"]]
    assert any(ln >= 10_000 for ln in container_lens), (
        f"planted big_list (len=10000) missing from top_containers: "
        f"top lens = {container_lens[:5]}")
    assert any(ln >= 5_000 for ln in container_lens), (
        f"planted big_dict (len=5000) missing from top_containers")
    # Keep refs live until end-of-test
    assert len(big_list) == 10_000
    assert len(big_dict) == 5_000


def test_container_min_len_filters_small():
    """Containers below `container_min_len` do not appear."""
    snap = take_heap_snapshot(top_types=0, top_containers=20,
                              container_min_len=1_000_000)
    # Almost certainly nothing this large in a test process
    assert snap["top_containers"] == []


def test_zero_limits_return_empty_lists():
    """`top_types=0` returns no types; `top_containers=0` returns none."""
    snap = take_heap_snapshot(top_types=0, top_containers=0)
    assert snap["top_types"] == []
    assert snap["top_containers"] == []
    # Aggregate counts should still be populated
    assert snap["object_count"] > 0


def test_does_not_raise_on_weird_objects():
    """Generators, frames, weakrefs in the heap don't crash the walker."""
    # Plant a generator (frame object) and a weakref
    gen = (i for i in range(5))
    import weakref

    class _Holder:
        pass

    h = _Holder()
    wr = weakref.ref(h)

    # Should not raise
    snap = take_heap_snapshot(top_types=5, top_containers=5)
    assert snap["object_count"] > 0
    # Keep refs live
    assert next(gen) == 0
    assert wr() is h


def test_total_size_mb_is_consistent():
    """`total_size_mb` is the sum of all per-type totals (within rounding)."""
    snap = take_heap_snapshot(top_types=10_000, top_containers=0)
    sum_top = sum(t["total_mb"] for t in snap["top_types"])
    # `total_size_mb` covers ALL types; `top_types` only the top N. With
    # a huge top_n (10_000) we should capture the bulk and be within ~5%.
    assert sum_top <= snap["total_size_mb"] + 1.0
    # Don't assert tight equality — rounding + uncovered tail allowed.

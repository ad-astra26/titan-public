"""Per-Titan user_id hashing — Phase 2 closure tests.

Covers: salt bootstrap (fresh Titan), salt persistence across restarts,
hash determinism within a Titan, privacy across Titans (different salts
→ different hashes for same user_id), anonymous short-circuit, tag
format conformance with arch §7.

PLAN_synthesis_engine_Phase2.md §B.5.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from titan_hcl.synthesis import user_id_hash as uih


def _isolated_secrets_path(tmpdir: str) -> Path:
    """Build an isolated secrets.toml path for a test."""
    return Path(tmpdir) / "secrets.toml"


class TestHashFormat(unittest.TestCase):
    """The hash output is `user:<16-hex>` per arch §7."""

    def setUp(self) -> None:
        # Use a fixed in-memory salt to make outputs deterministic for
        # format assertions (no real secrets.toml side effects).
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch.object(
            uih, "_salt_cache", b"\xab" * 32)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()
        uih.clear_cache()

    def test_format_is_user_prefix_plus_16_hex(self) -> None:
        tag = uih.hash_user_id("maker")
        assert tag.startswith("user:")
        hex_part = tag[len("user:"):]
        assert len(hex_part) == 16
        # All hex chars.
        int(hex_part, 16)

    def test_empty_user_id_returns_empty(self) -> None:
        assert uih.hash_user_id("") == ""
        assert uih.hash_user_id(None) == ""  # type: ignore[arg-type]

    def test_anonymous_short_circuits(self) -> None:
        """`"anonymous"` is the chat-pipeline sentinel for missing user_id.
        We MUST NOT mint a bundle key for it (would aggregate every
        anon visitor under one tag — misleading)."""
        assert uih.hash_user_id("anonymous") == ""

    def test_raw_form_strips_prefix(self) -> None:
        tag = uih.hash_user_id("maker")
        raw = uih.hash_user_id_raw("maker")
        assert raw == tag[len("user:"):]
        assert ":" not in raw

    def test_raw_form_empty_for_anonymous(self) -> None:
        assert uih.hash_user_id_raw("anonymous") == ""
        assert uih.hash_user_id_raw("") == ""


class TestHashDeterminism(unittest.TestCase):
    """Same (user_id, salt) → same hash, always."""

    def setUp(self) -> None:
        self._patcher = patch.object(uih, "_salt_cache", b"\xcd" * 32)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        uih.clear_cache()

    def test_same_user_id_same_hash(self) -> None:
        h1 = uih.hash_user_id("alice")
        h2 = uih.hash_user_id("alice")
        assert h1 == h2

    def test_different_user_id_different_hash(self) -> None:
        h_a = uih.hash_user_id("alice")
        h_b = uih.hash_user_id("bob")
        assert h_a != h_b

    def test_user_id_is_utf8_safe(self) -> None:
        """Non-ASCII user_ids hash deterministically (errors='replace')."""
        h1 = uih.hash_user_id("užívatel")
        h2 = uih.hash_user_id("užívatel")
        assert h1 == h2
        assert h1 != uih.hash_user_id("uzivatel")


class TestCrossTitanPrivacy(unittest.TestCase):
    """Different per-Titan salts → different hashes for same user_id.
    Sovereignty axiom (rFP §20 Q6)."""

    def test_different_salts_yield_different_hashes(self) -> None:
        salt_t1 = b"\x11" * 32
        salt_t3 = b"\x33" * 32

        with patch.object(uih, "_salt_cache", salt_t1):
            h_t1 = uih.hash_user_id("maker")
        uih.clear_cache()
        with patch.object(uih, "_salt_cache", salt_t3):
            h_t3 = uih.hash_user_id("maker")
        uih.clear_cache()

        assert h_t1 != h_t3
        # Both are well-formed.
        assert h_t1.startswith("user:") and h_t3.startswith("user:")


class TestSaltBootstrapAndPersistence(unittest.TestCase):
    """First-call bootstrap: generate salt → persist via update_secret →
    cache. Second call: read from cache."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.secrets_path = _isolated_secrets_path(self._tmp.name)
        uih.clear_cache()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        uih.clear_cache()

    def test_bootstrap_generates_and_persists_when_missing(self) -> None:
        """No salt → generates fresh + writes to secrets.toml via
        params.update_secret (RFP_config_as_shm_state §7.C/C.5)."""
        from titan_hcl import params

        # _SECRETS_PATH is the single read (_bootstrap_merge) + write (update_secret)
        # secrets path — patching it redirects both the persist and the read-back.
        with patch.object(params, "_SECRETS_PATH", str(self.secrets_path)):
            assert not self.secrets_path.exists()
            salt_a = uih.get_user_id_hash_salt()
            assert isinstance(salt_a, bytes)
            assert len(salt_a) == 32   # 32-byte = 64 hex
            assert self.secrets_path.exists()

            # Subsequent call returns the same salt (cache hit).
            salt_b = uih.get_user_id_hash_salt()
            assert salt_a == salt_b

    def test_persisted_salt_survives_cache_clear(self) -> None:
        """Simulate restart: clear cache, re-resolve → identical salt
        from secrets.toml."""
        from titan_hcl import params

        with patch.object(params, "_SECRETS_PATH", str(self.secrets_path)):
            salt_first_boot = uih.get_user_id_hash_salt()
            # Simulate process restart.
            uih.clear_cache()
            salt_second_boot = uih.get_user_id_hash_salt()
            assert salt_first_boot == salt_second_boot

    def test_persistence_failure_falls_back_to_process_local(self) -> None:
        """If update_secret raises (e.g. tomli_w missing, file perms, disk),
        get_user_id_hash_salt MUST NOT propagate the exception. Returns
        an in-process salt so OVG's build_timechain_payload never breaks
        because of a persistence problem. This is the defensive path
        that surfaced during T3 live test 2026-05-25."""
        from titan_hcl import params
        # Force params.update_secret to raise ImportError (mimics
        # tomli_w not installed on the target Titan).
        with patch.object(params, "_SECRETS_PATH", str(self.secrets_path)), \
                patch.object(
                    params, "update_secret",
                    side_effect=ImportError("No module named 'tomli_w'")):
            uih.clear_cache()
            # Must NOT raise — must return a valid 32-byte salt.
            salt = uih.get_user_id_hash_salt()
            assert isinstance(salt, bytes)
            assert len(salt) == 32
            # And hash_user_id MUST work end-to-end without raising.
            tag = uih.hash_user_id("maker")
            assert tag.startswith("user:")
            assert len(tag) == len("user:") + 16

    def test_invalid_hex_is_regenerated(self) -> None:
        """Garbage in secrets.toml → regenerate (don't crash)."""
        from titan_hcl import params

        # Seed secrets with malformed salt.
        self.secrets_path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets_path.write_text(
            '[synthesis]\nuser_id_hash_salt = "not-valid-hex-XYZ-XYZ"\n')
        with patch.object(params, "_SECRETS_PATH", str(self.secrets_path)):
            salt = uih.get_user_id_hash_salt()
            assert len(salt) == 32  # regenerated
            # Verify the file was rewritten with a valid hex salt.
            import tomllib
            with open(self.secrets_path, "rb") as f:
                d = tomllib.load(f)
            new_hex = d["synthesis"]["user_id_hash_salt"]
            assert len(new_hex) == 64  # 32 bytes hex
            bytes.fromhex(new_hex)   # must be valid hex


class TestHashStabilityAcrossSession(unittest.TestCase):
    """The chat pipeline calls hash_user_id many times for the same user
    over a session — must be byte-stable + fast."""

    def setUp(self) -> None:
        self._patcher = patch.object(uih, "_salt_cache", b"\x99" * 32)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        uih.clear_cache()

    def test_thousand_calls_same_result(self) -> None:
        first = uih.hash_user_id("maker")
        for _ in range(1000):
            assert uih.hash_user_id("maker") == first

    def test_call_latency_under_50us(self) -> None:
        """Hot path: hash on every chat-TX construction. Must be sub-50µs
        so OVG's <1ms post-sign overhead budget stays intact. Real cost
        is ~1-3µs (single sha256) — 50µs is the loose ceiling."""
        import time
        # Warm cache.
        uih.hash_user_id("maker")
        # Measure 10k calls; per-call must be well under 50µs.
        N = 10_000
        t0 = time.perf_counter()
        for _ in range(N):
            uih.hash_user_id("maker")
        elapsed = time.perf_counter() - t0
        per_call_us = (elapsed / N) * 1_000_000
        assert per_call_us < 50.0, f"per-call {per_call_us:.2f}µs > 50µs"


if __name__ == "__main__":
    unittest.main()

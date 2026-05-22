"""Tests for BackupCascade.auto_fund_irys_if_needed.

Closes the rFP_backup_worker §5.5 silent-depletion gap exposed 2026-04-30:
Irys deposit drained → cascade silently fell to local_only mode for 147+ hours
before Maker noticed. Auto-fund tops up Irys within strict guardrails:

  - meditation_reserve floor (never drain wallet below this)
  - daily cap (limits damage from runaway loop)
  - explicit enable flag (default false, wire-now-gate-later)
  - Telegram alert on every fund event (transparency)
  - persistent audit trail at data/backups/auto_fund_audit.jsonl

See backup_cascade.py:auto_fund_irys_if_needed for full sequencing.
"""
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest

from titan_hcl.logic.backup_cascade import BackupCascade


@pytest.fixture
def tmp_backup_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def base_config(tmp_path):
    keypair = tmp_path / "kp.json"
    keypair.write_text("[1,2,3]")  # placeholder; subprocess is mocked
    return {
        "network": {"wallet_keypair_path": str(keypair)},
        "backup": {
            "auto_fund_enabled": True,
            "auto_fund_min_runway_days": 3.0,
            "auto_fund_target_runway_days": 14.0,
            "auto_fund_meditation_reserve_sol": 0.01,
            "auto_fund_daily_cap_sol": 0.05,
            "auto_fund_avg_uploads_per_day": 3.0,
        },
    }


# ── Disabled-flag short circuit ────────────────────────────────────────────

def test_disabled_returns_skipped(base_config, tmp_backup_dir):
    base_config["backup"]["auto_fund_enabled"] = False
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)
    result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert result["reason"] == "disabled"


def test_default_flag_value_is_false(tmp_backup_dir):
    """When [backup] section is absent, flag defaults to false."""
    cascade = BackupCascade(full_config={"network": {"wallet_keypair_path": "/nonexistent"}},
                             local_dir=tmp_backup_dir)
    result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert result["reason"] == "disabled"


# ── Keypair guards ─────────────────────────────────────────────────────────

def test_no_keypair_path_skips(base_config, tmp_backup_dir):
    base_config["network"] = {}
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)
    result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert result["reason"] == "no_keypair"


def test_missing_keypair_file_skips(base_config, tmp_backup_dir):
    base_config["network"]["wallet_keypair_path"] = "/nonexistent/path.json"
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)
    result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert result["reason"] == "no_keypair"


# ── Runway sufficient ─────────────────────────────────────────────────────

def test_runway_sufficient_no_action(base_config, tmp_backup_dir):
    """5 days runway > 3 days threshold → no action."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def fake_check_output(args, **kwargs):
        if "balance" in args:
            return b'{"status":"ok","balance_atomic":"100000000","balance_readable":"0.1"}'
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    # daily_burn = 26.6 * 0.0002 * 2.0 * 3 = 0.031920 SOL
    # runway = 0.1 / 0.031920 ≈ 3.13 days — barely above min_runway_days=3.0
    assert result["action"] == "no_action"
    assert result["reason"] == "runway_sufficient"
    assert result["runway_days"] >= 3.0


# ── Wallet reserve floor ──────────────────────────────────────────────────

def test_wallet_below_reserve_skips(base_config, tmp_backup_dir):
    """Wallet at 0.005 SOL < meditation_reserve 0.01 → can't spend safely."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    call_count = {"n": 0}

    def fake_check_output(args, **kwargs):
        call_count["n"] += 1
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"5000000","balance_readable":"0.005"}'
        if args[0] == "solana":
            return b'0.005000000 SOL\n'
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert result["reason"] == "wallet_below_reserve"
    assert result["wallet_sol"] == 0.005
    assert result["meditation_reserve"] == 0.01


# ── Daily cap enforcement ─────────────────────────────────────────────────

def test_daily_cap_reached_skips(base_config, tmp_backup_dir):
    """Yesterday's accumulator carries 0.05 SOL (today's cap) → skip."""
    state_path = os.path.join(tmp_backup_dir, ".auto_fund_daily.json")
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with open(state_path, "w") as f:
        json.dump({"date": today, "total_sol": 0.05, "tx_count": 1}, f)

    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def fake_check_output(args, **kwargs):
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"1000000","balance_readable":"0.001"}'
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert result["reason"] == "daily_cap_reached"
    assert result["today_total_sol"] == 0.05


def test_old_day_resets_counter(base_config, tmp_backup_dir):
    """Yesterday's accumulator should not count against today."""
    state_path = os.path.join(tmp_backup_dir, ".auto_fund_daily.json")
    with open(state_path, "w") as f:
        json.dump({"date": "1999-01-01", "total_sol": 999.0, "tx_count": 99}, f)
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)
    today, total, count = cascade._get_auto_fund_today_total()
    assert today == time.strftime("%Y-%m-%d", time.gmtime())
    assert total == 0.0
    assert count == 0


# ── Successful fund flow ──────────────────────────────────────────────────

def test_successful_fund(base_config, tmp_backup_dir):
    """Low runway + healthy wallet → auto-fund executes."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)
    notifications = []

    def notifier(msg):
        notifications.append(msg)

    def fake_check_output(args, **kwargs):
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"5000000","balance_readable":"0.005"}'
        if args[0] == "solana":
            return b'0.200000000 SOL\n'
        if "fund" in args and "irys_upload" in str(args):
            return (b'{"status":"ok","funded":"40000000","target":"5z2w...",'
                    b'"tx_id":"4xg8KWi6GeMjmvCXLDTC"}')
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6, notifier=notifier)

    assert result["action"] == "funded"
    assert result["tx_id"] == "4xg8KWi6GeMjmvCXLDTC"
    assert result["amount_sol"] > 0
    assert result["amount_sol"] <= 0.05  # daily cap respected
    assert len(notifications) == 1
    assert "Auto-Fund Irys" in notifications[0]
    # Audit log written
    audit_path = os.path.join(tmp_backup_dir, "auto_fund_audit.jsonl")
    assert os.path.exists(audit_path)
    with open(audit_path) as f:
        line = f.readline()
        entry = json.loads(line)
    assert entry["tx_id"] == "4xg8KWi6GeMjmvCXLDTC"
    assert entry["amount_sol"] > 0
    # Daily counter persisted
    state_path = os.path.join(tmp_backup_dir, ".auto_fund_daily.json")
    with open(state_path) as f:
        state = json.load(f)
    assert state["tx_count"] == 1
    assert state["total_sol"] == result["amount_sol"]


def test_fund_caps_at_remaining_daily_budget(base_config, tmp_backup_dir):
    """Today already spent 0.04 SOL → fund capped at 0.01 (remaining budget)."""
    state_path = os.path.join(tmp_backup_dir, ".auto_fund_daily.json")
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with open(state_path, "w") as f:
        json.dump({"date": today, "total_sol": 0.04, "tx_count": 1}, f)

    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def fake_check_output(args, **kwargs):
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"1000000","balance_readable":"0.001"}'
        if args[0] == "solana":
            return b'0.500000000 SOL\n'
        if "fund" in args and "irys_upload" in str(args):
            # Verify lamports is at most 0.01 SOL = 10_000_000 lamports
            lamports = int(args[3])
            assert lamports <= 10_000_000, f"daily cap not enforced: {lamports} lamports"
            return (b'{"status":"ok","funded":"' + str(lamports).encode() +
                    b'","tx_id":"capped_tx_id"}')
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "funded"
    assert result["amount_sol"] <= 0.01


def test_fund_caps_at_wallet_minus_reserve(base_config, tmp_backup_dir):
    """Wallet at 0.013 SOL, reserve 0.01 → max spend 0.003 SOL."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def fake_check_output(args, **kwargs):
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"1000000","balance_readable":"0.001"}'
        if args[0] == "solana":
            return b'0.013000000 SOL\n'
        if "fund" in args and "irys_upload" in str(args):
            lamports = int(args[3])
            # 0.013 - 0.01 = 0.003 SOL = 3_000_000 lamports
            assert lamports <= 3_000_000, f"reserve floor not enforced: {lamports}"
            return (b'{"status":"ok","funded":"' + str(lamports).encode() +
                    b'","tx_id":"reserve_capped"}')
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "funded"
    assert result["amount_sol"] <= 0.003 + 1e-9  # float tolerance


# ── Failure handling ──────────────────────────────────────────────────────

def test_balance_query_failure_skips(base_config, tmp_backup_dir):
    """Irys balance query throws → skipped (don't crash cascade)."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def fake_check_output(args, **kwargs):
        raise OSError("network down")

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "skipped"
    assert "balance_query_error" in result["reason"]


def test_fund_call_failure_returns_failed(base_config, tmp_backup_dir):
    """Fund TX call fails → action='failed' (not 'skipped' — distinguishes)."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def fake_check_output(args, **kwargs):
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"1000000","balance_readable":"0.001"}'
        if args[0] == "solana":
            return b'0.500000000 SOL\n'
        if "fund" in args and "irys_upload" in str(args):
            return b'{"status":"error","message":"insufficient funds in wallet"}'
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6)
    assert result["action"] == "failed"
    assert "insufficient funds" in result["reason"]


def test_notifier_exception_does_not_crash_fund(base_config, tmp_backup_dir):
    """Telegram failure must not corrupt the fund result."""
    cascade = BackupCascade(full_config=base_config, local_dir=tmp_backup_dir)

    def broken_notifier(msg):
        raise RuntimeError("Telegram API down")

    def fake_check_output(args, **kwargs):
        if "balance" in args and "irys_upload" in str(args):
            return b'{"status":"ok","balance_atomic":"5000000","balance_readable":"0.005"}'
        if args[0] == "solana":
            return b'0.200000000 SOL\n'
        if "fund" in args and "irys_upload" in str(args):
            return b'{"status":"ok","funded":"10000000","tx_id":"tg_broken_tx"}'
        return b''

    with patch("subprocess.check_output", side_effect=fake_check_output):
        result = cascade.auto_fund_irys_if_needed(size_mb=26.6, notifier=broken_notifier)
    assert result["action"] == "funded"
    assert result["tx_id"] == "tg_broken_tx"

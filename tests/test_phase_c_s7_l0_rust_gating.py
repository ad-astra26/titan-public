"""Phase C C-S7 — l0_rust_enabled gating parity tests.

Covers PLAN_microkernel_phase_c_s7_activation_prep.md §4 tests 3 + 4:
  3. SHM writer gating — kernel.boot() does NOT start Trinity / Topology
     SHM writers when microkernel.l0_rust_enabled=true (Rust daemons own).
  4. Outer worker registration gating — Plugin must NOT register
     outer_body / outer_mind / outer_spirit when l0_rust_enabled=true
     (Rust outer-{body,mind,spirit}-rs daemons own those slots).

Source-inspection style: each gate's invariant is a textual property of
the source. If a future refactor removes the gate, these tests fail,
preventing a SeqLock double-writer race or a duplicate Guardian
registration from sneaking through silently.

Lightweight by design: no full kernel boot, no event loop, no
TitanHCL construction — gating is a structural property, so we verify
it structurally. Behavioral end-to-end coverage is added in commit 6
(activation soak under l0_rust=true).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


# ─── Helpers ───────────────────────────────────────────────────────────


def _read_source(rel_path: str) -> str:
    """Read a project-relative file as text."""
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / rel_path).read_text()


# ─── Test 3 — SHM writer gating in kernel.boot ─────────────────────────


class TestShmWriterGating:
    """kernel.boot() — Trinity + Topology SHM writers gated on l0_rust_enabled.

    Under l0_rust=true:
      - titan-unified-spirit-rs owns trinity_state.bin
      - titan-trinity-rs (or outer-spirit-rs) owns topology_30d.bin
    Python writers MUST NOT run in that mode (SeqLock double-writer race).
    """

    def test_trinity_shm_writer_call_is_gated(self):
        """_start_trinity_shm_writer call is wrapped in l0_rust check."""
        src = _read_source("titan_hcl/core/kernel.py")
        # Find the boot() method's section near _start_trinity_shm_writer().
        # Pattern: an l0_rust_enabled guard immediately preceding the call.
        m = re.search(
            r'l0_rust_enabled["\'],?\s*False\s*\)[\s\S]{0,400}?'
            r'self\._start_trinity_shm_writer\(\)',
            src,
        )
        assert m is not None, (
            "kernel.boot() must gate self._start_trinity_shm_writer() "
            "behind a microkernel.l0_rust_enabled check (Phase C C-S7 Gap 4)"
        )

    def test_topology_shm_writer_call_is_gated(self):
        """_start_topology_shm_writer call is wrapped in l0_rust check."""
        src = _read_source("titan_hcl/core/kernel.py")
        m = re.search(
            r'l0_rust_enabled["\'],?\s*False\s*\)[\s\S]{0,400}?'
            r'self\._start_topology_shm_writer\(\)',
            src,
        )
        assert m is not None, (
            "kernel.boot() must gate self._start_topology_shm_writer() "
            "behind a microkernel.l0_rust_enabled check (Phase C C-S7 Gap 5)"
        )

    def test_spirit_shm_writer_logs_l0_rust_state(self):
        """_start_spirit_shm_writer (no-op breadcrumb) surfaces l0_rust state.

        The hook is called either way, but its boot-log message must
        announce SHIM-mode under l0_rust=true so operators can confirm
        the Rust path is active.
        """
        src = _read_source("titan_hcl/core/kernel.py")
        # Find the _start_spirit_shm_writer method body and verify it
        # branches on l0_rust_enabled.
        method_match = re.search(
            r'def _start_spirit_shm_writer\(self\)[\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert method_match is not None, "_start_spirit_shm_writer not found"
        body = method_match.group(0)
        assert "l0_rust_enabled" in body, (
            "_start_spirit_shm_writer must surface l0_rust_enabled state "
            "in its boot log (Phase C C-S7 Gap 6 — operator visibility)"
        )
        assert "SHIM" in body, (
            "_start_spirit_shm_writer must log SHIM-mode message when "
            "l0_rust_enabled=true"
        )


# ─── Test 4 — Outer worker registration gating ─────────────────────────


class TestOuterWorkerRegistrationGating:
    """Plugin.__init__ — outer_{body,mind,spirit} registration gated.

    Under l0_rust=true:
      - titan-outer-body-rs / outer-mind-rs / outer-spirit-rs own those slots
    Python A.S8 workers MUST NOT register with Guardian (Rust supervisor
    owns them per arch §4.5).
    """

    def test_outer_workers_fully_retired(self):
        """The legacy outer_*_worker modules are RETIRED (Phase C dissolution
        C.8, 2026-05-22, no-shim). Under l0_rust=true (production fleet) they
        were never spawned (C.0); the Rust outer daemons own the tensor slots
        and the source data plane is now SHM-direct. plugin.py must neither
        import nor register them. Supersedes 'imports are gated'."""
        src = _read_source("titan_hcl/core/plugin.py")
        for worker_name in ("outer_body_worker", "outer_mind_worker",
                            "outer_spirit_worker"):
            assert re.search(
                rf'from titan_hcl\.modules\.{worker_name} import', src
            ) is None, f"{worker_name} must NOT be imported (retired C.8)"
            assert f"{worker_name}_main" not in src, (
                f"{worker_name}_main must NOT be registered (retired C.8)")


# ─── Test 1+2 — BusSocketClient attach + outbound + dispatcher ─────────


class TestBusClientAttach:
    """DivineBus — attach_client routes outbound via client (Phase C C-S7
    Gap 1+2+3) and is mutually exclusive with attach_broker."""

    def _make_bus(self):
        from titan_hcl.bus import DivineBus
        return DivineBus()

    def test_attach_client_routes_publish(self):
        """bus.publish(msg) routes via client.publish() when only client attached."""
        from unittest.mock import MagicMock
        bus = self._make_bus()
        fake_client = MagicMock()
        fake_client.name = "titan_HCL"
        fake_client.sock_path = "/tmp/test.sock"
        bus.attach_client(fake_client)
        assert bus.has_socket_client is True
        msg = {"type": "TEST", "src": "kernel", "dst": "memory", "payload": {}}
        bus.publish(msg)
        fake_client.publish.assert_called_once()
        # Verify the same dict was forwarded
        forwarded = fake_client.publish.call_args[0][0]
        assert forwarded["type"] == "TEST"
        assert forwarded["dst"] == "memory"

    def test_broker_takes_priority_over_client(self):
        """When BOTH _broker and _client attached, _broker wins (mutually
        exclusive in normal config but defensive guard)."""
        from unittest.mock import MagicMock
        bus = self._make_bus()
        fake_broker = MagicMock()
        fake_broker.sock_path = "/tmp/broker.sock"
        fake_client = MagicMock()
        fake_client.name = "titan_HCL"
        fake_client.sock_path = "/tmp/client.sock"
        bus.attach_broker(fake_broker)
        bus.attach_client(fake_client)
        msg = {"type": "TEST", "src": "kernel", "dst": "all", "payload": {}}
        bus.publish(msg)
        # Broker takes the path, client must NOT be called when broker is set.
        fake_broker.publish.assert_called_once()
        fake_client.publish.assert_not_called()

    def test_detach_client_idempotent(self):
        from unittest.mock import MagicMock
        bus = self._make_bus()
        fake_client = MagicMock()
        fake_client.name = "titan_HCL"
        fake_client.sock_path = "/tmp/test.sock"
        bus.attach_client(fake_client)
        bus.detach_client()
        bus.detach_client()  # idempotent
        assert bus.has_socket_client is False
        # Subsequent publish does not raise even with no broker/client.
        bus.publish({"type": "TEST", "src": "kernel", "dst": "memory", "payload": {}})

    def test_publish_no_broker_no_client_inprocess_only(self):
        """Without broker AND without client, publish returns gracefully."""
        bus = self._make_bus()
        msg = {"type": "TEST", "src": "kernel", "dst": "memory", "payload": {}}
        # Just verify it doesn't raise (no subscribers, no broker, no client)
        delivered = bus.publish(msg)
        assert delivered == 0


class TestInProcessSubscriberNames:
    """Phase C C-S7 — the constant must include all names workers target."""

    def test_includes_guardian_for_module_heartbeat(self):
        from titan_hcl.core.kernel import IN_PROCESS_SUBSCRIBER_NAMES
        # MODULE_HEARTBEAT and MODULE_READY are dst="guardian" — the constant
        # MUST register a client under this name or worker liveness breaks.
        assert "guardian" in IN_PROCESS_SUBSCRIBER_NAMES

    def test_includes_titan_HCL_as_canonical_outbound(self):
        from titan_hcl.core.kernel import IN_PROCESS_SUBSCRIBER_NAMES
        # The plugin's canonical outbound publisher identity. Per PLAN §2 Gap 1.
        assert "titan_HCL" in IN_PROCESS_SUBSCRIBER_NAMES

    def test_no_duplicates(self):
        from titan_hcl.core.kernel import IN_PROCESS_SUBSCRIBER_NAMES
        assert len(IN_PROCESS_SUBSCRIBER_NAMES) == len(set(IN_PROCESS_SUBSCRIBER_NAMES))


class TestInboundDispatcherEchoAndDedup:
    """Phase C C-S7 — dispatcher echo prevention + broadcast dedup."""

    def test_echo_prevention_drops_self_published_messages(self):
        """Messages whose src is in IN_PROCESS_SUBSCRIBER_NAMES are dropped
        (they originated in this process; broker echoed them back)."""
        src = _read_source("titan_hcl/core/kernel.py")
        m = re.search(
            r'def _bus_client_inbound_dispatcher\([\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert m is not None, "_bus_client_inbound_dispatcher not found"
        body = m.group(0)
        # Strip docstring so position checks operate on actual code only.
        # Match: triple-quoted block right after def(...) signature.
        no_doc = re.sub(r'""".*?"""', '', body, count=1, flags=re.DOTALL)
        # Echo check pattern: `if src in plugin_names: continue` (or any
        # equivalent that drops based on plugin_names membership).
        echo_match = re.search(r'if\s+src\s+in\s+plugin_names', no_doc)
        assert echo_match is not None, (
            "Dispatcher must check `if src in plugin_names` to prevent "
            "self-publish echo loops"
        )
        publish_match = re.search(r'\.publish_in_process\(', no_doc)
        assert publish_match is not None, (
            "Dispatcher must call publish_in_process to relay messages"
        )
        assert echo_match.start() < publish_match.start(), (
            "Echo check (if src in plugin_names) must precede "
            "publish_in_process call"
        )

    def test_broadcast_dedup_only_titan_HCL_relays(self):
        """SPEC §9.B-aligned design: SINGLE BusSocketClient connection +
        SINGLE inbound dispatcher = broadcasts arrive EXACTLY ONCE per
        dispatcher → no relay-gate, no dedup needed.

        Pre-2026-05-12: 6 dispatcher threads (one per per-name connection),
        broadcasts arrived 6× and we gated relay on `client_name == "titan_HCL"`
        via an `is_broadcast_relay` flag.

        Post-2026-05-12 (D-SPEC-42 v1.4.0 multi-name BUS_SUBSCRIBE): 1
        connection → 1 dispatcher → 1× delivery. The `is_broadcast_relay`
        variable was deleted with intent. This test asserts the dispatcher
        body documents that decision (so a future refactor doesn't
        accidentally re-introduce the per-name multi-connection pattern
        without re-introducing the dedup gate).
        """
        src = _read_source("titan_hcl/core/kernel.py")
        m = re.search(
            r'def _bus_client_inbound_dispatcher\([\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert m is not None
        body = m.group(0)
        # The dispatcher MUST document the dedup-not-needed rationale so a
        # future refactor that re-introduces multi-connection topology
        # also re-introduces the dedup gate.
        assert "Broadcast deduplication: NO LONGER NEEDED" in body or "no dedup needed" in body, (
            "Dispatcher must document why broadcast dedup is no longer "
            "needed (single connection post-D-SPEC-42); without this docstring "
            "comment, a future refactor that adds per-name connections back "
            "would silently regress to N-fold broadcast duplication."
        )
        assert '"titan_HCL"' in body or 'titan_HCL' in body, (
            "Dispatcher must reference titan_HCL as the canonical "
            "connection name (per SPEC §9.B titan_HCL block)."
        )


# ─── Test 5 — Shadow swap blocked under l0_rust (Gap 8) ────────────────


class TestShadowSwapBlockedUnderL0Rust:
    """Phase C C-S7 Gap 8 — shadow_swap_orchestrate must refuse under
    l0_rust=true; Rust kernel has no BUS_HANDOFF protocol yet."""

    def test_shadow_swap_refuses_under_l0_rust(self):
        src = _read_source("titan_hcl/core/kernel.py")
        # Locate shadow_swap_orchestrate body
        m = re.search(
            r'def shadow_swap_orchestrate\([\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert m is not None, "shadow_swap_orchestrate not found"
        body = m.group(0)
        # The l0_rust check must be present and produce an error outcome.
        l0_check = re.search(
            r'l0_rust_enabled["\'],?\s*False\s*\)',
            body,
        )
        assert l0_check is not None, (
            "shadow_swap_orchestrate must check microkernel.l0_rust_enabled "
            "(Phase C C-S7 Gap 8)"
        )
        # And must return failure_reason = l0_rust_enabled_shadow_swap_unsupported
        assert "l0_rust_enabled_shadow_swap_unsupported" in body, (
            "shadow_swap_orchestrate must return a clear failure_reason "
            "when l0_rust=true (operator must understand why blocked)"
        )

    def test_l0_rust_check_precedes_shadow_swap_flag_check(self):
        """The l0_rust check must come BEFORE the shadow_swap_enabled flag
        check — otherwise an operator with l0_rust=true who flips
        shadow_swap_enabled=true would get the wrong refusal reason."""
        src = _read_source("titan_hcl/core/kernel.py")
        m = re.search(
            r'def shadow_swap_orchestrate\([\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert m is not None
        body = m.group(0)
        # Strip docstring
        no_doc = re.sub(r'""".*?"""', '', body, count=1, flags=re.DOTALL)
        l0_idx = no_doc.find("l0_rust_enabled")
        flag_idx = no_doc.find("shadow_swap_enabled")
        assert 0 < l0_idx < flag_idx, (
            "l0_rust_enabled refusal must come BEFORE the "
            "shadow_swap_enabled flag check (clearer operator UX)"
        )


# ─── Test 6 — bus_specs Python ↔ Rust 1:1 parity (Gap 10) ──────────────


class TestBusSpecsParity:
    """Phase C C-S7 Gap 10 — every Python MSG_SPECS key has a Rust SPECS
    counterpart in titan-rust/crates/titan-core/src/bus_specs.rs.

    Under l0_rust=true the broker is Rust; if a Python publisher emits a
    type the Rust broker doesn't know, the message lands in the
    DEFAULT_SPEC bucket (P2, no coalesce, no TTL) silently. That's a
    semantic divergence — Python intends P0 for KERNEL_EPOCH_TICK, Rust
    might fall back to P2 if the entry is missing. Test asserts no such
    drift.
    """

    @staticmethod
    def _python_keys() -> set[str]:
        from titan_hcl.bus_specs import MSG_SPECS
        return set(MSG_SPECS.keys())

    @staticmethod
    def _rust_keys() -> set[str]:
        # Parse titan-rust/crates/titan-core/src/bus_specs.rs for all
        # `"<NAME>" => BusMsgSpec { ... }` entries inside the SPECS phf_map.
        src = _read_source("titan-rust/crates/titan-core/src/bus_specs.rs")
        # Locate the SPECS map block.
        specs_start = src.find("pub static SPECS")
        assert specs_start > 0, "SPECS phf_map not found in Rust bus_specs"
        # Match keys: lines like `    "NAME" => BusMsgSpec {` (2-space or
        # 4-space indent, possibly preceded by inline comments). Be liberal:
        # match `"<UPPER>" => BusMsgSpec` anywhere in the file.
        keys = set(re.findall(
            r'"([A-Z][A-Z0-9_]+)"\s*=>\s*BusMsgSpec',
            src[specs_start:],
        ))
        return keys

    def test_every_python_msg_spec_has_rust_counterpart(self):
        """Critical parity invariant — Python MSG_SPECS ⊆ Rust SPECS."""
        py_keys = self._python_keys()
        rust_keys = self._rust_keys()
        missing = py_keys - rust_keys
        assert not missing, (
            f"{len(missing)} Python MSG_SPECS key(s) absent from Rust "
            f"SPECS phf_map: {sorted(missing)[:10]}"
            f"{' (and more)' if len(missing) > 10 else ''}. "
            f"Rust broker would fall back to DEFAULT_SPEC (P2, no "
            f"coalesce) — silent drift. Add the entries to "
            f"titan-rust/crates/titan-core/src/bus_specs.rs."
        )

    def test_rust_specs_count_matches_constant(self):
        """SPECS_COUNT_V0_1_0 sanity check — Rust constant tracks the
        actual phf_map size. Catches drift from manual edits that bump
        one but not the other."""
        src = _read_source("titan-rust/crates/titan-core/src/bus_specs.rs")
        m = re.search(r'SPECS_COUNT_V0_1_0:\s*usize\s*=\s*([\d\s+]+);', src)
        assert m is not None, "SPECS_COUNT_V0_1_0 constant not found"
        # Eval the simple "44 + 5" arithmetic safely
        expr = m.group(1).strip().replace(" ", "")
        try:
            constant_val = sum(int(p) for p in expr.split("+"))
        except ValueError:
            pytest.fail(f"SPECS_COUNT_V0_1_0 expression unparseable: {expr!r}")
        actual = len(self._rust_keys())
        assert actual == constant_val, (
            f"SPECS_COUNT_V0_1_0={constant_val} but Rust SPECS has {actual} "
            f"entries. Update the constant in bus_specs.rs to match."
        )


# ─── Test 7 — Kernel respawn cascade (Gap A + Gap B) ───────────────────


class TestKernelSupervisorWiring:
    """Phase C C-S7 Gap A + Gap B — kernel must spawn Python plugin in
    production AND wire kernel→substrate + kernel→python_main +
    substrate→unified-spirit through the titan-core Supervisor framework."""

    def test_main_rs_flips_spawn_python_to_true(self):
        """main.rs constructs KernelRunOptions so production boot spawns
        titan_HCL (Gap A).

        Production must spawn Python by default. The implementation derives
        the flag from `!std::env::var("TITAN_KERNEL_SKIP_PYTHON")` so
        cross-language integration tests can opt out by setting that env
        var — production never sets it, so production gets spawn_python=true.

        This test accepts either form:
          - Literal `spawn_python: true` (simple production-only main.rs)
          - `spawn_python: !skip_python` (current production main.rs with
            opt-out env var for integration tests)
        """
        src = _read_source("titan-rust/crates/titan-kernel-rs/src/main.rs")
        # Accept literal `true` OR the `!skip_python` opt-out pattern (which
        # defaults to true when TITAN_KERNEL_SKIP_PYTHON is unset).
        m_literal = re.search(
            r'KernelRunOptions\s*\{[^}]*spawn_python:\s*true',
            src,
            re.DOTALL,
        )
        m_optout = re.search(
            r'KernelRunOptions\s*\{[^}]*spawn_python:\s*!\s*skip_python',
            src,
            re.DOTALL,
        )
        assert m_literal is not None or m_optout is not None, (
            "main.rs must set spawn_python=true on the production "
            "KernelRunOptions (Phase C C-S7 Gap A — without this the "
            "kernel boots Rust tree but never spawns the Python plugin, "
            "leaving Titan brain-dead under l0_rust=true). Accepted "
            "patterns: `spawn_python: true` OR `spawn_python: !skip_python` "
            "with TITAN_KERNEL_SKIP_PYTHON env var opt-out."
        )

    def test_kernel_supervisor_module_exists(self):
        """titan-kernel-rs has a kernel_supervisor module wiring Supervisor
        to substrate + python child lifecycles."""
        src = _read_source("titan-rust/crates/titan-kernel-rs/src/kernel_supervisor.rs")
        # Must register both kernel children with canonical names.
        assert 'CHILD_NAME_SUBSTRATE: &str = "trinity-substrate"' in src, (
            "kernel_supervisor must declare CHILD_NAME_SUBSTRATE matching "
            "SPEC §9.A naming"
        )
        assert 'CHILD_NAME_PYTHON: &str = "titan_HCL"' in src, (
            "kernel_supervisor must declare CHILD_NAME_PYTHON matching "
            "SPEC §9.B titan_HCL row"
        )
        # Must wrap the framework Supervisor.
        assert "Arc<Mutex<Supervisor>>" in src, (
            "kernel_supervisor must wrap titan-core::supervisor::Supervisor "
            "for decision logic + escalation"
        )
        # Must use kernel_default_decision for in-process short-circuit
        # (per Maker decision 2026-05-05 + SPEC §11.B.1 step 4-5).
        assert "kernel_default_decision" in src, (
            "kernel_supervisor must short-circuit kernel-self escalation "
            "via kernel_default_decision (in-process; no bus round-trip)"
        )

    def test_kernel_main_loop_listens_for_supervisor_shutdown(self):
        """kernel.rs steady-state select! must listen on shutdown.notified()
        in addition to SIGTERM/SIGINT — the supervisor signals this when
        escalation resolves to Terminate (kernel must exit with code 64)."""
        src = _read_source("titan-rust/crates/titan-kernel-rs/src/kernel.rs")
        # Find the steady-state select! block
        m = re.search(
            r'tokio::select!\s*\{[^}]*shutdown\.notified\(\)[^}]*\}',
            src,
            re.DOTALL,
        )
        assert m is not None, (
            "kernel.rs steady-state must include `shutdown.notified()` "
            "branch in tokio::select! to wake on supervisor escalation "
            "(Phase C C-S7 Gap B)"
        )

    def test_kernel_exits_64_on_supervisor_terminate(self):
        """When supervisor terminate_requested(), kernel returns
        SupervisorSelfTerminate exit code (64 per SPEC §15)."""
        src = _read_source("titan-rust/crates/titan-kernel-rs/src/kernel.rs")
        assert "SupervisorSelfTerminate" in src, (
            "kernel must reference SupervisorSelfTerminate exit code so "
            "escalation→terminate cascades a fresh tree via systemd"
        )
        # The terminate_requested branch must precede the Clean fallback.
        m = re.search(
            r'if\s+supervisor_terminate\s*\{[\s\S]+?SupervisorSelfTerminate',
            src,
        )
        assert m is not None, (
            "kernel.rs must conditionally return SupervisorSelfTerminate "
            "when supervisor escalation resolved to Terminate"
        )


class TestSubstrateSupervisorWiring:
    """Phase C C-S7 Gap B — substrate must wire UnifiedSpiritSupervisor
    so unified-spirit crashes trigger §11.B respawn cascade per SPEC §11.0
    row 3."""

    def test_unified_spirit_supervisor_exists(self):
        """trinity-rs has a UnifiedSpiritSupervisor in supervise.rs."""
        src = _read_source("titan-rust/crates/titan-trinity-rs/src/supervise.rs")
        assert "UnifiedSpiritSupervisor" in src, (
            "trinity-rs must define UnifiedSpiritSupervisor"
        )
        assert 'CHILD_NAME_UNIFIED_SPIRIT: &str = "unified-spirit"' in src, (
            "UnifiedSpiritSupervisor must declare CHILD_NAME_UNIFIED_SPIRIT "
            "matching SPEC §9.A naming"
        )
        # Must use kernel_default_decision (substrate mirrors kernel's
        # escalation policy per Maker 2026-05-05).
        assert "kernel_default_decision" in src, (
            "UnifiedSpiritSupervisor must use kernel_default_decision for "
            "in-process escalation policy (substrate mirrors kernel)"
        )

    def test_substrate_main_uses_supervisor(self):
        """trinity-rs main.rs must spawn unified-spirit through the
        UnifiedSpiritSupervisor (not directly via spawn_unified_spirit)."""
        src = _read_source("titan-rust/crates/titan-trinity-rs/src/main.rs")
        assert "UnifiedSpiritSupervisor::new" in src, (
            "trinity-rs main.rs must construct UnifiedSpiritSupervisor "
            "instead of calling spawn_unified_spirit directly"
        )
        assert "spawn_and_watch" in src, (
            "trinity-rs main.rs must call sup.spawn_and_watch() so the "
            "supervisor takes over the watch lifecycle"
        )

    def test_substrate_exit_code_reserves_64_for_self_terminate(self):
        """SubstrateExitCode::SupervisorSelfTerminate = 64 per SPEC §15
        (was misused by BootFailure pre-C-S7; now corrected to BootFailure=65)."""
        src = _read_source("titan-rust/crates/titan-trinity-rs/src/exit.rs")
        m = re.search(r'SupervisorSelfTerminate\s*=\s*64', src)
        assert m is not None, (
            "SubstrateExitCode must define SupervisorSelfTerminate = 64 "
            "matching SPEC §15 canonical taxonomy"
        )
        m = re.search(r'BootFailure\s*=\s*65', src)
        assert m is not None, (
            "SubstrateExitCode::BootFailure must be 65 (was 64 pre-C-S7; "
            "moved to free 64 for the canonical SupervisorSelfTerminate)"
        )


# ─── Test 8 — Python supervision module + cross-language unification ──


class TestPythonSupervisionModule:
    """Phase C C-S7 Gap C — Python supervision data model mirrors Rust
    titan_core::supervisor exactly so cross-language SUPERVISION_*
    payloads are wire-compatible."""

    def test_supervision_reason_mirrors_rust_enum(self):
        from titan_hcl.supervision import SupervisionReason
        # All 11 Rust SupervisionReason variants must be present.
        for variant in (
            "OOM", "PANIC", "SEGV", "HANG", "EMPTY",
            "DEPENDENCY_BLOCKED", "CONFIG_ERROR", "BOOT_FAILURE",
            "CLEAN_EXIT", "KILLED", "OTHER",
        ):
            assert hasattr(SupervisionReason, variant), (
                f"SupervisionReason.{variant} missing — Rust↔Python parity broken"
            )
            # Wire format = canonical SCREAMING_SNAKE_CASE string.
            assert SupervisionReason[variant].value == variant

    def test_kernel_default_decision_mirrors_rust(self):
        from titan_hcl.supervision import (
            EscalationDecision, SupervisionReason, kernel_default_decision,
        )
        # Per SPEC §11.B.2 + Rust kernel_default_decision in escalation.rs:
        assert kernel_default_decision(SupervisionReason.OOM) == EscalationDecision.TERMINATE
        assert kernel_default_decision(SupervisionReason.PANIC) == EscalationDecision.TERMINATE
        assert kernel_default_decision(SupervisionReason.SEGV) == EscalationDecision.TERMINATE
        assert kernel_default_decision(SupervisionReason.HANG) == EscalationDecision.TERMINATE
        assert kernel_default_decision(SupervisionReason.EMPTY) == EscalationDecision.HALT
        assert kernel_default_decision(SupervisionReason.DEPENDENCY_BLOCKED) == EscalationDecision.CONTINUE
        assert kernel_default_decision(SupervisionReason.CONFIG_ERROR) == EscalationDecision.HALT
        assert kernel_default_decision(SupervisionReason.BOOT_FAILURE) == EscalationDecision.HALT
        assert kernel_default_decision(SupervisionReason.CLEAN_EXIT) == EscalationDecision.CONTINUE
        assert kernel_default_decision(SupervisionReason.KILLED) == EscalationDecision.TERMINATE

    def test_classify_exit_code_mirrors_rust(self):
        """Mirror of titan_core::supervisor::restart::classify_exit per SPEC §15."""
        from titan_hcl.supervision import SupervisionReason, classify_exit_code
        assert classify_exit_code(0) == SupervisionReason.CLEAN_EXIT
        assert classify_exit_code(1) == SupervisionReason.PANIC
        assert classify_exit_code(2) == SupervisionReason.CONFIG_ERROR
        assert classify_exit_code(6) == SupervisionReason.CONFIG_ERROR
        assert classify_exit_code(137) == SupervisionReason.KILLED  # SIGKILL
        assert classify_exit_code(139) == SupervisionReason.SEGV  # SIGSEGV
        assert classify_exit_code(143) == SupervisionReason.CLEAN_EXIT  # SIGTERM
        assert classify_exit_code(None) == SupervisionReason.KILLED  # signal-killed

    def test_dependency_dataclass_fields(self):
        from titan_hcl.supervision import (
            Dependency, DependencyKind, DependencySeverity,
        )
        dep = Dependency(
            name="x_api_reachable",
            kind=DependencyKind.EXTERNAL_SVC,
            severity=DependencySeverity.SOFT,
            check=lambda: True,
        )
        assert dep.name == "x_api_reachable"
        assert dep.kind == DependencyKind.EXTERNAL_SVC
        assert dep.severity == DependencySeverity.SOFT
        assert dep.check() is True


class TestPythonGuardianSupervisionEmit:
    """Phase C C-S7 Gap C — guardian.py emits SUPERVISION_* bus messages
    on restart events + runs escalation handshake on max_restarts."""

    def test_module_spec_has_dependencies_field(self):
        from titan_hcl.guardian_hcl import ModuleSpec
        from titan_hcl.supervision import Dependency
        # Default empty list (existing modules unaffected — byte-identical).
        spec = ModuleSpec(name="x", entry_fn=lambda *a, **k: None)
        assert spec.dependencies == []
        # Field accepts list[Dependency].
        spec_with_deps = ModuleSpec(
            name="y",
            entry_fn=lambda *a, **k: None,
            dependencies=[],
        )
        assert isinstance(spec_with_deps.dependencies, list)

    def test_module_info_has_reason_buffer(self):
        from titan_hcl.guardian_hcl import ModuleInfo, ModuleSpec
        spec = ModuleSpec(name="x", entry_fn=lambda *a, **k: None)
        info = ModuleInfo(spec=spec)
        # reason_buffer is a deque with maxlen 16 (SPEC §11.B step 3).
        from collections import deque
        assert isinstance(info.reason_buffer, deque)
        assert info.reason_buffer.maxlen == 16
        assert info.last_escalation_id is None

    def test_guardian_imports_supervision_messages(self):
        # Phase 11 split (§11.I): guardian.py → orchestrator/core.py.
        src = _read_source("titan_hcl/orchestrator/core.py")
        # All 6 SUPERVISION_* constants imported from .bus
        for const_name in (
            "SUPERVISION_CHILD_DOWN",
            "SUPERVISION_CHILD_RESTARTED",
            "SUPERVISION_DEPENDENCY_BLOCKED",
            "SUPERVISION_DEPENDENCY_DEGRADED",
            "SUPERVISION_DEPENDENCY_RECOVERED",
            "SUPERVISION_ESCALATION",
        ):
            assert const_name in src, (
                f"orchestrator/core.py must import {const_name} for "
                "cross-language SUPERVISION emit (SPEC §11.G.4 + §11.G.6)"
            )

    def test_guardian_restart_emits_child_down_and_restarted(self):
        # Phase 11 split (§11.I): guardian.py → orchestrator/core.py.
        src = _read_source("titan_hcl/orchestrator/core.py")
        # restart() must publish SUPERVISION_CHILD_DOWN before stop + start
        # AND SUPERVISION_CHILD_RESTARTED after successful start.
        # Pattern: both message types appear inside restart() body.
        m = re.search(
            r'def restart\(self,\s*name:\s*str[\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert m is not None, "Guardian.restart not found"
        body = m.group(0)
        assert "SUPERVISION_CHILD_DOWN" in body, (
            "Guardian.restart must publish SUPERVISION_CHILD_DOWN "
            "(per SPEC §11.B step 1 + §11.G.4 audit log)"
        )
        assert "SUPERVISION_CHILD_RESTARTED" in body, (
            "Guardian.restart must publish SUPERVISION_CHILD_RESTARTED "
            "after successful respawn (per SPEC §11.B step 5)"
        )

    def test_guardian_handles_escalation_per_spec(self):
        # Post Phase 11 split (§11.I): escalation handling lives in the
        # titan_hcl orchestrator (guardian.py was split into orchestrator/ +
        # supervisor/).
        src = _read_source("titan_hcl/orchestrator/core.py")
        # _handle_escalation must use kernel_default_decision (in-process)
        # + emit SUPERVISION_ESCALATION + handle all 3 EscalationDecision
        # variants per SPEC §11.B.1.
        m = re.search(
            r'def _handle_escalation[\s\S]+?'
            r'(?=\n    def |\Z)',
            src,
        )
        assert m is not None, "Guardian._handle_escalation not found"
        body = m.group(0)
        assert "SUPERVISION_ESCALATION" in body
        assert "kernel_default_decision" in body
        assert "EscalationDecision.CONTINUE" in body
        assert "EscalationDecision.TERMINATE" in body
        # Phase 11 (§11.I): the orchestrator must NOT self-terminate on a
        # module escalation — only the kernel (L0) may recycle the orchestrator
        # peer. A TERMINATE decision disables the offending module locally and
        # relies on the emitted SUPERVISION_ESCALATION to signal the kernel.
        assert "os._exit(64)" not in body, (
            "Orchestrator must NOT os._exit(64) on a module escalation — that "
            "would kill all sibling modules. Only the kernel may recycle the "
            "orchestrator (SPEC §11.I role split supersedes §11.B.1 step 6b "
            "self-terminate for the Python orchestrator)."
        )
        assert "ModuleState.DISABLED" in body, (
            "TERMINATE decision must disable the offending module locally "
            "instead of terminating the orchestrator process."
        )


class TestBusSpecsSupervisionParity:
    """SUPERVISION_* messages added to Python MSG_SPECS must match the
    Rust SPECS phf_map priorities (parity test from Commit 5)."""

    def test_supervision_messages_in_python_msg_specs(self):
        from titan_hcl.bus_specs import MSG_SPECS
        for msg_type, expected_priority in [
            ("SUPERVISION_CHILD_DOWN", 0),
            ("SUPERVISION_CHILD_RESTARTED", 0),
            ("SUPERVISION_ESCALATION", 0),
            ("SUPERVISION_ESCALATION_RESPONSE", 0),
            ("SUPERVISION_DEPENDENCY_BLOCKED", 0),
            ("SUPERVISION_DEPENDENCY_RECOVERED", 0),
            ("SUPERVISION_DEPENDENCY_DEGRADED", 1),  # informational
        ]:
            assert msg_type in MSG_SPECS, (
                f"{msg_type} missing from Python MSG_SPECS — parity broken"
            )
            assert MSG_SPECS[msg_type].priority == expected_priority, (
                f"{msg_type} priority mismatch with Rust SPECS"
            )


# ─── Default-off byte-identical guarantee ──────────────────────────────


class TestDefaultOffPathPreserved:
    """microkernel.l0_rust_enabled defaults to False; default path must
    remain byte-identical to pre-C-S7 behavior."""

    def test_config_default_is_false(self):
        """config.toml ships l0_rust_enabled = false."""
        src = _read_source("titan_hcl/config.toml")
        m = re.search(r'^l0_rust_enabled\s*=\s*false\s*$', src, re.MULTILINE)
        assert m is not None, (
            "config.toml must default l0_rust_enabled = false "
            "(SPEC §3.0 Running-Titans Safety Rule)"
        )

    def test_kernel_gates_use_get_with_false_default(self):
        """All kernel.py l0_rust gates default to False on missing key —
        guarantees that a config without the key still picks the legacy path."""
        src = _read_source("titan_hcl/core/kernel.py")
        # Every l0_rust_enabled lookup must specify False as the default.
        # Extract every lookup and verify.
        lookups = re.findall(
            r'\.get\(\s*["\']l0_rust_enabled["\'],?\s*([^)]+)\)',
            src,
        )
        assert lookups, "Expected l0_rust_enabled lookups in kernel.py"
        for default in lookups:
            default_clean = default.strip().rstrip(',').strip()
            assert default_clean == "False", (
                f"l0_rust_enabled lookup must default to False, got: {default!r}"
            )

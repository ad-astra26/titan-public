"""
titan_plugin/core/kernel_interface.py — KernelView typed Protocol.

Narrow, typed read-only facade for what the Plugin (and future in-process
L1/L2/L3 code paths) may read from the kernel. Used for type-checking the
L0 boundary in Phase A.

Design rationale (PLAN §3 D5):

  1. typing.Protocol — not an abstract base class. Structural typing
     means TitanKernel satisfies KernelView by duck shape alone, zero
     runtime overhead, zero inheritance coupling.
  2. Mutation methods (kernel.boot, kernel.start_modules, kernel.shutdown)
     are INTENTIONALLY NOT in this Protocol. Plugin calls those on the
     concrete TitanKernel instance; upper layers using KernelView cannot
     accidentally mutate kernel state through the typed surface.
  3. Phase C evolution: when L0 moves to Rust, PyO3 bindings will expose
     the same Protocol shape. Upper-layer Python code migrated to
     `kernel: KernelView` annotations today works unchanged tomorrow —
     only the concrete class identity changes (TitanKernel → PyO3-bound
     Rust wrapper).

This file ships zero runtime code. It's pure typing surface — importing
it adds one Protocol class to the module namespace and nothing more.

See:
  - titan-docs/rFP_microkernel_v2_shadow_core.md §L0
  - titan-docs/PLAN_microkernel_phase_a_s3.md §2.3 + §3 D5
  - titan_plugin/core/kernel.py (concrete impl that satisfies this shape)
  - titan_plugin/core/plugin.py (primary consumer via self.kernel)
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class KernelView(Protocol):
    """Read-only facade for TitanKernel state.

    Any object satisfying this Protocol is accepted as a kernel reference
    by upper-layer code that declares `kernel: KernelView`. In Phase A
    only TitanKernel satisfies it; in Phase C, a PyO3-bound Rust wrapper
    will satisfy the same shape.

    @runtime_checkable lets isinstance(kernel, KernelView) work for
    defensive boundary assertions. Zero overhead on the hot path — only
    checked at boundaries that opt in.

    Mutation is NOT exposed here. Callers that need to mutate kernel
    state (kernel.boot, kernel.start_modules, kernel.shutdown) must
    hold a concrete TitanKernel reference, not a KernelView.
    """

    # --- Infrastructure (L0-owned foundational services) ---

    @property
    def bus(self) -> object:
        """DivineBus instance — unified IPC backbone.

        Type hint is `object` rather than `DivineBus` to avoid a circular
        import at the Protocol definition site. Concrete TitanKernel
        annotates this field as `DivineBus`; mypy narrows correctly via
        structural subtyping.
        """
        ...

    @property
    def guardian(self) -> object:
        """Guardian instance — module supervisor.

        Same circular-import caveat as `bus`. TitanKernel returns
        a `titan_plugin.guardian.Guardian`.
        """
        ...

    @property
    def state_register(self) -> object:
        """StateRegister instance — legacy in-process state buffer.

        Read path for subsystems that haven't yet migrated to the shm
        RegistryBank. Will be deprecated once all readers use shm.
        """
        ...

    @property
    def registry_bank(self) -> object:
        """RegistryBank instance — /dev/shm state registry framework.

        New read path (S2) — TRINITY_STATE, NEUROMOD_STATE, EPOCH_COUNTER
        + future registries (SPHERE_CLOCKS_STATE, CHI_STATE, IDENTITY,
        INNER_SPIRIT_45D per S3b).
        """
        ...

    # --- Identity + network (may be None in limbo mode) ---

    @property
    def soul(self) -> object:
        """SovereignSoul or None (limbo mode)."""
        ...

    @property
    def network(self) -> object:
        """HybridNetworkClient or None (limbo mode)."""
        ...

    # --- Health monitors (always present) ---

    @property
    def disk_health(self) -> object:
        """DiskHealthMonitor instance."""
        ...

    @property
    def bus_health(self) -> object:
        """BusHealthMonitor instance."""
        ...

    # --- Identity + config metadata ---

    @property
    def config(self) -> dict:
        """Read-only view of the full merged config (config.toml +
        ~/.titan/secrets.toml). Mutation is discouraged; use the
        CONFIG_RELOAD bus message path for dynamic updates.
        """
        ...

    @property
    def titan_id(self) -> str:
        """Resolved titan identifier (T1 / T2 / T3).

        Source of truth: data/titan_identity.json (canonical precedence
        chain via resolve_titan_id()). Immutable post-init.
        """
        ...

    @property
    def limbo_mode(self) -> bool:
        """True if no keypair could be resolved — degraded operation.

        When True: soul is None, network is None. Kernel still boots and
        runs L0 services (bus, guardian, shm); L2/L3 subsystems that
        depend on the wallet degrade gracefully.
        """
        ...


__all__ = ["KernelView"]

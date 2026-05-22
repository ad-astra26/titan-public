# titan-rust — Microkernel v2 Phase C (Rust L0 + L1)

Phase C ships Titan's microkernel + supervision tree as a hierarchical Rust process tree, replacing today's Python L0 + L1 implementation. Behind flag `microkernel.l0_rust_enabled = false` default until C-S7 cascade.

## Canonical references (read first)

- **`titan-docs/specs/SPEC_titan_architecture.md`** v0.1.0 — single source of truth for every constant, byte layout, bus message, and supervision contract.
- **`titan-docs/specs/SPEC_titan_architecture_constants.toml`** — generates `crates/titan-core/src/constants.rs` + `titan_plugin/_phase_c_constants.py` via `arch_map phase-c regen`.
- **`titan-docs/finished/PLAN_microkernel_phase_c_l0_l1_rust.md`** — MASTER PLAN (24 D-decisions, 8-session sequence).
- **`titan-docs/PLAN_microkernel_phase_c_s2_kernel.md`** — current session PLAN (kernel binary).

## Build

```bash
# Debug (fast iteration; glibc-dynamic)
bash scripts/build_titan_rust.sh debug

# Release (production glibc-dynamic)
bash scripts/build_titan_rust.sh release

# Release musl-static (production deploy artifact for T1/T2/T3)
bash scripts/build_titan_rust.sh musl
```

## Test

```bash
# All workspace tests
cd titan-rust && cargo test --workspace

# Parity vectors (Rust ↔ Python byte-identical)
cd titan-rust && cargo test --test parity

# From repo root: pytest + cargo + arch_map enforcer
python scripts/arch_map.py phase-c verify --strict
python scripts/arch_map.py phase-c parity
```

## Crates (current — C-S2)

- **`titan-core`** — shared primitives library: frame, authkey, identity, atomic-write, shm SeqLock, supervisor primitives, bus_specs, auto-generated constants. Used by all binaries.

## Crates (planned — C-S2 onward)

| Crate | Session | Purpose |
|---|---|---|
| `titan-bus` | C-S2 | main bus broker (B.2 protocol port) |
| `titan-state` | C-S2 | shm slot registry (SPEC §7) |
| `titan-cgn` | C-S2 | CGN slot lifecycle (SPEC §18.2) |
| `titan-clocks` | C-S2 | circadian + π-heartbeat |
| `titan-kernel-rs` | C-S2 | kernel binary |
| `titan-fastbus` | C-S3 | lock-free shm ring |
| `titan-schumann` | C-S3 | 7.83 / 23.49 / 70.47 Hz generators |
| `titan-trinity-rs` | C-S3 | substrate (renamed from `titan-trinity-rs-placeholder` shipped in C-S2) |
| `titan-unified-spirit-rs-placeholder` | C-S3 | substrate→unified-spirit spawn pathway test |
| `titan-unified-spirit-rs` | C-S4 | 162D SELF orchestrator (rename from `titan-unified-spirit-rs-placeholder`) |
| `titan-trinity-daemon` | C-S5 | shared daemon library |
| `titan-inner-{body,mind,spirit}-rs` | C-S5 | inner trinity daemons |
| `titan-outer-{body,mind,spirit}-rs` | C-S6 | outer trinity daemons |

## Discipline (per `feedback_phase_c_spec_enforcement.md`)

1. **Reference SPEC at 100%** — every value cited; no inventing.
2. **`arch_map phase-c verify` before every commit** — pre-commit hook enforces.
3. **No drift from prior C-S* PLANs** — extend, never replace.

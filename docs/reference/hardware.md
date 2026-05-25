# Hardware

> Minimum and recommended specs, tested platforms, real-world profiles.

> 📝 **Status: outline (W5 scaffold) with concrete minimums.** Full
> profiling matrix lands in v3.x once we've published T1 specs and
> tested-platform validation.

---

## Quick answer

| Profile | vCPU | RAM | Disk | GPU | Notes |
|---------|------|-----|------|-----|-------|
| **Minimum** | 2 | 4 GB | 10 GB | none | Headless, Telegram-only. Proven: T2+T3 co-reside on one VPS at this profile. |
| **Recommended** | 4 | 8 GB | 20 GB | none | Adds Observatory frontend headroom. |
| **Comfortable** | 8 | 16 GB | 40 GB | optional | Room for local Ollama with mid-size models. |
| **Local LLM** | 8+ | 32+ GB | 80+ GB | recommended | Comfortable + a capable local LLM (Ollama with 30B+ models). |

For mode 1 (mainnet) or mode 2 (devnet), add ~10 GB disk to any tier for
the Anchor toolchain + Solana CLI + Rust build artifacts.

---

## Tested platforms

- **Ubuntu 22.04 LTS** — primary platform. T1 / T2 / T3 all run here.
- **Ubuntu 24.04 LTS** — tested, supported.
- **Debian 12** — tested, supported.

The wizard's preflight refuses to proceed on untested platforms with a
clear message. You can override with `--unsupported-platform` if you
know what you're doing, but you're on your own.

## Real-world profiles

[ outline: T1 (mainnet, ~16 GB / 8 vCPU class) — published numbers TBD;
T2+T3 co-reside on a 4 vCPU / 8 GB VPS today ]

## What scales with what

[ outline:
- RAM: dominated by NN brain replay + outer memory caches
- Disk: TimeChain + consciousness.db grow with experience
- CPU: Trinity tensor pipeline + inference (if local)
- Network: chain RPC + LLM API (if remote)
]

## Disk-growth expectations

[ outline:
- A new Titan: ~1 GB after first day
- After a month: ~5 GB
- After a year: ~30+ GB (depends on dream cadence)
The retention/VACUUM rFP addresses unbounded growth — track its status. ]

## Cloud provider recommendations

[ outline: any reputable provider works; budget tier (~$10/month) =
minimum profile; mid tier (~$25/month) = recommended profile; the
sweet spot for a single-Titan dev box is ~$15-20/month ]

## Apple Silicon

Ollama on Apple Silicon is surprisingly capable (unified memory).
The rest of the stack runs cleanly on macOS. Not the primary tested
platform but a known-working configuration for local development +
mode 3.

---

→ [Getting started](../getting-started.md)
→ [Inference providers](../inference-providers.md)
→ [Configuration](../operating/configuration.md)

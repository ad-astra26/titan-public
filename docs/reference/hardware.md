# Hardware

> Minimum and recommended specs, tested platforms, real-world profiles,
> what scales with what.

---

## Quick answer

| Profile | vCPU | RAM | Disk | GPU | Notes |
|---------|------|-----|------|-----|-------|
| **Minimum** | 2 | 4 GB | 10 GB | none | Headless, Telegram-only. Proven: T2+T3 co-reside on one VPS at this profile. |
| **Recommended** | 4 | 8 GB | 20 GB | none | Comfortable headroom for the full stack + the TC² web console. |
| **Comfortable** | 8 | 16 GB | 40 GB | optional | Room for local Ollama with mid-size models. |
| **Local LLM** | 8+ | 32+ GB | 80+ GB | recommended | Comfortable + a capable local LLM (Ollama with 30B+ models). |

For mode 1 (mainnet) or mode 2 (devnet), add ~10 GB disk to any tier
for the Anchor toolchain + Solana CLI + Rust build artifacts.

---

## Tested platforms

| OS | Status | Notes |
|----|--------|-------|
| **Ubuntu 24.04 LTS** | ✅ **recommended** | Native Python 3.12 — `setup_titan` runs end-to-end with no backports. The reference platform. |
| **Ubuntu 22.04 LTS** | ⚠ needs Python backport | Ships Python 3.10; Titan requires 3.11+. The fleet (T1/T2/T3) runs here via a 3.12 venv, but a fresh `setup_titan` install needs a deadsnakes 3.11+ backport first. Prefer 24.04. |
| **Debian 12** | ✅ supported | Tested clean install end-to-end. |
| Ubuntu 20.04 LTS | ⚠ deprecated | Python 3.12 not native; requires backports. Avoid. |
| Fedora 39+ | ⚠ unsupported | Likely works (Python + Rust + Solana all available) but untested. |
| RHEL/CentOS Stream | ⚠ unsupported | Same as Fedora. |
| Arch Linux | ⚠ unsupported | Should work; not tested. |
| macOS | ⚠ partial | Mode 3 works for development. Mode 1/2 untested on macOS hosts. |
| Windows (WSL2) | ❌ unsupported | systemd unit shape doesn't apply. Avoid. |

The wizard's preflight refuses to proceed on untested platforms with
a clear message. You can override with `--unsupported-platform` if
you know what you're doing; you're on your own.

---

## Real-world profiles (the three reference Titans)

Measured on the live fleet **2026-05-28** (see *Methodology* below).

| Titan | Box vCPU | Box RAM | Titan resident RAM | Disk (data/) | Inference | Network | Notes |
|-------|----------|---------|--------------------|--------------|-----------|---------|-------|
| **T1 (mainnet)** | 4 | 8 GB | **~2.3 GB** | 26 GB | Ollama Cloud (deepseek-v3.1:671b) | Helius premium RPC | Has the box to itself **+ runs the maintainer's Observatory showcase** (not part of a user install) |
| **T2 (devnet)** | 2 of 4 shared | 4 of 8 shared | ~2.3 GB | ~13 GB share | OpenRouter | Helius devnet RPC | Co-resident with T3 on one VPS |
| **T3 (devnet)** | 2 of 4 shared | 4 of 8 shared | ~2.3 GB | ~13 GB share | OpenRouter | Helius devnet RPC | Same VPS as T2 |

The headline number: **a single Titan's resident footprint is ~2.3 GB**
(measured as the sum of its ~40 `titan_hcl` cognitive workers ≈ 2.32 GB
+ 9 Rust microkernel daemons ≈ 40 MB, with the brain fully up and
`/health` returning 200). The **TC² web console** that ships with every
Titan adds negligible RAM — it is stdlib Python serving a prebuilt static
bundle. (The maintainer's separate three.js Observatory showcase, which is
*not* part of a user install, adds ~60–65 MB server-side where it runs.)

Two consequences for your sizing:

- **The 4 GB minimum is real, not marketing.** ~2.3 GB Titan + ~0.5–1 GB
  OS leaves working headroom on a 4 GB box for headless (Telegram-only)
  operation. T2+T3 co-residence on one 4 vCPU / 8 GB box (two Titans ≈
  4.6 GB resident) proves the **2 vCPU / 4 GB per-Titan minimum** holds
  through dreaming, meditation, and on-chain anchoring.
- **A full mainnet Titan + the TC² console fits the *recommended* 4 vCPU /
  8 GB tier with room to spare.** T1 — a real mainnet Titan (which also
  hosts the maintainer's heavier Observatory showcase) — runs on exactly
  that box (a DigitalOcean `s-4vcpu-8gb`). A user install is lighter still,
  since TC² replaces the Observatory. The recommended tier is recommended
  for comfort and dream-peak headroom, not because the floor is higher
  than it looks.

> **Methodology.** Box specs from `nproc` + `free -m`; per-Titan resident
> RAM from summing RSS of the `titan_hcl` worker processes + `titan-*-rs`
> daemons (`ps -eo rss,comm`); disk from `du -sh data/`. Captured during
> normal operation, not a synthetic benchmark. **Caveat (why T2/T3 are the
> cleaner reference):** T1 shares its box with the development environment,
> ad-hoc builds, and live editor/agent sessions, so its *box-level* `free`
> reading is noisy — the ~2.3 GB figure is the isolated Titan process-tree
> sum, which that noise does not affect. A gold-standard per-process RSS
> profile sampled over a clean 24 h window (no dev session) on a headless
> T2/T3-class box remains the ideal future measurement; the numbers here
> are conservative and already representative.

---

## What scales with what

### RAM is the most constrained resource

Dominated by:

- **NN brain replay** at boot (~1.5 GB for the standard set of models)
- **Outer-memory caches** in DuckDB (~500 MB at steady state, larger
  during recall storms)
- **FAISS indices** (~100 MB for a year-old Titan; grows
  sub-linearly thanks to vector quantization)
- **PyTorch + TorchRL working set** (~800 MB)
- **Inference provider** if local Ollama (multi-GB depending on model)

### Disk grows with experience

- **Day 1:** ~1 GB
- **Month 1:** ~5 GB
- **Year 1:** ~30+ GB (depends on dream cadence + cleanup policy)

The unbounded-growth issue is tracked in the
**retention/VACUUM rFP** (internal). Until that ships, plan ahead:
budget 50 GB disk for a year-long Titan, or set up periodic archive
to Arweave.

### CPU correlates with cognition density

The Trinity tensor pipeline (Rust) is the steady-state CPU consumer
(~30% of one vCPU). Inference (Python or Ollama) spikes during chat
turns and dreams. The TC² web console uses negligible CPU (stdlib Python
serving a static bundle).

### Network bandwidth

- **Steady state:** minimal (heartbeats, occasional chain RPC)
- **Dreaming/meditation:** spike for on-chain `commit_state` calls
  (modes 1/2) and Arweave uploads (rare)
- **Inference if remote:** depends on token volume per request

A 10 Mbps connection is sufficient for a single Titan with OpenRouter
inference. 1 Mbps suffices for Ollama-local Titans (the bulk of the
inference cost is local).

---

## Disk-growth expectations + retention strategy

Concretely, what grows:

- `data/consciousness.db` — ~200 MB/month at default cadence
- `data/inner_memory.db` — ~50 MB/month
- `data/titan_memory.duckdb` — ~100 MB/month (outer memory)
- `data/memory_vectors.faiss` — grows sub-linearly via quantization
- `/tmp/titan_brain.log` — rotated weekly by default (`logrotate`)

Mitigation today (until the retention rFP ships):

```bash
# Set retention via env var (in ~/.titan/secrets.toml):
[runtime]
consciousness_db_retention_days = 30   # default: unbounded
observatory_db_retention_days = 60
```

After change, restart Titan. The cleanup runs during the next
meditation cycle.

---

## Cloud provider recommendations

Any reputable provider works. Pricing/configuration as of May 2026:

| Provider | Tier | $/month | vCPU | RAM | Disk | Notes |
|----------|------|---------|------|-----|------|-------|
| Hetzner | CX22 | ~$5 | 2 | 4 GB | 40 GB | Min tier; perfect for mode 3 or single-mode-3 Titan |
| Hetzner | CX32 | ~$9 | 4 | 8 GB | 80 GB | **Recommended tier**; comfortably runs a full Titan + the TC² console |
| Hetzner | CCX13 | ~$14 | 2 | 8 GB | 80 GB | Dedicated CPU; smoother runtime than shared |
| DigitalOcean | s-2vcpu-4gb | $24 | 2 | 4 GB | 80 GB | More expensive than Hetzner but better US presence |
| DigitalOcean | s-4vcpu-8gb | ~$48 | 4 | 8 GB | 160 GB | **Proven** — T1 (full mainnet Titan + the maintainer's Observatory showcase) runs here |
| AWS Lightsail | 2GB | $10 | 2 | 2 GB | 60 GB | Tight on RAM; not recommended |
| Linode | Nanode | $5 | 1 | 1 GB | 25 GB | Too small for Titan |
| Vultr | High Perf 2vCPU | $12 | 2 | 4 GB | 64 GB | Works for minimum profile |

**Sweet spot for a single-Titan box:** $10–15/month tier with 4
vCPU + 8 GB. Hetzner CX32 is our standard recommendation.

For a multi-Titan box (T2+T3 model): 4 vCPU + 8 GB shared is the
proven config. Don't try to fit three or more Titans on the same
hardware below 16 GB.

---

## Apple Silicon (development only)

Ollama on Apple Silicon (M2/M3/M4) is surprisingly capable — the
unified memory architecture runs large quantized models well. M2 Pro
with 32 GB unified memory handles `llama3.3:70b` quantized to Q4_K_M
at usable speed.

The rest of the stack runs cleanly on macOS. Not the primary tested
platform but a known-working configuration for:

- **Local development** (mode 3)
- **Inference-only testing** (Ollama as an inference backend for a
  remote-host Titan)

NOT recommended for production mode-1/2 Titans (the systemd unit and
some VPS-assuming scripts don't apply).

---

## GPU? Generally no

Titan does not require a GPU. The Rust microkernel and the cognitive
pipeline run on CPU. The only GPU-relevant component is the
inference provider:

- **Cloud inference (Ollama Cloud / OpenRouter / future OpenAI):**
  GPU is on the provider's side, not yours.
- **Local Ollama:** GPU helps for >7B models. A modest GPU (e.g.,
  RTX 3060 12GB) handles models up to ~13B comfortably; for 70B+,
  you need 24GB+ VRAM.

If you have a GPU, Ollama auto-detects and uses it; no extra config.

---

## Before installing — checklist

Before running `setup_titan`, verify your hardware:

```bash
# CPU
nproc                                  # vCPU count
lscpu | grep "Model name"             # CPU model

# RAM
free -h                                # available RAM

# Disk
df -h /                                # root partition free space

# Network
curl -s -o /dev/null -w "%{time_total}\n" https://api.solana.com   # RPC latency

# OS
cat /etc/os-release | grep PRETTY_NAME
```

If any of these fall below the **minimum** tier, the wizard will
refuse. If they're between minimum and recommended, you'll see a
soft warning.

---

→ [Getting started](../getting-started.md)
→ [Inference providers](../inference-providers.md)
→ [Configuration](../operating/configuration.md)

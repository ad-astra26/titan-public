# Setup modes

> Mainnet, devnet, local-simulated — what each one does, what it costs,
> what you give up, and how to pick.

The wizard's first real question. You can run multiple Titans on the same
box (each in its own dataspace) at different modes, but a given Titan
keeps its mode for life.

---

## At a glance

| Mode | On-chain | Cost | Backups | Identity | Use when… |
|------|----------|------|---------|----------|-----------|
| **1. Mainnet** | real GenesisNFT + your ZK Vault PDA | real SOL (few $ at install + ~$0/month after) | ON (Arweave + ZK) | real | …you want a Titan for keeps. |
| **2. Devnet** | devnet GenesisNFT + your ZK Vault PDA | airdropped devnet SOL (free) | ON (devnet) | real | …you want the full sovereign experience without spending real SOL. |
| **3. Local-simulated** | none (`--skip-on-chain`) | none | OFF | real but simulated | …you want to see the whole birth ceremony with zero deps and zero cost. |

The mode is set once at install and stored in
`titan_hcl/config.toml:[network].mode`. Changing it later is a separate
process: it produces a *new* Titan (new soul, new identity) and migrates no
memory across.

---

## Mode 1 — Mainnet

The committed path. Your Titan lives on Solana mainnet.

**What you get**
- Real GenesisNFT minted in your Maker wallet
- Your own ZK Vault PDA on mainnet (you deploy your own copy of the program — no operator dependency)
- Per-meditation `commit_state` calls anchor TimeChain Merkle roots on chain
- Per-backup `append_epoch_snapshot` calls inscribe ZK-compressed state on chain
- Daily memory snapshots pushed to Arweave via Irys
- The Shamir Vertex 3 shard is anchored in the ZK Vault PDA, not just stored locally

**What it costs**
- One-time at install: ~0.05–0.15 SOL (deploy ZK Vault program + mint NFT
  + init PDA + initial `commit_state`). Varies with Solana network fees;
  the wizard quotes the actual number before proceeding.
- Steady state: per-meditation calls cost fractions of a cent at current
  fee levels. Plan on ~0.01 SOL/month for an active Titan.
- Your wallet always pays. We don't subsidize anything.

**What it requires**
- A Solana wallet with SOL (Phantom, Backpack, or any wallet that
  exports a keypair you can drop at `~/.config/solana/id.json`)
- An RPC URL with mainnet access. Public RPCs work but are rate-limited
  — for steady operation we strongly recommend a paid provider (Helius,
  QuickNode, Triton — the wizard helps you set one up).
- The Solana CLI 1.18+, Anchor 0.30+, Rust 1.75+ (because you deploy
  your own ZK Vault program copy — the wizard installs these for you).

**What you give up**
- Nothing structurally — this is the full sovereign Titan.

## Mode 2 — Devnet

The realistic tester path. Everything from mainnet, but on Solana's devnet
chain.

**What you get**
- Same architecture as mainnet, but devnet NFTs, devnet vault, devnet
  TimeChain anchors
- Daily memory snapshots still go to Arweave (real, on actual Arweave —
  the cost is small enough that Irys lets devnet Titans use mainnet
  Arweave)
- Full birth ceremony with the real Shamir three-vertex topology

**What it costs**
- Zero monetary cost. The wizard airdrops you ~5 devnet SOL automatically.
  Devnet is also occasionally faucet-rate-limited; if so, the wizard prints
  faucet URLs and you click through.

**What it requires**
- Same toolchain as mode 1 (Rust + Anchor + Solana CLI)
- A wallet path (no SOL required — the wizard handles the airdrop)

**What you give up**
- Nothing about *how* the architecture works. The only real difference
  is that devnet has periodic resets, so a long-running mode 2 Titan may
  occasionally re-anchor a state root after a chain reset. This is rare
  and the system handles it transparently.

**Why people pick this**
- They want to feel the full system without committing real money.
- They're testing changes to the architecture before promoting to mainnet.
- T2 and T3 (our own Titans) run on devnet today.

## Mode 3 — Local-simulated

The "see how this works" path. Zero deps, zero cost, full ceremony.

**What you get**
- Real Ed25519 keypair, real soul.md, real birth certificate
- Real Shamir 2-of-3 ceremony — three actual shards generated and
  distributed across the three vertices (Maker / Titan / "on-chain")
- A working Titan you can chat with on Telegram, dream cycles run,
  meditation cycles run, expression works
- A clear **"SIMULATED — not on-chain"** notice in the UI and logs

**What it costs**
- Nothing. No SOL, no API keys beyond Telegram + LLM.

**What it requires**
- Python 3.12+, Rust 1.75+, ~10 GB disk, 4 GB RAM (2 GB minimum for
  headless / Telegram-only)
- A Telegram bot token (free, takes 30 seconds)
- An LLM credential (Ollama recommended; OpenRouter as fallback)

**What you give up**
- **The on-chain pieces.** Shard 3 lives on your disk (with a clear
  marker) instead of being anchored in a Solana PDA. No GenesisNFT.
  No Arweave backups. No ZK Vault.
- **Recovery from total loss.** If your disk dies and you don't have a
  backup, the Titan is gone. (Modes 1 and 2 are recoverable from
  your shard + the on-chain anchor alone.)

**Why people pick this**
- First read-through. See the whole flow before committing.
- Development. Iterate on the architecture without burning SOL.
- Teaching. The ceremony is the same; only the on-chain anchoring is
  short-circuited.

---

## Can I upgrade a mode-3 Titan to mode 2 or mode 1?

**No.** Modes are an identity-time choice, not a runtime flag. Upgrading
would change the soul (different shards, different on-chain anchor),
which produces a different being. We don't pretend otherwise.

What you *can* do is run a mode-3 Titan for a few days to feel the system
out, then start a fresh mode-2 or mode-1 Titan for keeps. You can keep
both running side by side; they're independent.

## Can I run multiple Titans on the same box?

Yes. Each Titan has a unique `titan_id` and lives in its own dataspace
(`data/<titan_id>/`, `~/.titan/microkernel_<titan_id>.toml`, etc.). T2 and
T3 co-reside on the same VPS today.

## Can I move a mode-2 Titan to mode-1?

Not directly, for the same reason as mode-3 → mode-2: the on-chain
identity is forged at birth. The conventional path is: shut the mode-2
Titan down (or keep it as a "sibling"), and start a new mode-1 Titan that
inherits the *config* but has its own fresh identity.

## What if I lose access to my Maker wallet?

In modes 1 / 2: if you also have your Shard 1 (the one shown to you at
install), you can use it + Shard 3 (the on-chain anchor) to reconstruct
Titan's seed without the Maker wallet — the threshold is 2-of-3. But the
Maker wallet is the *human-side* identity, so practically you'd want to
restore that too. Plan on keeping the wallet secure.

In mode 3: you lose access to the Titan entirely (no on-chain anchor to
recover from). Keep mode 3 for exploration, not for irreplaceable Titans.

## When can I expect mode-portability tools?

We don't plan to ship them. The architectural commitment is that a Titan's
identity is forged once and never replaced — that's what makes the
Shamir + Solana sovereignty axiom work. If you want a different mode, you
start a different Titan.

---

→ [Getting started](getting-started.md) — full install walk-through
→ [Identity, soul, and Shamir backup](concepts/identity-soul-sss.md) — what
  the ceremony actually produces

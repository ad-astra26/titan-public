# Identity, soul, and Shamir backup

> The birth ceremony — what it produces, why each piece exists, and how
> recovery works.

> 📝 **Status: outline (W5 scaffold).** Full walkthrough lands in v3.x.
> Until then, see [Setup modes](../setup-modes.md) and
> [Getting started — step 4](../getting-started.md#step-4--the-genesis-ceremony)
> for the operational view.

---

## What this covers

When you bring up a Titan, a ceremony runs that produces five durable
artifacts: a keypair, a soul, a birth certificate, three Shamir shards,
and (in modes 1/2) an on-chain GenesisNFT + ZK Vault PDA. This page
explains each piece, why it exists, and how they interact to make a
Titan recoverable from very little.

---

## The three vertices of Shamir 2-of-3

- **Vertex 1 — Maker (you).** Offline shard, shown to you exactly once
  during setup. **Never logged.**
- **Vertex 2 — Titan / VPS.** Local encrypted shard.
- **Vertex 3 — On-chain.** Anchored in the ZK Vault PDA (modes 1/2) or
  flagged "SIMULATED" locally (mode 3).

Any two of three reconstruct the seed.

## The soul

[ outline: content-addressed identity rooted in Maker wallet + Titan
keypair + birth timestamp; never changes for the lifetime of the Titan ]

## The birth certificate

[ outline: file format, what's recorded, where it lives, Arweave mirror ]

## The keypair

[ outline: Ed25519, where it lives, permissions, rotation policy (locked
post-genesis) ]

## Recovery scenarios

- VPS disk loss + Maker wallet intact + you have your Shamir Vertex 1 shard → recover from on-chain + Vertex 1
- Maker wallet loss + Vertex 1 shard intact + Vertex 3 on-chain → recover the seed; rotate the human wallet
- Catastrophic on-chain reset (devnet only) → re-anchor from Vertex 1 + Vertex 2
- Mode 3 (local-simulated) → no on-chain anchor; no recovery from disk loss

The general guarantee: **any two of three vertices is enough.**

## What `setup_titan --restore` does

[ outline: how to invoke, what it asks for, what it produces ]

---

→ [Setup modes](../setup-modes.md)
→ [Getting started](../getting-started.md)
→ [Backup and recovery](../operating/backup-recovery.md)

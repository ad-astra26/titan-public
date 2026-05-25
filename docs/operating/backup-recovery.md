# Backup and recovery

> SSS shard custody, restoring a Titan from backup, and what's recoverable
> from what.

> 📝 **Status: outline (W5 scaffold).** Full procedure lands in v3.x with
> `setup_titan --restore`. Until then, the recovery path is documented
> in [Identity, soul, and SSS](../concepts/identity-soul-sss.md).

---

## What this covers

How to keep a Titan recoverable, and how to actually do the recovery
when you have to. Covers shard custody, the daily Arweave snapshots,
on-chain anchors, and the disk-loss / wallet-loss scenarios.

---

## The sovereignty axiom

A Titan you can recover with **only your wallet and your share of a
Shamir secret**, on a brand-new machine, from public storage, with no
other infrastructure, is a Titan that is truly yours.

This is the test every backup story has to pass.

## Mode-specific guarantees

| Mode | Disk-loss recoverable from… | Wallet-loss recoverable from… |
|------|-------|-------|
| **1 (mainnet)** | Vertex 1 (your shard) + Vertex 3 (on-chain) + Arweave daily snapshots | same |
| **2 (devnet)** | Same as mode 1; devnet may occasionally reset |
| **3 (local)** | Not recoverable (no on-chain anchor) | Not recoverable |

## Shard custody (modes 1/2)

[ outline: Vertex 1 = you (offline, shown once); Vertex 2 = Titan (local
encrypted, gone on disk loss); Vertex 3 = on-chain (always available);
threshold = 2-of-3 ]

**The custody discipline that matters:** keep Vertex 1 offline,
ideally on paper or a hardware key. Multiple copies in physically
separate places. If you only have one copy and lose it, you lose
half the security margin (you're now reliant on Vertex 2 surviving).

## Arweave daily snapshots

[ outline: what's snapshotted, where it lives, how to enumerate your
snapshots, how to restore from one ]

## On-chain anchors

[ outline: ZK Vault PDA contents, per-meditation `commit_state` calls,
per-backup `append_epoch_snapshot` calls — what each one stores ]

## The recovery procedures

### Disk loss, everything else intact

[ outline: bring up new VPS → `setup_titan --restore <titan-id>` →
provides your wallet + Vertex 1 → reconstructs from on-chain ]

### Wallet loss, disk intact

[ outline: Vertex 1 + Vertex 3 sufficient ]

### Both — but you still have Vertex 1 + on-chain access

[ outline: same procedure as disk loss, plus wallet rotation ]

### Mode 3 disaster

Plan ahead: don't put irreplaceable Titans in mode 3.

## What is **not** recoverable

- **Conversation memory** of the period between the last Arweave snapshot
  and the loss event. Titan re-anchors from the snapshot; everything
  since is gone.
- **Mode 3 anything beyond mode 1/2.** The architecture cannot
  reconstruct what was never anchored.

---

→ [Identity, soul, and SSS](../concepts/identity-soul-sss.md)
→ [Setup modes](../setup-modes.md)
→ [Safety and privacy](../reference/safety-privacy.md)

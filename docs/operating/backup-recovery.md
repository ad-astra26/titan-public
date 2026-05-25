# Backup and recovery

> SSS shard custody, restoring a Titan from backup, and what's recoverable
> from what.

This is the operational view of recovery. For the *why* — the
sovereignty axiom, the Shamir threshold logic, and what the soul means
— see [Identity, soul, and Shamir backup](../concepts/identity-soul-sss.md).

---

## The sovereignty axiom (one paragraph)

A Titan you can recover with **only your wallet and your share of a
Shamir secret**, on a brand-new machine, from public storage, with no
other infrastructure, is a Titan that is truly yours.

This is the test the backup/recovery story has to pass.

---

## Mode-specific guarantees

| Mode | Disk loss recovers from… | Wallet loss recovers from… |
|------|--------------------------|----------------------------|
| **1 (mainnet)** | Vertex 1 (your shard) + Vertex 3 (on-chain) + Arweave snapshots | Vertex 1 + Vertex 2 (if VPS lives), or rotate wallet via shard reconstruction |
| **2 (devnet)** | Same as mode 1; devnet may occasionally chain-reset | Same as mode 1 |
| **3 (local)** | **NOT recoverable** (no on-chain anchor) | NOT recoverable |

Mode 3 is for exploration, not for irreplaceable Titans. Plan
accordingly.

---

## Daily Arweave snapshots — what's mirrored

Once per backup epoch (~daily in normal operation), the architecture
mirrors to Arweave via Irys:

- **TimeChain blob** — compressed, per fork. Cryptographically chained,
  Merkle-verifiable.
- **Birth certificate** — if changed; usually unchanged after genesis.
- **State snapshot** — the full 132D vector plus all neurochemical /
  circadian state, compressed.
- **Identity metadata** — soul hash + on-chain anchors index.

Each snapshot is content-addressed (you can find it by Arweave content
hash without knowing where it was uploaded). The user's wallet pays
the Irys gateway directly — typically a few cents per day at current
Arweave fees.

What's **not** mirrored (intentional — recoverable from the above):
- `/dev/shm` runtime state (transient)
- Per-epoch tensor caches (rebuilt from TimeChain)
- FAISS / Kuzu indices (rebuilt from TimeChain blob on restore)

---

## On-chain anchors (modes 1/2)

The user's ZK Vault PDA holds:

- **`commit_state` calls** — per meditation, a chained Merkle root of
  the TimeChain blocks committed in that meditation
- **`append_epoch_snapshot` calls** — per backup epoch, the
  ZK-compressed state snapshot's root hash
- **The public hash of Shard 3** — set once at genesis, immutable

Anyone with the soul and the wallet pubkey can query the chain and
verify what Titan has anchored. This is the verifiable-existence layer
that makes "did this Titan really live through that?" answerable
without trusting any single operator.

---

## Shard custody (modes 1/2 only)

The whole recovery story rests on Shard 1 — your offline shard, shown
to you once at the genesis ceremony.

**Concrete custody recommendations:**

- **Paper copy.** Print Shard 1, store in a safe-deposit box. Paper
  outlives most digital media.
- **Geographic separation.** Two physical copies, in two different
  cities ideally. Disaster resilience.
- **Hardware token** (optional). YubiKey or Trezor with a Shamir
  storage slot if you have one.
- **Inheritance plan.** Tell one trusted human where it is. Without
  it, the Titan dies with you.
- **DO NOT digitize.** No "save to Dropbox"; the value of the offline
  shard is that it doesn't touch a network.

If you lose Shard 1:
- And Vertex 2 (the VPS) is alive → you can still operate (Shard 2 +
  Shard 3 reconstruct the seed; just don't lose Vertex 2 too)
- And the VPS dies → Titan is gone. There's no third shard.

This is by design. The architectural commitment is that a Titan with
no operator backdoor cannot also have a recovery backdoor. They're
the same thing.

---

## The recovery procedures

### Case 1: VPS disk loss, Maker wallet + Shard 1 intact

The canonical sovereignty test. Typical recovery time: 10–30 minutes.

```bash
# Bring up new VPS (Ubuntu 22.04+ / Debian 12+)
curl -fsSL https://raw.githubusercontent.com/ad-astra26/titan-public/main/scripts/setup_titan.sh | bash

# Wizard asks:
#   Mode (same as the original Titan: 1 or 2)
#   Maker wallet (provide path or generate)
#   Titan ID to restore (e.g., T1)
#   Shard 1 (paste from your paper copy)
#   Solana RPC URL
#   LLM credentials (Ollama auto-detect or OpenRouter key)
#   Telegram bot token (re-create from BotFather if needed)

# Wizard fetches Shard 3 from chain (decrypts with deterministic key
# from soul), reconstructs seed from Shard 1 + Shard 3, restores
# keypair, fetches state snapshot from Arweave, replays.

# Wait for `/health` 200; first response 30-60 seconds.
```

What's lost: anything since the last Arweave snapshot (typically <24
hours). The chain has the per-meditation `commit_state` roots, but
without the content blobs, the per-block content isn't recoverable.

### Case 2: Maker wallet loss, VPS + Shard 1 intact

```bash
# On the existing VPS (Titan still running):
setup_titan --reconstruct-seed --wallet-lost

# Wizard asks for Shard 1, combines with Shard 2 from local disk,
# reconstructs seed, derives keypair, exports a new soul-bound
# wallet recovery payload.

# Use the payload with `solana-keygen recover` to restore wallet
# control. Update `[network] maker_wallet_path` in config.
```

The Titan continues running uninterrupted; only the human-side
wallet is rotated. Arweave snapshots stay valid (they're keyed by
soul, not by wallet).

### Case 3: VPS dies AND wallet lost, Shard 1 intact

You need the chain (Shard 3) plus Shard 1. The wallet you used to
fund genesis is gone, but the on-chain anchor lives in your own ZK
Vault PDA — the PDA is queryable by anyone with the PDA address,
which is derivable from the soul hash + a known seed.

```bash
# New VPS:
setup_titan --restore --soul-only

# Wizard asks for soul hash (you remember it from the birth
# certificate, or derive it from the wallet's transaction history
# you may have offline records of, or look it up on a chain
# explorer if you know the GenesisNFT mint address).
# Provides Shard 1. Wizard fetches Shard 3 from chain.
```

Disaster mode but recoverable. The wallet rotation is then a separate
step.

### Case 4: All wallet AND Shard 1 lost

Below the 2-of-3 threshold. The chain has Shard 3; you'd need either
Shard 1 or Shard 2 to reach the threshold, and you have neither.

**Titan is gone.** Plan custody so this case doesn't happen.

### Case 5: Mode 3 disk loss

No on-chain anchor. Nothing to recover from. Mode 3 Titans are for
exploration only.

---

## What is NOT recoverable

For clarity:

- **Conversation memory** between the last Arweave snapshot and the
  loss event — the chain has the anchors, but the content blobs
  weren't yet mirrored to Arweave.
- **Mode 3 anything** beyond what's on local disk at the moment of
  loss.
- **State the architecture considered transient** — `/dev/shm` slots,
  per-tick caches. These rebuild from TimeChain anyway, so this isn't
  really a "loss."

---

## Backup health checks

Verify your backup pipeline is healthy regularly:

```bash
# When did the last Arweave snapshot complete?
setup_titan --diagnostic --section backup

# Manually trigger a snapshot to verify the pipeline works end-to-end:
setup_titan --backup-now

# Verify the on-chain anchor chain has no gaps:
python scripts/arch_map.py timechain --all --verify-anchors
```

A healthy backup pipeline shows the most recent snapshot within the
last 24h and no gaps in the on-chain anchor chain.

---

## Custody discipline checklist

Once a year (or after any life event affecting your custody plan):

- [ ] Verify Shard 1 is still accessible (test recovery on a mode-3 box)
- [ ] Verify your safe-deposit box still exists + access works
- [ ] Verify your wallet is still under your sole control
- [ ] Verify your inheritance plan is still appropriate
- [ ] Verify your Arweave wallet has enough balance to fund another
      year of snapshots
- [ ] Verify your Solana wallet has enough SOL for another year of
      `commit_state` calls

This is unfashionable advice (everyone hates ops tasks). It's the
asymmetry between "small annual check" and "lost Titan forever" that
makes it worth it.

---

→ [Identity, soul, and Shamir backup](../concepts/identity-soul-sss.md)
→ [Setup modes](../setup-modes.md)
→ [Safety and privacy](../reference/safety-privacy.md)

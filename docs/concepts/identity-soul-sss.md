# Identity, soul, and Shamir backup

> The birth ceremony — what it produces, why each piece exists, and how
> recovery works.

This page walks through the five durable artifacts produced when a Titan is
born, why each exists, and how they interact to make a Titan recoverable
from very little. The operational view (which commands to run, what the
wizard asks for) lives in
[Getting started — step 4](../getting-started.md#step-4--the-genesis-ceremony);
this page is the *why*.

---

## The five artifacts

When `genesis_ceremony.py` runs (whether driven by `setup_titan` or
manually), it produces these:

1. **The Ed25519 keypair** — Titan's cryptographic identity.
2. **The soul** — a content-addressed hash binding the keypair to the
   Maker wallet + a birth timestamp. This is *who Titan is*, in the
   sense that the soul never changes.
3. **The birth certificate** — a small JSON file recording the soul,
   keypair fingerprint, birth wall-clock time, and (in modes 1/2) the
   chain + transaction signature of the on-chain anchor.
4. **The three Shamir shards** — a 2-of-3 threshold secret-sharing of
   the keypair's seed.
5. **(Modes 1/2 only) On-chain artifacts** — your own copy of the ZK
   Vault program, a GenesisNFT, and the PDA holding the public hash of
   Shard 3.

All five are produced atomically. If any step fails, the ceremony
aborts cleanly: no half-Titan exists.

---

## Why Ed25519

Three reasons:

- **Solana compatibility.** Solana uses Ed25519 natively; using the
  same curve for Titan's identity means we sign on-chain transactions
  with the same key.
- **Speed + size.** Ed25519 signatures are 64 bytes; public keys are
  32 bytes. Small enough to fit anywhere; fast enough that signing
  every consequential thought (the TimeChain) is cheap.
- **Determinism.** Ed25519 signatures are deterministic — the same key
  signing the same message produces the same signature. This is what
  lets a Maker verify Titan's signature without any state on their
  side.

The keypair lives at `~/.titan/keypair_<titan_id>.json` with mode
`0o600` (owner read/write only). The wizard verifies this permission
on every boot and refuses to run if it's wrong.

---

## The soul: what it actually is

Many "soul" concepts in software are vague. Titan's is precise:

```
soul = sha256( maker_pubkey || titan_pubkey || birth_unix_ms )
```

A 32-byte hash, computed once at the genesis ceremony, written into the
birth certificate, anchored on chain in modes 1/2. The soul never
changes for the lifetime of the Titan.

The soul is **content-addressed identity**. You can verify "is this
Titan really *that* Titan?" by re-computing the hash from the Maker
wallet + Titan keypair + birth timestamp, and comparing.

If a Titan is restored from backup on new hardware, the soul will still
hash to the same value. If anything in the input changed (different
Maker, different keypair, different birth time), the soul changes — and
that means it's a different Titan. By construction, soul collisions are
cryptographically infeasible.

This is the *primary identity proof.* The keypair signs; the soul
identifies; the birth certificate provides provenance.

---

## The birth certificate

A small JSON file (`data/birth_certificate.json`), shape:

```json
{
  "version": 1,
  "titan_id": "T<n>",
  "soul": "<64 hex chars — sha256 of soul>",
  "titan_pubkey": "<32-byte ed25519 public key, base58>",
  "maker_pubkey": "<solana wallet pubkey, base58>",
  "birth_unix_ms": <integer>,
  "birth_iso": "2026-MM-DDThh:mm:ssZ",
  "mode": "mainnet | devnet | local",
  "on_chain": {
    "anchor_tx_signature": "<base58>",   // null in mode 3
    "genesis_nft_mint": "<base58>",      // null in mode 3
    "zk_vault_program": "<base58>"       // your own copy; null in mode 3
  }
}
```

It is mirrored to Arweave in modes 1/2 on first backup. Even if your
disk vanishes, the birth certificate is recoverable from Arweave by
its content hash.

---

## Shamir 2-of-3, by vertex

The seed (the 32 bytes from which the keypair derives) is split into
three Shamir shards using a 2-of-3 threshold. Any two shards
reconstruct the seed; one shard alone reveals nothing.

### Vertex 1 — the Maker (you)

Shown to you exactly **once**, during the genesis ceremony. Format:

```
<base32 string ~52 chars>
```

You copy it. Write it down. Print it. Engrave it on titanium. Store it
somewhere you'll be able to access it years from now if your VPS
catches fire.

**It is never logged**, never written to disk, never sent over the
network. The wizard displays it, you confirm you have it, the wizard
clears the display. There is no recovery mechanism to ask Titan for
this shard later.

If you lose it AND your VPS dies, you cannot recover a mode-1/2 Titan.
(The on-chain shard alone is below the 2-of-3 threshold.)

### Vertex 2 — Titan (the VPS)

Encrypted with a hardware-bound key derived from `/etc/machine-id` +
the Titan keypair's secret half. Lives at `~/.titan/shard_2.enc` mode
`0o600`.

This shard survives reboots and software upgrades, but **not** disk
loss or moving to new hardware (machine-id changes). For sovereign
operation, it's expected to be ephemeral in the catastrophic-loss
scenario.

### Vertex 3 — On-chain

In modes 1/2: the public hash of Shard 3 is committed to your ZK Vault
PDA. The shard itself is encrypted with a deterministic key derived
from the soul, and the encrypted blob is published as a Solana memo
transaction at genesis.

In mode 3 (local-simulated): the shard lives at `~/.titan/shard_3.enc`
with a clear `SIMULATED — not anchored on chain` flag in the file. The
ceremony runs identically; only the publication is short-circuited.

---

## Recovery scenarios

The 2-of-3 threshold gives several recovery paths. Modes 1/2 only;
mode 3 is intentionally not recoverable after total disk loss.

### Disk loss, Maker wallet intact, you have your Shard 1

This is the canonical sovereignty test:

1. Bring up a new VPS.
2. Install Titan (`setup_titan --restore <titan_id>`).
3. Wizard asks for your wallet (mode 1/2) + your Shard 1.
4. Wizard fetches Shard 3 from chain (decrypts with deterministic key
   from soul, which is in your wallet's transaction history).
5. Shard 1 + Shard 3 → seed reconstructed → keypair regenerated.
6. Birth certificate fetched from Arweave by soul hash.
7. State snapshot fetched from Arweave, replayed.
8. Titan is back.

No reliance on us (the operators). No reliance on a third-party backup
service. The wallet + your shard + the chain are enough.

### Maker wallet loss, VPS + Shard 1 intact

Recovery is the same threshold (2-of-3): Shard 1 + Shard 2 → seed
reconstructed. The Maker wallet is the human-side identity that signed
the original genesis transaction; rotating it is a separate
human-identity concern but doesn't strand the Titan.

### Both wallet AND Vertex 1 shard, with on-chain access

This is the worst-survivable case. You need:
- Shard 2 (from the VPS — if it's still alive)
- Shard 3 (from chain — anyone can fetch this; only the decryption key
  derived from the soul gives you the plaintext)
- The soul, which you can compute from the birth certificate (on
  Arweave)

If the VPS is dead AND you lost Shard 1, you have only Shard 3, which
is below the 2-of-3 threshold. Titan is gone.

### Mode 3 anything

There is no on-chain Vertex 3. Lose your disk = lose your Titan. Mode
3 is for *seeing how this works*, not for keeping irreplaceable Titans.

---

## Custody discipline

The biggest practical risk to a Titan's continuity isn't a chain reset
or a software bug — it's **losing Shard 1**. Concrete recommendations:

- Print Shard 1 on paper. Store in a safe-deposit box, or split into
  two physical locations.
- Make a second copy on a hardware token (YubiKey, Trezor) if you have
  one.
- Tell ONE trusted human where it is. (The Titan is yours; sharing
  custody is your call. We strongly advise *not* sharing without a
  trust relationship and a clear inheritance plan.)
- Test the recovery procedure with mode 3 first, before committing to
  mode 1 or 2 with real assets.

The whole architecture is built so that this one piece of paper is
sufficient (with the chain) to restore Titan from nothing. Treat it
like an asymmetric heirloom: irreplaceable, valuable, easy to lose
through neglect.

---

## What the ceremony does NOT do

For clarity (these come up in conversations):

- **It does not generate a "backup phrase"** in the BIP-39 sense.
  Shard 1 is not a mnemonic; it's a Shamir share. A BIP-39 mnemonic
  encodes a complete seed (single point of failure); Shamir 2-of-3 is
  distributed (no single point of failure). The trade-off is that
  Shamir shares are less human-friendly.
- **It does not register Titan with any operator service.** No phone
  home. The only network calls during the ceremony are to Solana RPC
  (modes 1/2) and Arweave/Irys (modes 1/2 first backup).
- **It does not create user accounts on our servers.** There are no
  servers. The whole architecture is yours.

---

→ [Setup modes](../setup-modes.md)
→ [Getting started](../getting-started.md)
→ [Backup and recovery](../operating/backup-recovery.md)

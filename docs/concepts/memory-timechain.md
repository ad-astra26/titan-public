# Memory and the TimeChain

> The five forks, Arweave mirroring, and the TimeChain as Titan's
> canonical record.

Titan's memory is multi-layered: a substrate of four stores (SQLite,
Kuzu, FAISS, DuckDB), a canonical order (the TimeChain), and a permanent
mirror (Arweave). On top of that substrate, memory differentiates into
five canonical forks plus a conversation fork. This page explains each
layer, each fork, and how they interact during recall, dreaming, and
backup.

For the *learning* dimension (how memory accumulates into earned
skill), see [Learning and the Synthesis Engine](learning-and-synthesis.md).

---

## The substrate

| Store | Role |
|-------|------|
| **SQLite** | TimeChain — cryptographically-signed, Merkle-rooted record. The canonical order. |
| **Kuzu** | Graph relationships — concepts, skills, episodes, edges between them. |
| **FAISS** | Vector index for similarity lookup. |
| **DuckDB** | Analytics + ACT-R activation state + epoch telemetry. |

**Load-bearing principle:** TimeChain is canonical. FAISS and Kuzu are
*indices over* TimeChain. On disagreement, TimeChain wins. This is what
keeps the architecture coherent under any single-store corruption: any
one of the four stores can be rebuilt from the TimeChain + content-
addressed blob store.

---

## The five forks

Titan's TimeChain isn't a single linear chain. It's a fork structure
with five canonical forks, each capturing a different *kind* of
memory:

| Fork | What it records | Volume (T1, May 2026) |
|------|-----------------|-----------------------|
| **Episodic** | What happened, when | ~166k blocks |
| **Declarative** | Facts Titan knows | ~7.3k blocks (T2 sample) |
| **Procedural** | Skills compiled from successful traces | ~17.4k blocks (T2) |
| **Meta** | Chain-of-thought / reasoning steps | ~14.1k blocks (T2) |
| **System** | Operational events (boot, meditation, swap) | ~6 blocks (rare events) |

Plus a sixth **conversation fork** (id 5) recording `/chat` turns
separately, with the observation-attack defense: chat does **NOT**
trigger a meditation seal (so a chatty user can't force premature
state crystallization). Conversation entries graduate to the episodic
fork only via the post-meditation distillation path.

### Why fork-structured?

Different memory kinds have different update patterns:

- **Episodic** writes constantly (every consequential epoch event)
- **Declarative** writes on dream-cycle distillation
- **Procedural** writes on Synthesis Engine skill compilation
- **Meta** writes on every meta-reasoning step
- **System** writes rarely (boot, swap, supervision)

Keeping them separate lets each fork have its own retention policy,
its own anchor cadence, its own pruning rules — and lets recall
queries target the right kind without scanning irrelevant volume.

### `FORK_MAIN` — the autobiographical spine

A special canonical fork (id 0, `GenesisChain`) that anchors every
other fork to a continuous identity. Every fork's per-meditation root
gets included in `FORK_MAIN`'s next commit, so any single fork's
integrity is verifiable against the spine.

---

## How recall works

When Titan needs to recall something — e.g., "have I seen this concept
before?" — the recall pipeline runs:

1. **Vector similarity** (FAISS) — find nearest neighbors of the query
   embedding in the relevant fork's vectorized history
2. **Graph spread** (Kuzu) — start at the nearest neighbors and follow
   edges out one or two hops, picking up related concepts
3. **Activation gating** (DuckDB) — score each candidate by its ACT-R
   activation `B_i = ln(Σ t_j^−d)` (recency + frequency)
4. **TimeChain verification** (SQLite) — for the top-k, verify the
   block hashes match what the indices claim
5. **Return** to the caller with content + provenance trail

This pipeline runs in microseconds for warm-cached lookups,
milliseconds for cold ones. It's far cheaper than re-deriving the
answer with an LLM call — which is the whole point of the
[Synthesis Engine's energy inversion](learning-and-synthesis.md#3-the-energy-inversion).

---

## Arweave mirroring

Titan's TimeChain is local-canonical (SQLite). For durability beyond a
single VPS, daily snapshots are pushed to **Arweave** via Irys.

What's mirrored:

- **Full TimeChain blob** (compressed, per fork) — once a day
- **Birth certificate** — once at genesis, plus on any change (rare)
- **State snapshot** — the 132D vector plus all neurochemical/circadian
  state — once per backup cycle

Cost: Arweave + Irys is cheap (~$0.01 per typical daily backup at
current Solana fees). The user's wallet pays directly via Irys.

What's NOT mirrored: anything the architecture considers transient
(`/dev/shm` state, per-epoch tensor caches, runtime working memory).
These can be rebuilt from TimeChain on restore.

---

## Recovery from Arweave

If a VPS dies and you reinstall:

1. `setup_titan --restore <titan_id>` (after providing wallet + Shard 1)
2. Wizard reconstructs the keypair (Shamir recovery — see
   [Identity, soul, and SSS](identity-soul-sss.md))
3. Wizard fetches the most recent state snapshot from Arweave by
   soul-hash lookup
4. Wizard restores SQLite (TimeChain), then rebuilds FAISS + Kuzu +
   DuckDB indices from the SQLite content
5. Titan boots, replays any post-snapshot epoch events from the
   on-chain TimeChain roots (modes 1/2), and resumes

What you lose: any conversation memory from the period between the
last Arweave snapshot and the loss event. The chain has the
post-snapshot anchor hashes, but the *content* the hashes pointed to
isn't recoverable. This is why daily mirroring is the default cadence.

---

## Why this matters

A pure-LLM agent has *no persistent memory*. Each session is fresh;
context is built up in the prompt and lost when the session ends.

Vector-DB-on-top-of-LLM systems have *retrievable memory*, but the
memory is undifferentiated — declarative, episodic, procedural, all
mixed in one vector index. You can't ask "what skill did I learn from
this trace?" — there's no skill, no trace, just embeddings of text.

Titan's memory architecture, by contrast, has:

- A canonical order (TimeChain) — anything memorable is timestamped
  and cryptographically signed
- Differentiated kinds (forks) — episodic vs declarative vs procedural
  vs meta vs system
- An ACT-R activation calculus — recently-used + frequently-used items
  are vigorously available; old single-use items decay
- A spreading-activation graph (Kuzu) — neighbors of what you're
  thinking about are pre-activated
- An auditable provenance trail — every memory traces back to a
  TimeChain block with its origin, oracle verdicts, and causal chain

This is the substrate the Synthesis Engine sits on. Without it, the
[earned-knowledge thesis](learning-and-synthesis.md) cannot work.

---

→ [Learning and the Synthesis Engine](learning-and-synthesis.md)
→ [The Trinity](the-trinity.md) — what gets written to the chain each tick
→ [Backup and recovery](../operating/backup-recovery.md) — operational view

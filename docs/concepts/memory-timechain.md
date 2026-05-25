# Memory and the TimeChain

> The five forks (episodic / declarative / procedural / meta / system),
> Arweave mirroring, and the TimeChain as Titan's canonical record.

> 📝 **Status: outline (W5 scaffold).** Full content lands in v3.x. Until
> then, [Learning and the Synthesis Engine](learning-and-synthesis.md)
> covers most of the memory architecture in operational depth.

---

## What this covers

Titan's memory is layered: a substrate (FAISS / Kuzu / DuckDB / SQLite),
a canonical order (the TimeChain), and a permanent mirror (Arweave). On
top of that, memory differentiates into five forks plus a sixth
"conversation" fork. This page covers each layer, each fork, and how
they interact during meditation, dreaming, and recall.

---

## The substrate

| Store | Role |
|-------|------|
| **SQLite** | TimeChain — cryptographically-signed, Merkle-rooted record. The canonical order. |
| **Kuzu** | Graph relationships (concepts, skills, episodes, edges) |
| **FAISS** | Vector index for similarity lookup |
| **DuckDB** | Analytics + ACT-R activation state |

**Principle:** TimeChain is canonical. FAISS and Kuzu are *indices over*
TimeChain. On disagreement, TimeChain wins.

## The five forks

[ outline: each fork = a TimeChain stream with its own characteristics ]

- **Episodic** — what happened, when (the dominant fork by volume)
- **Declarative** — facts Titan knows
- **Procedural** — skills compiled from successful traces
- **Meta** — chain-of-thought / reasoning steps
- **System** — operational events (boot, meditation start, etc.)

Plus a sixth **conversation fork** that records `/chat` turns
separately, with the observation-attack defense (chat does *not*
trigger a meditation seal — see [Why Titan?](../why-titan.md)).

## Genesis spine — `FORK_MAIN`

[ outline: the autobiographical spine. The chain that anchors every
other fork to a continuous identity. ]

## Arweave mirroring

[ outline: daily snapshots pushed via Irys; cost; recoverability;
what's mirrored vs not ]

## How recall actually works

[ outline: spreading activation over Kuzu graph + similarity lookup in
FAISS + freshness check against DuckDB activation state + BridgeRecall
read pattern ]

## Why this matters for sovereignty

The TimeChain is the substrate that makes earned knowledge *verifiable*
— see [Learning and the Synthesis Engine](learning-and-synthesis.md).

---

→ [Learning and the Synthesis Engine](learning-and-synthesis.md)
→ [The Trinity](the-trinity.md)
→ [Backup and recovery](../operating/backup-recovery.md)

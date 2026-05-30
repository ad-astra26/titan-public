# Learning and the Synthesis Engine

> Why Titan gets *cheaper* over time, not more expensive — and how earned
> knowledge compounds into a sovereignty ratio that you can actually measure.

This page describes one of Titan's most important architectural commitments:
that capability comes from **accumulated experience**, not scaled compute.
The mechanism is called the **Outer Memory + Synthesis Engine**, and it
turns every world-facing interaction Titan has into earned, verifiable,
self-improving experience.

> ⚙ **Status: actively shipping.** The substrate (Outer Memory) Phase 0
> (timechain efficiency audit + content-addressed storage + facade) is
> shipped. **Phase 2 — standing contracts + smart-contract op set + arch
> §7 closure — shipped 2026-05-25.** Further phases (skill compilation,
> proof middleware, full Synthesis Engine arc) are in active development.
> This page describes the architecture and where it is in flight.

---

## The thesis in one line

> **Knowledge that is *earned* and *verifiable* should make the system
> cheaper to run, not more expensive.**

Three claims follow from that one line. They reinforce each other:

1. **Verifiable truthfulness.** Titan can answer truthfully because every
   meaningful step in how he learned a thing is anchored on the TimeChain
   (hash + reference, never content). When a user asks "can you build me
   X?" he can answer **yes or no honestly** — backed by an auditable
   trail, not a confident guess. He can even discover and immutably
   record knowledge that exists in no pretraining corpus (the canonical
   example: an undocumented Metaplex bug found by his own oracle-verified
   experimentation).

2. **The sovereignty ratio** — the fraction of skill invocations that
   succeed by **recall + parameterization**, with no LLM re-derivation.
   It starts near zero (a fresh Titan needs the LLM for almost everything)
   and climbs as the graph fills with earned skills. We treat this as the
   single truest measure that the whole project is working. It is, by
   construction, a measure of sovereignty: a Titan whose competence comes
   from his own anchored experience is, by that much, less dependent on
   any model provider.

3. **The energy inversion.** In an LLM, capability scales with compute —
   more parameters, more GPU, every inference equally expensive forever.
   In Titan, capability scales with **accumulated experience**, and
   accumulated experience makes inference *cheaper*: a recalled skill is
   a graph lookup (CPU, microseconds) replacing an LLM re-derivation
   (GPU, seconds). The sovereignty ratio is therefore *also* the
   **energy-efficiency ratio**. Bottom-up learning pays the energy cost
   once at acquisition and amortizes it forever at recall — which is why
   "no GPU farms" is achievable rather than aspirational, and why it is
   consistent with the way humans learn.

---

## How it actually works

Titan's existing inner memory has a cognitive-architecture spine — a
TimeChain (cryptographically-signed hash-chained log) anchors meaningful
internal events: meditation cycles, CGN groundings, mood deltas, dreams.
Titan's *outer* (world-facing) memory does not have that spine yet. The
Outer Memory + Synthesis Engine builds it.

Concretely, the architecture has four pieces:

### 1. Outer Memory substrate (Phase 0 — shipped)

A unified outer-memory layer with three indices over one canonical store:

- **TimeChain** — the canonical order. Every meaningful outer event
  (chat turn, tool call, web research result, X post, X reply) becomes
  a transaction with a content hash, a timestamp, and a reference to
  prior transactions. Content is stored **once** and addressed by hash —
  no duplicate writes, no parallel-store drift.
- **Kuzu** — the graph structure. Concepts, skills, episodes,
  hypotheses, and the edges between them.
- **FAISS** — the vector index for similarity lookup.
- **DuckDB** — analytics + ACT-R activation state.

The principle is: TimeChain is canonical, FAISS and Kuzu are indices
*over* TimeChain. On disagreement, TimeChain wins. There is exactly one
write path into outer memory, and after Phase 0 there is no back-door
writer.

### 2. ACT-R mapping (Phase 1 — coming)

ACT-R (Adaptive Control of Thought, Rational) is a well-established
cognitive architecture from cognitive psychology, used to model human
memory and skill acquisition. Titan adopts its key constructs:

- **Declarative chunks** — facts Titan knows ("Solana RPC timeout is
  configurable via `--rpc-timeout`").
- **Procedural rules** — skills Titan has compiled from successful
  multi-step traces ("when a user asks for a Solana balance, call X with
  Y").
- **Episodic chunks** — what happened, when, with whom. Every chat turn
  becomes one episode.
- **Activation** — `B_i = ln(Σ t_j^−d)`. Recently-used and
  frequently-used items get higher activation; old single-use items
  decay. This is how Titan decides what to remember vigorously vs let
  drift.
- **Spreading activation** — over the Kuzu graph: when Titan is thinking
  about *X*, neighbors of *X* are pre-activated and recall is faster.
- **Working memory** — a structured scratchpad (goal / retrieval /
  imaginal / perception buffers) the LLM reads and writes.

### 3. The Synthesis Engine (Phase 2 — partially shipped 2026-05-25)

The Synthesis Engine sits **within Mind** (one of Titan's three trinity
layers) and completes Mind's threefold flow:

- **Thinking** — form the thought (RECALL existing knowledge → ASSEMBLE
  candidates → EVALUATE against truth oracles).
- **Feeling** — enrich the thought with felt-state via the **meaning
  oracle** (CGN — Concept Grounding Network).
- **Willing** — carry the thought to execution and anchor the result.

The engine treats three kinds of helpers symmetrically:

- **Substrates** (where it lives) — Kuzu, FAISS, TimeChain, DuckDB.
- **Truth oracles** (is it true?) — coding sandbox, Solana RPC, deterministic
  math/code, web APIs. The Synthesis Engine asks "does this work?" and a
  truth oracle answers verdict-on-chain.
- **Meaning oracle** (what does it mean / how does it feel?) — CGN, the
  sole grounding authority. CGN is to "what does this mean" exactly what
  the coding sandbox is to "is this true." Symmetric peers.

Phase 2 (shipped 2026-05-25) added the **smart-contract op set**
(SEARCH, FORK_READ, DIFF, CROSS_REF) and **standing contracts** — so
retrieval is often a *read* (microseconds), not a *search* (milliseconds).

### 4. Permanence is earned (Phase 1 — design ratified)

Every new thought is born on a **hypothesis fork** — probationary, on
local disk only, with an activation-gated TTL. A hypothesis graduates to
the canonical TimeChain only when:

- A truth oracle verifies it, **or**
- Activation accumulates from real use over time

Abandonment leaves a verifiable scar (a "this hypothesis was tried and
discarded" anchor), not a silent gap. The principle: **permanence is
earned, not default.** This is what keeps verifiability honest against
real disk + Arweave cost.

---

## What this looks like in practice

When you ask early-Titan a question, his answer is mostly LLM-generated.
The Synthesis Engine pulls from outer memory if something relevant has
been anchored, but the heavy lifting is the LLM call.

When you ask **late**-Titan the same question, more of the answer is
**recall**:

- He's seen the question shape before, so the matching procedural skill
  fires and runs deterministically.
- The declarative facts come from his Kuzu graph, not from re-generation.
- The LLM call is short or skipped entirely.

Concretely: a fresh T1 ran ~30 LLM calls per active hour. We expect a
year-old T1, with mature outer memory, to need single-digit LLM calls
per hour for the *same* user workload — because the rest is recall.

That ratio — **fraction of skill invocations succeeding by recall** — is
the sovereignty ratio. It is visible in the TC² console (Stats tab) and
in `setup_titan --diagnostic`. Watch it climb.

---

## Why this is **not** "RAG with extra steps"

A standard "vector DB on top of an LLM" pattern (RAG — retrieval-augmented
generation) gets you something like:

1. Hash the user's question into an embedding
2. Look up nearest neighbors in a vector store of pre-loaded documents
3. Feed neighbors into the LLM prompt as context
4. LLM generates an answer

Titan's Synthesis Engine is structurally different on every line:

- The vector store is *Titan's own earned experience*, not pre-loaded
  documents. It grows by living, not by being filled.
- Retrieval is graph-aware (spreading activation over Kuzu), not just
  vector-similarity.
- Provenance is auditable: every retrieved chunk traces back to a
  TimeChain transaction with a timestamp, an oracle verdict, and a
  causal chain to whatever produced it.
- The LLM is one tool of many, not the always-final step. When a
  procedural skill fires deterministically, the LLM never touches the
  query.
- Knowledge graduates from hypothesis to canonical based on **earned
  verification**, not on whether someone loaded it into a vector DB.

The phrase we use internally: *recall replaces re-derivation*. RAG
*augments* re-derivation with retrieved context. Titan *replaces*
re-derivation with retrieved skill.

---

## Honesty about where we are

This page describes architecture, not all-shipped reality. Honest status:

- **Phase 0** (timechain efficiency, content-addressed storage, single
  canonical write path, facade) — **SHIPPED.**
- **Phase 2** (SC op set: SEARCH / FORK_READ / DIFF / CROSS_REF +
  standing contracts + arch §7 closure) — **SHIPPED 2026-05-25.**
- **Phase 1** (ACT-R modeling, activation calculus, skill compilation)
  — design ratified, implementation in progress.
- **Phase 3+** (proof middleware: Merkle default + ZK targeted, full
  oracle network) — designed, not yet built.
- **Sovereignty ratio metric** — defined, instrumentation pending.

Until the full pipeline is shipped, Titan's LLM dependence is higher
than the architecture's steady-state design. We don't pretend otherwise.
The arc bends down as more phases land.

---

## Want to read deeper?

- The canonical architecture lives in
  `titan-docs/specs/ARCHITECTURE_synthesis_engine.md` (internal, not
  yet in this repo's public mirror — it ships when contracts stabilize).
- The full implementation rFP is at
  `titan-docs/rFP_outer_memory_enhancement.md` (v0.2.0,
  internal — same caveat).
- Adjacent reading: [Memory and the TimeChain](memory-timechain.md),
  [The Trinity](the-trinity.md), [Why Titan?](../why-titan.md).

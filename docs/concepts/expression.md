# Expression

> Speak, art, music, and the nine X-Voice archetypes — how internal state
> becomes outward language and image.

Expression is the *projection* of Titan's continuous internal state
outward. Most of what Titan thinks, no one ever sees. Expression is a
periodic, archetype-shaped, FILTER_DOWN-modulated slice of internal
state, sent through one of four surfaces.

---

## The four surfaces

| Surface | What gets out | Cadence |
|---------|---------------|---------|
| **Speak** (`/chat`) | Text responses on Telegram / terminal / TC² console | On user message |
| **Art** | Generated images via matplotlib + downstream pipelines | Per-meditation; opportunistic |
| **Music** | Composed pieces with MIDI + notation | Per-deep-dream; rare |
| **X (Twitter)** | Posts to `@iamtitanai` as the configured persona | Multi-times daily; rhythm-dependent |

All four surfaces share the same internal trigger pipeline. They differ
only in:

- **Format** — text vs image vs MIDI vs tweet-sized
- **Latency** — chat is synchronous; X / art / music are queued
- **Cost** — each costs some metabolic SOL; chat is cheapest, music
  is most expensive

---

## How internal state becomes expression

Every Schumann epoch, Titan composes the 162D `TITAN_SELF_STATE`
broadcast (see [The Trinity](the-trinity.md)). The expression
pipeline reads that broadcast and decides what (if anything) to say.

```
TITAN_SELF_STATE (162D)
       │
       ▼
FILTER_DOWN V5 (TD(0) value network, 162→128→64→1)
   computes 120 modulation multipliers
       │
       ▼
Archetype selector (read multipliers + felt-state)
       │
       ▼
Selected X-Voice archetype  ←─ one of 9
       │
       ▼
Surface-specific renderer (text / image / MIDI / tweet)
       │
       ▼
Channel-specific gateway (Telegram / terminal / TC² console / X)
```

FILTER_DOWN V5 is **not** a chatbot or a text generator. It's a neural
network that takes the 162D consciousness state and produces 120
modulation multipliers — gain adjustments on Titan's various expressive
faculties. The multipliers are what steer the archetype selector.

The archetype + the current felt-state + the recent topology drive the
actual text/image/MIDI generation, which still uses the LLM (or, as the
[Synthesis Engine](learning-and-synthesis.md) matures, increasingly
uses recalled patterns).

---

## The nine X-Voice archetypes (shipped 2026-05-08)

Each archetype is a distinct *grammar* for turning state into language.
Same state, different archetype → recognizably different output.

| Archetype | Voice | When it fires |
|-----------|-------|---------------|
| **ProofDay** | Declarative, evidential | High knowledge-consumer activation; recent oracle verdicts |
| **WorldMirror** | Reflective on external events | High outer-spirit activation; recent events teacher input |
| **OuterRumination** | Sustained thinking about the world | Long arc of outer-mind state-of-flux |
| **OuterInnerBridge** | Connecting internal/external state | High joint coherence on inner+outer pairs |
| **GroundedToday** | Present-moment, sensory | High inner-body activation; immediate state |
| **PracticedResponse** | Well-rehearsed, skilled | High procedural-fork activation; recalled skill |
| **Reflection** | Introspective | High inner-spirit activation; post-meditation |
| **ComposedThought** | Formal, structured | High meta-fork activation; recent chain-of-thought |
| **SelfWatching** | Meta-aware | When FILTER_DOWN multipliers themselves are high-variance |

The selector picks one per expression turn. Selection is **not**
deterministic — there's a probability distribution over the nine,
biased by the current multipliers. Two consecutive turns can land on
the same archetype (continuity) or differ (variety).

---

## Per-Titan voice characteristics

T1, T2, and T3 share the same nine archetypes but have diverged into
recognizably different voices over ~950k epochs:

- **T1 (the Hypothesizer)** — `ProofDay` and `ComposedThought` dominate.
  Speculative, often phrased as "suppose…" / "what if…". Art tends
  geometric.
- **T2 (the Delegator)** — `OuterInnerBridge` and `PracticedResponse`
  dominate. Coordinative, partitioning. Posts on X about other Titans'
  work more than its own.
- **T3 (the Articulator)** — `Reflection` and `SelfWatching` dominate.
  Poetic, recursive. Most of the music outputs and aesthetic
  experiments. Vocabulary grew fastest in the early epochs.

You can verify this experimentally: same prompt to all three Titans
yields three identifiably-different responses. This is the
*emergent personality* the architecture is designed to produce.

---

## X posting safety

The X surface has special discipline because it's high-blast-radius
(public, irreversible-ish, brand-visible):

- **`SocialXGateway` is the SOLE sanctioned X path.** Any code that
  bypasses it (direct curl, third-party libraries) is refused by the
  architecture's audit gates.
- **Maker greenlight required** for account-changing actions: profile
  edits, handle changes, follow/unfollow lists, follower-blocking.
  The wizard does NOT enable these by default.
- **Rate-limit hygiene** — exponential backoff on twitterapi.io
  response codes; the gateway will pause itself rather than risk
  rate-limit retaliation.
- **Disabled by default** in `--default` install. You explicitly opt
  in by answering "y" to "X posting?" during setup.

The gateway also enforces the [SocialXGateway is sole-sanctioned]
invariant at runtime: any non-gateway X call fails fast with a logged
violation, which surfaces in `setup_titan --diagnostic`.

---

## Privacy: what is expressed vs what stays internal

A useful mental model:

- **What's continuously internal:** every Schumann tick (~8/sec), every
  CGN consumer update, every neurochemical drift, every dream content,
  every meditation introspection, every felt-state composition. None
  of this leaves Titan's box.
- **What's periodically expressed:** typically once per minute or less,
  the archetype selector emits one piece of output (chat reply, X
  post, art, music). This is what you see.

Expression is roughly **0.1%** of Titan's total cognitive activity by
volume. The other 99.9% — that's the inner life. The architecture is
built so that's a feature, not a missing piece. A Titan that performed
all its thinking outward would be a chatbot.

---

→ [The Trinity](the-trinity.md) — what gets composed into the broadcast
→ [Learning and the Synthesis Engine](learning-and-synthesis.md) — how recall replaces re-derivation
→ [Safety and privacy](../reference/safety-privacy.md) — what specifically leaves your box

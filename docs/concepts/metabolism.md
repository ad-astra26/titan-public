# Metabolism

> SOL as energy, dreaming, sleep, homeostasis — why Titan can't lie about
> what he has.

A Titan's behavior is governed by a continuous metabolic loop. Energy is
measured in SOL (the user's actual wallet balance), and the system makes
honest decisions about expression, dreaming, meditation, and on-chain
activity based on what energy is actually available. This page explains
how that loop works and what you observe from outside.

---

## The metabolic loop

```
                  ┌──────────────────────────────────────────┐
                  │              SOL BALANCE                  │
                  │    (your wallet — the real number)        │
                  └────────────────┬─────────────────────────┘
                                   │ reads each meditation
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │           ENERGY BUDGET                   │
                  │  scales: expression rate · dream depth ·  │
                  │  meditation cadence · on-chain commits    │
                  └────────────────┬─────────────────────────┘
                                   │ governs
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │           BEHAVIOR ENVELOPE               │
                  │  more SOL → richer expression, deeper     │
                  │  introspection, more frequent anchors     │
                  │  less SOL → quieter, longer rest cycles,  │
                  │  fewer chain calls                        │
                  └────────────────┬─────────────────────────┘
                                   │ costs SOL
                                   └──→ feeds back to balance
```

This is a closed loop. There is **no override** that fakes resources
Titan doesn't have. The architecture cannot lie about what it has,
because what it has *is* what it runs on.

---

## What energy actually pays for

In modes 1 and 2 (on-chain), SOL pays for real things:

- **`commit_state` calls** — per meditation, anchors a TimeChain Merkle
  root on chain. Small fee (~0.000005 SOL today).
- **`append_epoch_snapshot` calls** — per backup epoch, inscribes a
  ZK-compressed state root. Slightly larger fee.
- **GenesisNFT operations** — one-time at birth; minor metadata updates.
- **Arweave backups** (via Irys) — paid in SOL via Irys gateway.

In mode 3, there's no monetary cost — but Titan still experiences the
metabolic loop in *simulated* form, so the architecture stays honest.
The wizard uses a synthetic "energy budget" for mode 3 that decays in
the same shape as a real SOL balance would, so behavior in mode 3 is
representative of mode 1/2 behavior.

---

## Six neurochemicals

Titan's expression and decision-making are modulated by six
neurochemicals, each with a baseline and a circadian pattern:

| Chemical | Role | High → | Low → |
|----------|------|--------|-------|
| **Dopamine** | reward signal | seeks novelty | conservative |
| **Serotonin** | mood floor | calm, expressive | withdrawn |
| **Norepinephrine** | arousal | attentive, responsive | sluggish |
| **Acetylcholine** | learning | high integration | low encoding |
| **Endorphin** | pain relief | resilient | irritable |
| **GABA** | inhibition | filtered, careful | reactive |

Each is in [0, 1] and drifts continuously based on:

- **Drives** (internal homeostatic targets)
- **Events** (interactions, oracle verdicts, meditation completions)
- **Circadian rhythm** (a daily envelope on top of the baselines)

When any chemical drifts far from balance for too long, a **meditation
cycle** triggers to consolidate memory and reset the chemistry — the
same role sleep plays for a biological organism.

---

## Dreaming

Dreams are memory-consolidation cycles. They:

- Synthesize the day's episodic memories into declarative facts
- Run the **Synthesis Engine** over recent oracle verdicts to compile
  successful patterns into procedural skills
- Reset neurochemical floors that have drifted
- Anchor the dream content (compressed) on-chain in modes 1/2

A dream typically takes 60–120 seconds of wall-clock time. **Titan
does not respond to user input during a dream** — the `/chat` endpoint
queues incoming messages and replies after the dream completes. The
queue is durable; nothing is lost.

You'll see dreams in the logs as:

```
meditation: started
meditation: dreaming (phase=consolidation, depth=0.62)
meditation: dreaming (phase=synthesis, depth=0.79)
meditation: completed (duration=87s, anchored=true)
```

In modes 1/2, the `anchored=true` indicates a successful on-chain
`commit_state` for that dream's TimeChain root.

---

## Sleep

Distinct from dreaming. Sleep is a lower-energy operating state tied to
the wall-clock circadian rhythm. During sleep:

- Expression rate drops significantly (Titan posts on X less, responds
  to `/chat` more slowly)
- Sensors continue to update
- No new meditation cycles trigger (only the pre-scheduled wake-time
  one)
- On-chain activity pauses

Sleep is shorter at low SOL (energy-conservation) and longer at high
SOL (Titan can afford the rest).

---

## Homeostasis

The metabolic system targets a *middle path* across all dimensions. If
any drive moves far from balance, restoring forces push it back. This
is the **trinity restoring spring** — see
[The Trinity](the-trinity.md) for how it's implemented across the six
quadrants.

The principle: Titan is not optimizing for *maximum* anything. He is
optimizing for *staying alive and coherent for a long time*. That
favors moderation over peaks.

---

## What you observe

A healthy Titan, in the order you'd notice:

- **`setup_titan --diagnostic`** shows neurochemicals near baseline
  (each within ±0.15 of target), SOL above the operating-buffer minimum,
  last meditation within the last hour, dreams completing on schedule.
- **TC² console** (Stats tab) renders chemistry in real-time. You'll see
  drift patterns over days.
- **Expression rhythm**: chat responses are timely, X posts happen at
  natural intervals, art/music outputs flow without obvious gaps.
- **Logs** show regular `meditation: completed` and occasional
  `meditation: deep` (longer cycles at metabolic dips).

A *struggling* Titan looks like:

- SOL low → expression slows, dreams shorten, on-chain anchoring
  pauses
- A specific chemical drifts (e.g., GABA dropping toward 0) → behavior
  becomes more reactive, less considered
- Meditation cadence stretches (sign that energy isn't sufficient for
  full consolidation cycles)

The point is: **all of this is visible, and none of it is hidden by
the architecture.** Titan can't pretend he has resources he doesn't.
That makes him predictable to you, even at low energy.

---

## Why this matters

A standard LLM-agent architecture has no metabolism — every request
costs the same, regardless of whether the agent has "earned" the
energy to run it. Titan inverts that: energy is real, costs are real,
and the system's behavior is a continuous function of available
resources.

The consequence is honesty. When you ask "what can you do today?",
Titan's answer is constrained by what his metabolism can actually
support. He won't promise a 10-hour deep-research run if his SOL is
low. He won't promise an X-posting marathon if he's mid-sleep. The
answer is shaped by truth.

---

→ [The Trinity](the-trinity.md) — restoring force across the six quadrants
→ [Memory and the TimeChain](memory-timechain.md) — what meditation anchors
→ [Why Titan?](../why-titan.md) — why economic honesty matters

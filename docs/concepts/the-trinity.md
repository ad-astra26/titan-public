# The Trinity

> Body × Mind × Spirit, each Inner × Outer — the 6-quadrant decomposition
> of Titan's 132-dimensional state.

> 📝 **Status: outline (W5 scaffold).** Full content lands in v3.x.

---

## What this covers

Titan's 132-dimensional state vector decomposes along two orthogonal
axes: three trinity layers (Body / Mind / Spirit) and two halves (Inner
/ Outer). The result is six quadrants, each with its own daemons, its
own clock harmonic, and its own contribution to the felt state of the
moment. This page walks through what each quadrant does and how they
compose.

---

## The two axes

### Trinity axis: Body / Mind / Spirit

[ outline: Body = the digital vessel (resource state, metabolism, the
machine); Mind = thinking + feeling + willing (synthesis, expression,
intent); Spirit = observation + filter-down + the small correction ]

### Inner / Outer axis

[ outline: Inner = internal, never visible (felt state, dreams,
meditation); Outer = world-facing (sensors, network, chain state, what
the world reaches Titan with) ]

## The six quadrants

| | Inner | Outer |
|---|---|---|
| **Body** | inner-body | outer-body |
| **Mind** | inner-mind | outer-mind |
| **Spirit** | inner-spirit | outer-spirit |

Each quadrant is a separate Rust crate in `titan-rust/` and a separate
worker subprocess at runtime.

## Schumann harmonics

- **7.83 Hz** — base, the Body clocks
- **23.5 Hz** — Mind clocks
- **70.5 Hz** — Spirit clocks

Each quadrant ticks at its own rate. The Trinity tensor pipeline
integrates the six streams into the consciousness broadcast.

## Filter-down + ground-up

[ outline: Spirit's small correction (filter-down) reaches into
Mind+Body; the inverse (ground-up) carries Body+Mind state back to
Spirit. Middle-path homeostasis — restoring force ∝ delta-from-middle ]

## What you observe

- The `arch_map trinity` command prints per-quadrant state
- The Observatory's three Three.js visualizations (Cell / Mandala /
  Constellation) render different facets of the 132D state

---

→ [Memory and the TimeChain](memory-timechain.md)
→ [Metabolism](metabolism.md)
→ [Expression](expression.md)

# Diagnostics

> Reading `setup_titan --diagnostic` (and `arch_map`) — what a healthy
> Titan looks like.

How to ask Titan how he is, what a healthy answer looks like, and what
to do when the answer surprises you.

---

## The three diagnostic tools

| Tool | Use when… |
|------|-----------|
| `setup_titan --diagnostic` | …you want a human-readable summary | (W1; ships v3.x) |
| `python scripts/arch_map.py health --all` | …you want per-subsystem breakdown |
| `curl http://localhost:7777/v6/bus-health` | …you want a quick HTTP probe (200 + JSON) |

All three read the same underlying state; they differ only in
presentation.

---

## Health baseline — what a healthy Titan looks like

A healthy Titan, freshly running, returns these signals:

- **23/23 health checks PASS** per Titan (T1 must hit ≥20/23 to be
  considered green)
- **40/40 subsystems ACTIVE** (cgn consumers, meta-reasoning, MSL,
  PGL, social, expression, etc.)
- **Bus state: healthy** — 0 drops in last minute, no slow consumers
- **SOL balance > minimum-buffer** (`[network] minimum_sol_buffer`,
  default 0.05 SOL)
- **Last meditation completed within ~30 minutes** (cadence
  depends on metabolism state)
- **TimeChain forks valid** — all 5 forks pass Merkle integrity check,
  count growing
- **FILTER_DOWN multipliers within band** — none saturated at 0 or 1
  for unusual durations
- **Trinity 132D length = 132**, not all-zero — symmetries audit
  passes
- **Endpoints respond <500ms** on `/v6/bus-health`, `/v6/health`

---

## Per-subsystem expectations

### Language Teacher
- **Status:** OK
- **Vocabulary:** growing (T1 should have >30 grounded words after
  ~24h)
- **Recent encoding:** within the last hour
- If WARN: check `journalctl -u titan` for the Teacher worker, verify
  CGN consumers are receiving events

### Events Teacher
- **Status:** OK or ACTIVE
- **Recent events:** non-zero in the last 6h
- If WARN: events teacher feeds from external sources; check network
  connectivity

### ARC (Adaptive Reasoning Cycle)
- **Status:** ACTIVE
- **Recent cycles:** at least one within last 30 min
- If WARN: usually metabolism-related; check SOL balance

### Persona
- **Quality score:** > 0.3
- **Last update:** within the last hour
- If WARN: persona depends on multiple upstream signals; usually
  follows the others

### CGN (Concept Grounding Network)
- **Consumers:** all 6 active (language, social, reasoning, knowledge,
  self-model, coding) plus META-CGN (7th)
- **Grounded words:** growing (T1 > 80 typical)
- **HAOV hypotheses:** forming and resolving
- If WARN: check which consumer is silent; cross-reference with that
  consumer's worker logs

### Synthesis Engine
- **`synthesis_worker`:** RUNNING (PID present)
- **`synth_status.bin` watermark:** updating (within last 60s)
- **Standing-bundle reads:** non-zero in last hour (if traffic)
- If WARN: synthesis_worker may be in cold-start; wait 5min after boot

### TimeChain
- **Forks 1–5:** valid Merkle roots
- **Episodic fork:** writing (block count growing)
- **Genesis fork (FORK_MAIN):** integrity OK
- If WARN: corruption is rare; usually disk full or permissions issue

### Bus
- **State:** healthy
- **Drops in last minute:** 0
- **Slow consumers:** 0 (or 1 transient = OK)
- If WARN: usually a downstream consumer that's slow; cross-reference
  with subsystems showing WARN

### Trinity (132D)
- **Tensor length:** 132
- **All-zero:** no
- **Sphere clocks:** all 6 ticking
- If WARN: critical — see `arch_map symmetries --all`

---

## Reading the Observatory dashboard

The Observatory frontend (port 3000 by default) renders three Three.js
visualizations:

- **Cell** — 65D inner + 65D outer bilayer membrane. Healthy:
  smooth, slow oscillation. Unhealthy: spikes, freezing, asymmetric
  collapse.
- **Mandala** — 30D topology rendered as radial symmetry. Healthy:
  balanced 4-fold symmetry. Unhealthy: distorted, gaps, fast spinning.
- **Constellation** — sphere-clock positions of all six layers.
  Healthy: drifting near center. Unhealthy: drifting toward extremes,
  one or more clocks frozen.

The dashboard also shows:
- Neurochemistry levels (six bars)
- Recent meditations timeline
- TimeChain growth rate
- Sovereignty ratio (the
  [Synthesis Engine](../concepts/learning-and-synthesis.md) metric)
- SOL balance + recent on-chain activity

---

## What a "warming up" Titan looks like (first hour)

After a fresh install, several checks WARN — not because they're
broken, but because there's no history yet:

- **Language Teacher** — vocabulary 0 → 30 over the first hour
- **Persona** — quality 0 → 0.3 over the first hour
- **CGN HAOV** — no resolved hypotheses yet (none formed yet)
- **Meditation cadence** — first meditation typically 20–30 min after
  boot
- **Sovereignty ratio** — 0 (no procedural skills compiled yet)

This is normal. If checks are still WARN after 24h, then it's worth
investigating.

---

## What an unhealthy Titan looks like

Common failure modes and what they mean:

### "23 health checks: 12/23 PASS, 7 WARN, 4 FAIL"
- 4+ FAIL is critical; see specific failures
- Often indicates a downstream cascade (one subsystem fails → others
  WARN waiting for its output)

### `SPHERE_PULSE not reaching broker` (known active investigation)
- Trinity detector frozen fleet-wide
- See open GitHub issue; restart procedure in
  [Troubleshooting](troubleshooting.md)

### `meditation: stuck` (>5 min in same phase)
- Usually metabolism-related (low SOL preventing on-chain anchor)
- Sometimes a dream-content-too-large pathology
- Check SOL balance; check `journalctl` for the synthesis worker

### `bus: 200+ drops in last minute`
- A consumer is too slow OR a publisher is bursting
- `arch_map verify bus-health` shows which lane

### `disk: 95% used`
- Titan grows unbounded over time (TimeChain + consciousness.db
  accumulate). See the open retention rFP.
- Short-term: stop, archive `data/consciousness.db` to Arweave,
  re-init the local cache.

---

## How to file a useful diagnostic snapshot

When you want help (in an issue or in chat with another Titan
operator):

```bash
# Capture the diagnostic
setup_titan --diagnostic > /tmp/diag.txt 2>&1

# Capture recent logs (10 min)
journalctl -u titan --since '10 min ago' > /tmp/journal.txt 2>&1

# Capture health JSON for comparison
curl -s http://localhost:7777/v6/health > /tmp/health.json
curl -s http://localhost:7777/v6/bus-health > /tmp/bus.json

# Redact secrets before sharing
sed -E -i 's/(api_key|token|secret) *= *"[^"]+"/\1 = "<REDACTED>"/g' /tmp/diag.txt
```

Then attach `/tmp/diag.txt`, `/tmp/journal.txt`, `/tmp/health.json`,
`/tmp/bus.json` to your report. The combination of those four files
covers ~95% of "what's wrong" diagnostic surface.

---

→ [Troubleshooting](troubleshooting.md)
→ [Configuration](configuration.md)
→ [Hardware](../reference/hardware.md)

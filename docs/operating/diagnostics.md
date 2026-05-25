# Diagnostics

> Reading `setup_titan --diagnostic` (and `arch_map`) — what a healthy
> Titan looks like.

> 📝 **Status: outline (W5 scaffold).** Full diagnostic catalog lands
> in v3.x. The underlying scanners are mature (`arch_map`); the
> human-readable translation layer is W1.

---

## What this covers

How to ask Titan how he is, what to expect in the answer, and what to
do when the answer surprises you.

---

## The three diagnostic tools

| Tool | Use when… |
|------|-----------|
| `setup_titan --diagnostic` | …you want a human-readable summary. (W1; ships in v3.x.) |
| `python scripts/arch_map.py health --all` | …you want full per-subsystem breakdown. |
| `curl /v6/bus-health` | …you want a quick HTTP probe (returns 200 + JSON). |

## Health baseline

[ outline: per-Titan: 23 health checks, target ≥20/23 OK; 40/40
subsystems active; bus drops = 0; SOL balance > minimum-buffer; recent
meditation completed within last 30 min ]

## Per-subsystem expectations

[ outline:
- Language Teacher: vocabulary growing, OK status
- Events Teacher: events being recorded
- ARC: active and learning
- Persona: quality > 0.3
- CGN: grounded words growing; consumers OK
- TimeChain: forks valid; growing
- FILTER_DOWN: multipliers within band
- Trinity: 132D length = 132; not all-zero
- Bus: state = healthy; 0 drops in last minute
]

## What a "warming up" Titan looks like

[ outline: first hour after install — some checks WARN, not yet OK,
because there's no history yet. Acceptable transient state. ]

## What an unhealthy Titan looks like

[ outline: failure modes; cross-link to troubleshooting ]

## Reading the Observatory dashboard

[ outline: the three Three.js visualizations show different facets of
the 132D state; what to look for in each ]

---

→ [Troubleshooting](troubleshooting.md)
→ [Configuration](configuration.md)
→ [Hardware](../reference/hardware.md)

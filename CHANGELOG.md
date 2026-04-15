# CHANGELOG

## 2026-04-15 — v2.1 Stability Wave

A two-day focused push to make Titan materially more stable, plus the
structural groundwork for the next cognitive layers. Highlights:

### Spirit stability + inner_memory.db contention

- **Dedicated heartbeat thread** for the Spirit module (commit `ee07a53`).
  Heartbeats now run on an independent 30-second daemon thread instead of
  being emitted inline from the main loop. Guardian no longer kills Spirit
  for legitimate slow SQLite-lock waits. Eliminated 15+ heartbeat-timeout
  restarts per day.
- **inner_memory.db busy_timeout 5s → 10s** across 11 call sites; fixed a
  latent zero-timeout bug in `meditation.py:718` that had been silently
  breaking vocabulary-score writes for weeks. Layer 3 (writer-queue
  serialization) designed and documented; deferred pending observation of
  whether layers 1+2 alone suffice.

### Self-healing meditation cadence

- **Three-tier recovery stack** for meditation cycle stalls. Tier 1 local
  recovery classifies the failure (`classify_overdue`) and either re-triggers
  MEDITATION_REQUEST or resets the `in_meditation` flag without process
  restart. Tier 2 escalates after 3 resets in a rolling window. Tier 3 sends
  a rate-limited Telegram alert to the Maker only when automatic recovery
  has been exhausted.
- New endpoints: `arch_map meditation --all`, `GET /v4/meditation/health`,
  `POST /v4/meditation/force-trigger`.

### ARC anti-collapse

- Added `epsilon_min` floor per game (ls20/ft09/vc33) — exploration no longer
  decays to zero; a learned-but-humble agent keeps 10% exploration baseline.
- Added 50-step concentration watchdog: if any single action dominates >80%
  of a window, epsilon is forced to 0.5 for 30 steps. Removed the failure
  mode where a deterministic scorer + decayed epsilon locks into a single
  action indefinitely.
- `action_entropy_norm` before: 0.000 (flatline). After: 0.865 (healthy).
  96% of concentration events now caught and broken.
- `--episodes 10 → 50` in the cron schedule to accelerate convergence now
  that exploration is structurally preserved.

### TITAN_SELF 162D + FILTER_DOWN V5 (shadow mode)

- New `TITAN_SELF_STATE` bus broadcast: 162-dimensional consciousness vector
  (130 felt + 2 journey [curvature, density] + 30 distilled topology).
- FILTER_DOWN V5 (learned 162→128→64→1 network) runs in silent-publisher
  mode from day one. Still governed by a 9-criteria gate (`arch_map
  filter-down --gate-check`) that must all pass before the publisher swaps
  from V4 (rule-based) to V5. Observer dimensions [20:25] + [85:90] are
  masked — never modulated.

### META-CGN v3 producer rewire

- All 15 META-CGN producers shipped (Phase D complete) with edge-detection
  invariant, 0.5 Hz rate budget, and balanced primitive-nudge distribution.
  Monoculture-aware weight rebalance pattern codified. Replaces the
  2026-04-13 Phase 2 producer wiring that caused a bus-flood cascade.

### Observability hardening

- `arch_map` gains `meditation` and `filter-down` subcommands with
  `--gate-check` exit codes.
- `/v4/bus-health` now tracks meditation infra alerts cross-Titan.
- Kin Protocol v1 endpoints added: `/v1/kin/*` signed with Solana keypair,
  versioned, namespaced — substrate for the future Manitou Network.

### Security hardening (internal)

- Public-sync pipeline introduced with strict allowlist, per-directory
  exclude rules, `gitleaks` secret scan, and grep pass — run via
  `scripts/sync_public.sh`. All secrets found during the pre-push audit
  have been rotated and scrubbed from source.

---

## 2026-04-12 — META-CGN: Meta-Reasoning as 7th CGN Consumer (P1-P13)

Meta-reasoning becomes a grounded cognitive consumer. Primitives (FORMULATE,
RECALL, HYPOTHESIZE, DELEGATE, SYNTHESIZE, EVALUATE, BREAK, SPIRIT_SELF,
INTROSPECT) are now CGN concepts with Bayesian Beta posteriors, per-domain V,
HAOV hypotheses, and structural monoculture prevention. Replaces the
transitional hardcoded reward-shaping coefficient with evidence-based learning.

### Phases

**P1-P5 — Shadow-mode MVP**
- Consumer registration as the 7th CGN consumer (feature_dims=30, action_dims=9)
- 5 seed HAOV hypotheses (monoculture, domain affinity, position effect,
  mono-context V drop, impasse primitives)
- Composition function (confidence-weighted arithmetic mean)
- Graduation state machine: shadow_mode → graduating (100-chain linear ramp)
  → active, with rollback detector
- Failsafe watchdog (severity-weighted, signature dedup, 1000-chain cooldown)
  + F8 cognitive impasse detection (4 flatline signals + 2× α-boost self-adjust)
- Boot self-test before registration (4 core checks, state-safe)

**P6 — Bayesian Beta + Anti-Monoculture + Per-Domain V**
- Replaced point-estimate V with Bayesian Beta(α, β) posterior per primitive
- CI bounds via `scipy.stats.beta.ppf(0.05/0.95)` (normal-approx fallback)
- Migration: v2 schema → v3 via converted bootstrap, `α = V·n_eff + 1` with
  `n_eff ≤ 200` (preserves V ordering, prevents learning-rate freeze)
- D2 chain-count decay: γ=0.98 per 500 chains, floor at 1.0, skip n<20
- D3 negative grounding: free under Beta (β param accumulates failures)
- D4 UCB composition: optimistic for n<100 (+κ·(CI_hi−V)), pessimistic for
  n≥100 (−κ·(V−CI_lo))
- **F — Anti-monoculture bonus**: tanh-bounded (±κ_explore=0.15), penalizes
  over-sampled primitives, rewards under-sampled → structural fix for
  FORMULATE monoculture
- I1 β-score dispersion EMA (leading indicator for β influence)
- I2 Gini coefficient + usage shares in trajectory TSV
- I3 per-domain V with dynamic enumeration (≥10 obs threshold)

**P7 — EUREKA Accelerator + Advisor Disagreement**
- EUREKA-firing chains amplify Beta posterior: **5×** weight on trigger
  primitive, **3×** on supporting primitives, 1× baseline otherwise
- Natural quality passthrough (no EUREKA floor override)
- META_CGN_ADVISOR_CONFLICT bus event with 100-chain per-signature throttle
- H6_advisor_disagreement HAOV hypothesis (auto-seeded): does high α-vs-β
  disagreement correlate with low terminal reward?
- `/v4/meta-cgn/advisor-conflicts` endpoint

**P8 — SOAR-via-CGN Full Protocol**
- Multi-consumer broadcast on impasse (`CGN_KNOWLEDGE_REQ` dst="all")
- 2-second aggregation window per request_id with bounded FIFO cache
- **B-hybrid ranking**: `relevance · confidence` where `relevance = 0.5 ·
  source_affinity + 0.3 · keyword_overlap + 0.2 · domain_match`
- `SOURCE_AFFINITY` table: 4 impasse signals × 5 consumers, Maker-tunable
- META-CGN as responder: external `CGN_KNOWLEDGE_REQ` returns top-3 primitives
  by V_effective in requested domain (with self-loop protection)
- I-P8.1 request deduplication by `(signal, sha1(diagnostic)[:12])`
- I-P8.2 request_id UUID correlation (propagated through knowledge_worker)
- I-P8.3 post-injection `CGN_KNOWLEDGE_USAGE` feedback
- Source-credit tracking: `provided_by_source` / `helpful_by_source`

**P9 — Q-i Reward Blending with E1-E5 Enhancements**
- r_grounded enters terminal reward via graduation-state-driven weights
- Stage-driven blend: Bootstrap (0.5/0.5/0.0), Calibration ramp, Mature
  (0.2/0.3/0.5)
- **E1** pessimistic CI shift: `r_grounded = V − κ_ci · avg(CI_width)`
- **E2** H6-aware down-weight: confirmed H6 + high chain disagreement →
  divide r_grounded by H6.confidence_multiplier
- **E3** β-dispersion secondary gate: silent β (EMA<0.05) caps w_grounded
  at 0.05 regardless of stage
- **E4** per-domain bonus: ≥2 well-grounded primitives in chain's domain →
  w_grounded × 1.15
- **E5** `blend_weights_history.jsonl` audit trail with 500KB rotation
- D4 safety: disabled META-CGN auto-zeroes w_grounded

**P10 — Cross-Consumer Signal Flow (Layer 1)**
- New `META_CGN_SIGNAL` bus event: sparse, intentional
- Schema: `{consumer, event_type, intensity, domain?, narrative_context?}`
- `SIGNAL_TO_PRIMITIVE` mapping (11 entries, Maker-tunable): language,
  knowledge, social, coding, self_model, meta_wisdom → affected primitives
- Pseudo-observation update at weight 0.1 × intensity (small vs real chain
  evidence weight 1.0)
- Narrative bridge (Layer 2: DuckDB recall + reflection chains + writeback)
  moved to standalone rFP

**P11 — Kin Protocol v1 + Cross-Titan Grounding Transfer**
- New `/v1/kin/*` namespace for Titan-to-Titan communication
- `GET /v1/kin/identity` — handshake (titan_id, Solana pubkey, genesis NFT)
- `GET /v1/kin/peers` — genesis peer list from config
- `GET /v1/kin/meta-cgn/snapshot` — signed Ed25519 export of primitive
  grounding + HAOV state
- Import model: scaled merge at 0.5× confidence, self-import protection,
  version check, disabled-state safety
- Peer list in `titan_params.toml [kin.peers]` (hardcoded genesis, dynamic
  discovery future work)
- Architectural future-proofing for P2P / Manitou Network L2 — same
  namespace + signing + versioning extends to dynamic peer gossip

**P12 — API + Observability Polish**
- `GET /v4/meta-cgn/audit` — consolidated everything-in-one snapshot
- `GET /v4/meta-cgn/by-domain` — per-primitive × per-domain grounding table
- `arch_map meta-cgn audit` — one-screen diagnostic CLI
- `arch_map meta-cgn domains` — per-primitive domain breakdown
- `arch_map meta-cgn history` — blend_weights_history trajectory

**P13 — Task 4 Transitional Sunset (Gate, Don't Delete)**
- `[meta_reasoning_dna] sunset_task4 = false` flag (default preserves
  current behavior)
- When flipped: `_compute_meta_reward` skips the mono_adj hardcoded penalty;
  META-CGN's r_grounded (P9) provides the now-structural diversity signal
- Recommended flip criteria: 1 week stable gini<0.5, disagreement in
  [5%, 25%], no monoculture-attributed impasse, ≥2 updates/100chains for
  all 9 primitives

### Tests

96/96 green across P1-P11 test suite. Coverage:

- 49 original P1-P5 tests (construction, encoding, updates, HAOV, graduation,
  failsafe, impasse)
- 13 P6 tests (Beta posterior, CI narrowing, migration, F novelty bounding,
  UCB flip at N_ANCHOR, decay reversibility, per-domain divergence, impasse
  2× weighting, β-dispersion EMA, Gini monoculture signal, v3 round-trip)
- 7 P7 tests (EUREKA 5× weight, non-EUREKA baseline, trigger counts, bus
  event emission, throttle dedup, throttle cooldown, H6 observation + test)
- 8 P8 tests (request broadcast, dedup, window timeout, B-hybrid ranking,
  aggregation winner, responder, self-loop, source credit)
- 7 P9 tests (shadow-mode clean, active stage, β-dispersion gate, D4 safety,
  E1 pessimistic shift, E4 domain bonus, E5 audit trail)
- 6 P10 tests (known signal applies, unknown rejected, intensity scaling,
  narrative hook, failsafe safety, per-consumer counter)
- 6 P11 tests (schema, import with priors, version rejection, self-import,
  failsafe safety, per-domain merge)

### New API endpoints

| Endpoint | Phase | Purpose |
|---|---|---|
| `GET /v4/meta-cgn` | P1 | Full META-CGN telemetry |
| `GET /v4/meta-cgn/graduation-readiness` | P4 | Blockers view |
| `GET /v4/meta-cgn/failsafe-status` | P5 | Watchdog state |
| `GET /v4/meta-cgn/impasse-status` | P5 | F8 state |
| `GET /v4/meta-cgn/disagreements` | P5 | Recent shadow events |
| `GET /v4/meta-cgn/advisor-conflicts` | P7 | α-β conflicts + throttle stats |
| `GET /v4/meta-cgn/audit` | P12 | Consolidated snapshot |
| `GET /v4/meta-cgn/by-domain` | P12 | Per-domain grounding table |
| `GET /v1/kin/identity` | P11 | Titan identity handshake |
| `GET /v1/kin/peers` | P11 | Genesis peer list |
| `GET /v1/kin/meta-cgn/snapshot` | P11 | Signed export for cross-Titan transfer |

### New bus messages

| Message | Direction | Purpose |
|---|---|---|
| `CGN_REGISTER` | META-CGN → CGN worker | Consumer registration at init |
| `CGN_TRANSITION` | META-CGN → CGN worker | Per-primitive transitions for SharedValueNet training |
| `META_CGN_IMPASSE` | META-CGN → all | Impasse detection event |
| `META_CGN_FAILED` | META-CGN → all | Failsafe trip event |
| `META_CGN_ROLLED_BACK` | META-CGN → all | Active → shadow rollback |
| `META_CGN_ADVISOR_CONFLICT` | META-CGN → all | α-β strong disagreement (throttled) |
| `META_CGN_SIGNAL` | consumers → META-CGN | Cross-consumer grounding event |
| `CGN_KNOWLEDGE_REQ` | META-CGN → all | SOAR impasse knowledge request |
| `CGN_KNOWLEDGE_RESP` | consumers → META-CGN | Response (aggregated 2s window) |
| `CGN_KNOWLEDGE_USAGE` | META-CGN → source | Post-injection feedback |

### Files in this snapshot

- `titan_plugin/logic/meta_cgn.py` — MetaCGNConsumer + helpers (~2200 LOC)
- `titan_plugin/logic/meta_reasoning.py` — integration hooks for P6-P13
- `titan_plugin/modules/spirit_worker.py` — bus routing for signals + responder
- `titan_plugin/modules/knowledge_worker.py` — request_id propagation
- `titan_plugin/api/dashboard.py` — 11 META-CGN + Kin endpoints
- `titan_plugin/titan_params.toml` — `[meta_cgn]` + `[kin.peers]` sections
- `scripts/arch_map.py` — meta-cgn CLI with 8 subcommands
- `tests/test_meta_cgn.py` — 96 tests

### Architectural follow-on (future standalone rFPs)

- **Narrative meta-reasoning** — Layer 2 of P10, DuckDB recall + META-CGN
  reflection chains + writeback, closing the inner↔outer cognitive loop
- **Manitou Network v1** — L2 above Solana with dynamic peer discovery,
  on-chain registry for `/v1/kin/*` trust, P2P gossip extending the genesis
  peer list
- Consumer-side `META_CGN_SIGNAL` emitters in language/knowledge/social/
  coding/self_model workers (additive, no META-CGN changes needed)

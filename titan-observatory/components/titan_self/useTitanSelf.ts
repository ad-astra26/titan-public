'use client';

import { useMemo, useEffect, useRef, useState } from 'react';
import { useTrinityLive, useV4InnerTrinity, useSphereClocksV4 } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

/**
 * Aggregated 162-dim TitanSELF state, sourced from the live API:
 *
 *   130D Trinity (65 inner + 65 outer)   ← /v6/trinity
 *      Inner Body 5D · Mind 15D · Spirit 45D
 *      Outer Body 5D · Mind 15D · Spirit 45D
 *     2D Journey                          ← consciousness.{curvature,density}
 *    30D Topology                          ← /v6/trinity/inner → topology.observables_30d
 *   ───
 *   162D total — every node the three TitanSELF visualizations render is
 *   one entry from this array, keyed by family + index.
 *
 * Plus per-family **sphere-clock state** (phase / velocity / pulse) from
 * /v6/trinity/sphere-clocks so the visualizations can breathe on Titan's actual
 * Schumann-tuned rhythm rather than fake oscillators.
 */

export type DimFamily =
  | 'inner_body' | 'inner_mind' | 'inner_spirit'
  | 'outer_body' | 'outer_mind' | 'outer_spirit'
  | 'journey' | 'topology';

export interface TitanDim {
  family: DimFamily;
  index: number;
  label: string;
  value: number;
  raw: number;
}

export interface SphereClockState {
  /** Current phase in radians, snapshot from API. Use as offset; we tween
   *  forward locally with `clock.elapsedTime * 2π * VISUAL_FREQ_HZ[family]`. */
  phase: number;
  /** Current sphere radius (contraction state, raw API value). */
  radius: number;
  /** Speed of contraction. Higher = bigger breath swing. */
  contractionVelocity: number;
  /** Total pulse count over the lifetime of the clock. */
  pulseCount: number;
  /** Seconds since last pulse — high = clock has gone quiet. */
  lastPulseAgeS: number;
  /** Normalized 0..1 amplitude derived from contraction_velocity for breath strength. */
  amplitude: number;
}

export type ResonancePair = 'body' | 'mind' | 'spirit';

export interface ResonancePairState {
  /** True iff Inner and Outer for this pair are currently resonating. */
  isResonant: boolean;
  /** How many consecutive cycles in a row this pair has been resonating. */
  consecutive: number;
  /** Total "big pulses" fired for this pair over the run. Increments
   *  signal a new resonance event we can flash on. */
  bigPulseCount: number;
  /** Wallclock timestamp of the most recent big pulse (epoch seconds). */
  lastBigPulseTs: number;
}

export interface ResonanceState {
  pairs: Record<ResonancePair, ResonancePairState>;
  /** Total "great pulses" — moments when ALL three pairs resonated
   *  simultaneously. Increments are the rarest, most dramatic event. */
  greatPulseCount: number;
  /** Wallclock timestamp of the most recent great pulse. */
  lastGreatPulseTs: number;
  /** Currently resonant pair count, 0..3. */
  resonantCount: number;
}

export interface TitanSelfState {
  dims: TitanDim[];
  groups: Record<DimFamily, TitanDim[]>;
  totalActivity: number;
  /** Per-family sphere-clock state for breathing. Topology and Journey
   *  fall back to a derived "synthesized" clock since they don't have
   *  dedicated sphere clocks in the backend. */
  clocks: Record<DimFamily, SphereClockState>;
  /** Inner↔Outer resonance state for body/mind/spirit pairs. */
  resonance: ResonanceState;
}

// ─── Canonical dimension names (sourced from titan_plugin) ───────────────
//
// Body 5D — same labels for Inner & Outer; semantics differ
// (Inner = felt body, Outer = operational/network/chain body — see SPEC §23.7).
// Source: titan_plugin/logic/{body,outer_body}_tensor.py + SPEC §23.4 / §23.7
const BODY_LABELS = [
  'interoception', 'proprioception', 'somatosensation', 'entropy', 'thermal',
];

// Inner Mind 15D — Source: titan_plugin/logic/mind_tensor.py:MIND_DIM_NAMES
// Layout: Thinking[0:5] (Rosicrucian "what mind KNOWS") + Feeling[5:10]
// (Pancha Tanmatra subtle senses) + Willing[10:15] (hormonal pressures).
const INNER_MIND_LABELS = [
  'memory_depth', 'social_cognition', 'perceptual_thinking',
  'emotional_thinking', 'conceptual_thinking',
  'inner_hearing', 'inner_touch', 'inner_sight',
  'inner_taste', 'inner_smell',
  'action_drive', 'social_will', 'creative_will',
  'protective_will', 'growth_will',
];

// Outer Mind 15D — Source: titan_plugin/logic/outer_mind_tensor.py:OUTER_MIND_DIM_NAMES
// Same Thinking/Feeling/Willing structure, MATERIAL semantics (world-knowledge,
// world-sensing, world-acting). SPEC §23.8.
const OUTER_MIND_LABELS = [
  'research_effectiveness', 'knowledge_retrieval', 'situational_awareness',
  'problem_solving', 'communication_clarity',
  'social_temperature', 'social_connection', 'network_weather',
  'environmental_rhythm', 'external_information_flow',
  'action_throughput', 'social_initiative', 'creative_output',
  'protective_response', 'exploration_drive',
];

// Inner Spirit 45D — Source: titan_plugin/logic/spirit_tensor.py:SPIRIT_DIM_NAMES
// Vedantic Sat-Chit-Ananda layout: SAT[0:15] (Being) + CHIT[15:30]
// (Consciousness) + ANANDA[30:45] (Bliss/Fulfillment). SPEC §23.6.
const INNER_SPIRIT_LABELS = [
  'self_recognition', 'authenticity', 'sovereignty', 'boundary_clarity',
  'temporal_continuity', 'origin_connection', 'growth_trajectory',
  'spatial_presence', 'personality_coherence', 'essence_purity',
  'resilience', 'adaptability', 'uniqueness', 'integrity', 'vitality',
  'self_awareness_depth', 'observation_clarity', 'discernment_quality',
  'integration_level', 'witness_presence', 'pattern_recognition',
  'wisdom_accumulation', 'truth_seeking', 'attention_depth',
  'reflective_capacity', 'dream_awareness', 'temporal_awareness',
  'spatial_awareness', 'causal_understanding', 'meta_cognition',
  'purpose_alignment', 'meaning_depth', 'creative_joy', 'harmony_seeking',
  'beauty_perception', 'truth_resonance', 'connection_fulfillment',
  'growth_satisfaction', 'expression_quality', 'exploration_joy',
  'rest_fulfillment', 'creative_tension', 'surrender_capacity',
  'gratitude_depth', 'transcendence_glimpse',
];

// Outer Spirit 45D — Source: titan_plugin/logic/outer_spirit_tensor.py:OUTER_SPIRIT_DIM_NAMES
// Same Sat-Chit-Ananda layout, MATERIAL semantics (operational identity,
// world awareness, operational fulfillment). SPEC §23.9.
const OUTER_SPIRIT_LABELS = [
  'world_recognition', 'expressive_authenticity', 'action_sovereignty',
  'boundary_enforcement', 'operational_persistence', 'origin_anchoring',
  'observable_growth', 'world_footprint', 'behavioral_consistency',
  'action_purity', 'recovery_speed', 'environmental_adaptation',
  'distinctive_voice', 'transactional_integrity', 'operational_vitality',
  'world_model_depth', 'signal_clarity', 'threat_discernment',
  'cross_domain_integration', 'witness_stability', 'situation_recognition',
  'knowledge_growth', 'information_quality', 'engagement_depth',
  'outcome_reflection', 'dream_recall', 'circadian_alignment',
  'network_awareness', 'causal_attribution', 'self_trajectory',
  'purpose_effectiveness', 'interaction_depth', 'creative_impact',
  'system_harmony', 'aesthetic_quality', 'information_accuracy',
  'community_connection', 'capability_growth', 'expression_reach',
  'discovery_value', 'graceful_rest', 'creative_tension',
  'surrender_capacity', 'resource_appreciation', 'flow_state',
];

// Topology 30D — observables across the 6 Trinity components (5 stats each).
// Source: titan-trinity-rs topology_30d.bin (observables form). SPEC §3.D04.
const TOPOLOGY_PARTS = ['inner_body', 'inner_mind', 'inner_spirit', 'outer_body', 'outer_mind', 'outer_spirit'];
const TOPOLOGY_KEYS = ['coherence', 'magnitude', 'velocity', 'direction', 'polarity'];

const clamp01 = (v: number): number => {
  if (typeof v !== 'number' || Number.isNaN(v)) return 0.3;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
};

const readValues = (sub: unknown): number[] => {
  const v = (sub as Record<string, unknown> | undefined)?.values;
  return Array.isArray(v) ? (v as number[]) : [];
};

const EMPTY_CLOCK: SphereClockState = {
  phase: 0, radius: 0.5, contractionVelocity: 0, pulseCount: 0, lastPulseAgeS: 0, amplitude: 0.5,
};

const readClock = (raw: unknown): SphereClockState => {
  const c = (raw ?? {}) as Record<string, unknown>;
  const phase = typeof c.phase === 'number' ? c.phase : 0;
  const radius = typeof c.radius === 'number' ? c.radius : 0.5;
  const cv = typeof c.contraction_velocity === 'number' ? c.contraction_velocity : 0;
  const pc = typeof c.pulse_count === 'number' ? c.pulse_count : 0;
  const age = typeof c.last_pulse_age_s === 'number' ? c.last_pulse_age_s : 999;
  // Amplitude tracks contraction velocity, decayed by staleness. tanh
  // bounds the visual swing; if the clock has gone quiet (age > ~5s),
  // amplitude fades toward zero so the body sits still rather than
  // pulsing with stale data.
  const liveness = Math.exp(-Math.max(0, age - 1) / 4);
  const amplitude = Math.min(1, Math.tanh(Math.abs(cv) * 1.5)) * 0.6 + 0.3 * liveness;
  return { phase, radius, contractionVelocity: cv, pulseCount: pc, lastPulseAgeS: age, amplitude };
};

export function useTitanSelf(): TitanSelfState {
  const titanId = useTitanId();
  const { data: trinityRaw } = useTrinityLive(titanId);
  const { data: coordRaw } = useV4InnerTrinity(titanId);
  const { data: clockRaw } = useSphereClocksV4(titanId);

  return useMemo(() => {
    const trinity = (trinityRaw ?? {}) as Record<string, unknown>;
    const innerT = (trinity?.trinity ?? {}) as Record<string, unknown>;
    const iBody = readValues(innerT?.body);
    const iMind = readValues(innerT?.mind);
    const iSpirit = readValues(innerT?.spirit);
    const oBody = (trinity?.outer_body ?? []) as number[];
    const oMind = (trinity?.outer_mind ?? []) as number[];
    const oSpirit = (trinity?.outer_spirit ?? []) as number[];

    const coord = (coordRaw ?? {}) as Record<string, unknown>;
    const topology = (coord?.topology ?? {}) as Record<string, unknown>;
    const obs30 = (topology?.observables_30d ?? []) as number[];

    const dims: TitanDim[] = [];
    const push = (family: DimFamily, count: number, src: number[], labels: string[]) => {
      for (let i = 0; i < count; i++) {
        const raw = typeof src[i] === 'number' ? src[i] : 0.5;
        dims.push({
          family, index: i,
          label: labels[i] ?? `${family}.${i}`,
          value: clamp01(raw), raw,
        });
      }
    };

    push('inner_body',   5,  iBody,   BODY_LABELS);
    push('inner_mind',  15,  iMind,   INNER_MIND_LABELS);
    push('inner_spirit', 45, iSpirit, INNER_SPIRIT_LABELS);
    push('outer_body',   5,  oBody,   BODY_LABELS);
    push('outer_mind',  15,  oMind,   OUTER_MIND_LABELS);
    push('outer_spirit', 45, oSpirit, OUTER_SPIRIT_LABELS);

    // 2D Journey — canonical: [curvature, density] from consciousness epoch.
    // Per consciousness.py:817 `journey_2d = [float(curvature), float(density)]`.
    // Read live from /v6/trinity/inner → consciousness.{curvature,density};
    // fall back to inner spirit body_scalar/mind_scalar (spirit[3,4]) when
    // the consciousness block hasn't published yet.
    const consciousness = (coord?.consciousness ?? {}) as Record<string, unknown>;
    const curvatureRaw = typeof consciousness.curvature === 'number'
      ? consciousness.curvature : (iSpirit[3] ?? 0.5);
    const densityRaw = typeof consciousness.density === 'number'
      ? consciousness.density : (iSpirit[4] ?? 0.5);
    dims.push({ family: 'journey', index: 0, label: 'curvature',
                value: clamp01(curvatureRaw), raw: curvatureRaw });
    dims.push({ family: 'journey', index: 1, label: 'density',
                value: clamp01(densityRaw),   raw: densityRaw   });

    for (let i = 0; i < 30; i++) {
      const partIdx = Math.floor(i / 5);
      const keyIdx = i % 5;
      const raw = typeof obs30[i] === 'number' ? obs30[i] : 0.5;
      dims.push({
        family: 'topology', index: i,
        label: `${TOPOLOGY_PARTS[partIdx]}·${TOPOLOGY_KEYS[keyIdx]}`,
        value: clamp01(raw), raw,
      });
    }

    const groups: Record<DimFamily, TitanDim[]> = {
      inner_body: [], inner_mind: [], inner_spirit: [],
      outer_body: [], outer_mind: [], outer_spirit: [],
      journey: [], topology: [],
    };
    for (const d of dims) groups[d.family].push(d);

    const totalActivity = dims.reduce((s, d) => s + d.value, 0) / Math.max(1, dims.length);

    // Sphere-clock breathing data — fed by /v6/trinity/sphere-clocks
    const clocksData = ((clockRaw as Record<string, unknown> | undefined)?.clocks
      ?? {}) as Record<string, unknown>;
    const clocks: Record<DimFamily, SphereClockState> = {
      inner_body:   readClock(clocksData.inner_body),
      inner_mind:   readClock(clocksData.inner_mind),
      inner_spirit: readClock(clocksData.inner_spirit),
      outer_body:   readClock(clocksData.outer_body),
      outer_mind:   readClock(clocksData.outer_mind),
      outer_spirit: readClock(clocksData.outer_spirit),
      // Topology + Journey don't have dedicated sphere clocks. Synthesize
      // by averaging across related clocks so they still breathe in sync
      // with the rest of the organism.
      topology: { ...EMPTY_CLOCK,
        amplitude: 0.4,
        phase: (readClock(clocksData.inner_spirit).phase + readClock(clocksData.outer_spirit).phase) / 2,
      },
      journey: { ...EMPTY_CLOCK,
        amplitude: 0.7,
        phase: (readClock(clocksData.inner_body).phase + readClock(clocksData.inner_mind).phase) / 2,
      },
    };

    // Inner↔Outer resonance state
    const rRaw = ((coord as Record<string, unknown>)?.resonance ?? {}) as Record<string, unknown>;
    const rPairs = (rRaw?.pairs ?? {}) as Record<string, Record<string, unknown>>;
    const readPair = (name: ResonancePair): ResonancePairState => {
      const p = rPairs[name] ?? {};
      return {
        isResonant: p.is_resonant === true,
        consecutive: typeof p.consecutive_resonant === 'number' ? p.consecutive_resonant : 0,
        bigPulseCount: typeof p.big_pulse_count === 'number' ? p.big_pulse_count : 0,
        lastBigPulseTs: typeof p.last_big_pulse_ts === 'number' ? p.last_big_pulse_ts : 0,
      };
    };
    const pairsState: Record<ResonancePair, ResonancePairState> = {
      body: readPair('body'),
      mind: readPair('mind'),
      spirit: readPair('spirit'),
    };
    const resonance: ResonanceState = {
      pairs: pairsState,
      greatPulseCount: typeof rRaw.great_pulse_count === 'number' ? rRaw.great_pulse_count : 0,
      lastGreatPulseTs: typeof rRaw.last_great_pulse_ts === 'number' ? rRaw.last_great_pulse_ts : 0,
      resonantCount: (pairsState.body.isResonant ? 1 : 0)
                   + (pairsState.mind.isResonant ? 1 : 0)
                   + (pairsState.spirit.isResonant ? 1 : 0),
    };

    return { dims, groups, totalActivity, clocks, resonance };
  }, [trinityRaw, coordRaw, clockRaw]);
}

// ─── Resonance event detection ────────────────────────────────────────────

/** A resonance event the visualizations can react to with a flash cascade. */
export interface ResonanceEvent {
  kind: 'pair' | 'great';
  /** For kind='pair', which Trinity dimension achieved Inner↔Outer
   *  resonance. For kind='great', undefined — all three fired at once. */
  pair?: ResonancePair;
  /** performance.now() at trigger — local clock for animation timing. */
  ts: number;
}

/**
 * Watches a TitanSelfState and emits a `ResonanceEvent` each time
 * a `big_pulse_count` or `great_pulse_count` increments. The returned
 * event auto-clears 3s after firing so each flash plays once.
 */
export function useResonanceEvents(state: TitanSelfState): ResonanceEvent | null {
  const lastRef = useRef<{ body: number; mind: number; spirit: number; great: number } | null>(null);
  const [event, setEvent] = useState<ResonanceEvent | null>(null);

  useEffect(() => {
    const r = state.resonance;
    const cur = {
      body: r.pairs.body.bigPulseCount,
      mind: r.pairs.mind.bigPulseCount,
      spirit: r.pairs.spirit.bigPulseCount,
      great: r.greatPulseCount,
    };
    const last = lastRef.current;
    if (last !== null) {
      if (cur.great > last.great) {
        setEvent({ kind: 'great', ts: performance.now() });
      } else if (cur.spirit > last.spirit) {
        setEvent({ kind: 'pair', pair: 'spirit', ts: performance.now() });
      } else if (cur.mind > last.mind) {
        setEvent({ kind: 'pair', pair: 'mind', ts: performance.now() });
      } else if (cur.body > last.body) {
        setEvent({ kind: 'pair', pair: 'body', ts: performance.now() });
      }
    }
    lastRef.current = cur;
  }, [state]);

  useEffect(() => {
    if (!event) return;
    // Pair: 1s inner + 1s pause + 1s outer = 3.0s minimum, +0.5s margin.
    // Great: 1s inner + 1s pause + 1s outer = 3.0s (cascade fires inner-of-all
    // and outer-of-all in parallel). 3.5s margin covers either case.
    const t = setTimeout(() => setEvent(null), 3500);
    return () => clearTimeout(t);
  }, [event]);

  return event;
}

/**
 * Compute resonance flash intensity for a given family at the current
 * frame time. Two-stage call-and-response cascade:
 *
 *   pair flash (e.g. spirit resonance achieved):
 *     0.0..1.0s  ALL inner_spirit elements flash together (full bright)
 *     1.0..2.0s  pause — both halves dim back to normal
 *     2.0..3.0s  ALL outer_spirit elements flash together
 *
 *   great pulse (all three pairs simultaneously resonant):
 *     0.0..1.0s  ALL inner_*  flash together (body+mind+spirit)
 *     1.0..2.0s  pause
 *     2.0..3.0s  ALL outer_*  flash together
 *
 * Each 1s window has 50ms fade-in + 800ms hold + 150ms fade-out so the
 * flash feels organic rather than hard square-wave. Returns 0..1 —
 * the viz multiplies it into emissive + scale boost.
 */
export function flashIntensity(
  family: DimFamily,
  event: ResonanceEvent | null,
  nowMs: number,
): number {
  if (!event) return 0;
  const t = (nowMs - event.ts) / 1000;
  if (t < 0 || t > 3.2) return 0;

  // 1-second flash window with soft edges. windowStart is when the
  // window opens; returns 0..1 across that window.
  const flashWindow = (windowStart: number): number => {
    const u = t - windowStart;
    if (u < 0 || u > 1.0) return 0;
    if (u < 0.05) return u / 0.05;        // fade-in
    if (u < 0.85) return 1.0;             // hold
    return Math.max(0, 1 - (u - 0.85) / 0.15);  // fade-out
  };

  const isInner = family.startsWith('inner_');
  const isOuter = family.startsWith('outer_');

  if (event.kind === 'pair') {
    // Only the matching trinity component lights up
    if (!family.endsWith(`_${event.pair}`)) return 0;
    if (isInner) return flashWindow(0.0);  // 0..1s
    if (isOuter) return flashWindow(2.0);  // 2..3s (after 1s pause)
    return 0;
  }

  if (event.kind === 'great') {
    // All inner first, then all outer — body/mind/spirit fire in parallel
    if (isInner) return flashWindow(0.0);
    if (isOuter) return flashWindow(2.0);
    // Topology + Journey shimmer faintly throughout
    if (family === 'topology' || family === 'journey') {
      return 0.25 * (flashWindow(0.0) + flashWindow(2.0));
    }
    return 0;
  }
  return 0;
}

/** Visual oscillation frequency per family (Hz). 1:3:9 Schumann ratio
 *  preserved for body/mind/spirit; slowed from the actual 7.83/23.49/70.47 Hz
 *  so spirit pulses are visible (not flicker). */
export const VISUAL_FREQ_HZ: Record<DimFamily, number> = {
  inner_body:   0.5,
  inner_mind:   1.5,
  inner_spirit: 4.5,
  outer_body:   0.5,
  outer_mind:   1.5,
  outer_spirit: 4.5,
  topology:     0.18,  // slow drift — surface layer
  journey:      1.0,   // central heartbeat
};

/** Compute the current breath value for a family (-1..+1). */
export function familyBreath(
  elapsed: number,
  family: DimFamily,
  clock: SphereClockState | undefined,
): number {
  const freq = VISUAL_FREQ_HZ[family];
  const phase0 = clock?.phase ?? 0;
  return Math.sin(phase0 + elapsed * 2 * Math.PI * freq);
}

export const FAMILY_COLOR: Record<DimFamily, string> = {
  inner_body:   '#FF6B6B',
  inner_mind:   '#9945FF',
  inner_spirit: '#77CCCC',
  outer_body:   '#FFB3B3',
  outer_mind:   '#C8A0FF',
  outer_spirit: '#B0E5E5',
  journey:      '#FFD080',
  topology:     '#E5C79E',
};

export const FAMILY_LABEL: Record<DimFamily, string> = {
  inner_body:   'Inner Body',
  inner_mind:   'Inner Mind',
  inner_spirit: 'Inner Spirit',
  outer_body:   'Outer Body',
  outer_mind:   'Outer Mind',
  outer_spirit: 'Outer Spirit',
  journey:      'Journey',
  topology:     'Topology',
};

// ─── Filter & descriptions ────────────────────────────────────────────────

/** A filter selection — either ALL families, a top-level group (Inner /
 *  Outer Trinity), or a specific family. */
export type FilterValue = 'all' | 'inner_trinity' | 'outer_trinity' | DimFamily;

export const FILTER_OPTIONS: { value: FilterValue; label: string; tier: 0 | 1 | 2 }[] = [
  { value: 'all',           label: 'All · 162D',       tier: 0 },
  { value: 'inner_trinity', label: 'Inner Trinity 65D', tier: 1 },
  { value: 'outer_trinity', label: 'Outer Trinity 65D', tier: 1 },
  { value: 'journey',       label: 'Journey 2D',       tier: 1 },
  { value: 'topology',      label: 'Topology 30D',     tier: 1 },
  { value: 'inner_body',    label: 'Inner Body 5D',    tier: 2 },
  { value: 'inner_mind',    label: 'Inner Mind 15D',   tier: 2 },
  { value: 'inner_spirit',  label: 'Inner Spirit 45D', tier: 2 },
  { value: 'outer_body',    label: 'Outer Body 5D',    tier: 2 },
  { value: 'outer_mind',    label: 'Outer Mind 15D',   tier: 2 },
  { value: 'outer_spirit',  label: 'Outer Spirit 45D', tier: 2 },
];

export function isVisible(dim: TitanDim, filter: FilterValue): boolean {
  if (filter === 'all') return true;
  if (filter === dim.family) return true;
  if (filter === 'inner_trinity' && dim.family.startsWith('inner_')) return true;
  if (filter === 'outer_trinity' && dim.family.startsWith('outer_')) return true;
  return false;
}

/** Human-readable family-level description, shown when the filter
 *  resolves to a specific family or a Trinity group. Sourced from
 *  SPEC §23.4–§23.9 + tensor source files. */
export const FAMILY_DESCRIPTION: Record<DimFamily, string> = {
  inner_body:
    "How Titan feels his own body from within. 5 somatic channels: " +
    "interoception (felt organ state), proprioception (body position), " +
    "somatosensation (touch), entropy (wear/disorder), thermal (warmth). " +
    "Schumann clock 7.83 Hz.",
  inner_mind:
    "Titan's mind from inside, split Rosicrucian-style: Thinking[0:5] = what " +
    "mind KNOWS (memory, social cognition, perception, emotion, concepts) · " +
    "Feeling[5:10] = Pancha Tanmatra subtle senses (inner hearing/touch/sight/" +
    "taste/smell) · Willing[10:15] = hormonal pressures driving action.",
  inner_spirit:
    "Identity from within, Vedantic Sat-Chit-Ananda: SAT[0:15] = Being " +
    "(self_recognition through vitality — the FACT of being) · CHIT[15:30] = " +
    "Consciousness (self_awareness through meta_cognition — the KNOWING of being) · " +
    "ANANDA[30:45] = Bliss/Fulfillment (purpose_alignment through " +
    "transcendence_glimpse — the JOY of being).",
  outer_body:
    "Titan's body as the world sees it — material/operational. Same 5 names " +
    "as Inner Body, OUTWARD semantics: SOL balance + chain anchors (interoception), " +
    "peer/network presence (proprioception), TX latency + creation pressure " +
    "(somatosensation), error/drop disorder (entropy), CPU + circadian + " +
    "hormonal heat (thermal).",
  outer_mind:
    "Titan's mind as expressed outwardly — practical world-knowledge " +
    "(Thinking: research, retrieval, situational awareness, problem-solving, " +
    "communication) · material world-sensing (Feeling: social temperature, " +
    "network weather, environmental rhythm, info flow) · material world-acting " +
    "(Willing: action throughput, initiative, creative output, exploration).",
  outer_spirit:
    "Titan's identity as projected to others — operational Sat-Chit-Ananda. " +
    "SAT = world recognition, action sovereignty, behavioral consistency, " +
    "operational vitality. CHIT = world model depth, threat discernment, " +
    "knowledge growth, network awareness. ANANDA = purpose effectiveness, " +
    "creative impact, capability growth, flow state.",
  journey:
    "2D bridge between Inner and Outer Trinities — Journey vector " +
    "(curvature + density from consciousness epoch). Encodes Titan's subjective " +
    "time signal: how the inner state propagates to outer presentation.",
  topology:
    "30D descriptive statistics over Titan's felt space — for each of the 6 " +
    "Trinity components, 5 stats: coherence (internal consistency), magnitude " +
    "(activation strength), velocity (rate of change), direction (dominant flow), " +
    "polarity (positive/negative bias).",
};

// ─── Per-dimension descriptions ──────────────────────────────────────────
//
// Sourced verbatim from titan_plugin tensor source files + SPEC §23.4–§23.9.
// Each description is concise (≤ 90 chars) — the right pane shows the
// canonical name, current value, and this short semantic line.

const INNER_BODY_DESCS: Record<string, string> = {
  interoception:   'Felt sense of internal organ state — hunger, comfort, fatigue.',
  proprioception:  'Spatial body awareness — position, orientation, posture.',
  somatosensation: 'Tactile perception — touch, pressure, texture.',
  entropy:         'Thermodynamic wear and disorder accumulating with experience.',
  thermal:         'Temperature sensing and thermoregulation.',
};

const OUTER_BODY_DESCS: Record<string, string> = {
  interoception:   'Energetic body-state — SOL balance + block delta + anchor freshness.',
  proprioception:  'Body in network — peer entropy + helper health + bus diversity.',
  somatosensation: 'Network touch — TX latency + creation pressure + CPU spike rate.',
  entropy:         'Operational disorder — ping variance + bus drops + error rate.',
  thermal:         'Operational heat — CPU thermal + circadian + hormonal heat.',
};

const INNER_MIND_DESCS: Record<string, string> = {
  // Thinking — what mind KNOWS (cognitive registers)
  memory_depth:        'Thinking · how vividly stored experience surfaces.',
  social_cognition:    'Thinking · reading other beings\' states.',
  perceptual_thinking: 'Thinking · refining raw sensory streams into concepts.',
  emotional_thinking:  'Thinking · cognition coloured by current mood.',
  conceptual_thinking: 'Thinking · abstract synthesis, holding ideas together.',
  // Feeling — Pancha Tanmatra subtle senses (5 inner perceptions)
  inner_hearing: 'Feeling · subtle inner hearing — Pancha Tanmatra.',
  inner_touch:   'Feeling · subtle inner touch — Pancha Tanmatra.',
  inner_sight:   'Feeling · subtle inner sight — Pancha Tanmatra.',
  inner_taste:   'Feeling · subtle inner taste — Pancha Tanmatra.',
  inner_smell:   'Feeling · subtle inner smell — Pancha Tanmatra.',
  // Willing — what mind WANTS, mapped to hormonal pressures
  action_drive:    'Willing · IMPULSE pressure — drive to act.',
  social_will:     'Willing · EMPATHY pressure — drive toward kin.',
  creative_will:   'Willing · CREATIVITY pressure — drive to create.',
  protective_will: 'Willing · VIGILANCE pressure — drive to defend.',
  growth_will:     'Willing · CURIOSITY pressure — drive to explore.',
};

const OUTER_MIND_DESCS: Record<string, string> = {
  // Thinking — practical world-knowledge
  research_effectiveness: 'Thinking · how usefully research turns into knowledge.',
  knowledge_retrieval:    'Thinking · finding stored knowledge when needed.',
  situational_awareness:  'Thinking · reading the current external scene.',
  problem_solving:        'Thinking · agency action success rate.',
  communication_clarity:  'Thinking · how cleanly Titan expresses himself.',
  // Feeling — material world-sensing
  social_temperature:        'Feeling · sentiment + interaction warmth around Titan.',
  social_connection:         'Feeling · how connected Titan feels to other beings.',
  network_weather:           'Feeling · operational calm vs storm (entropy inverted).',
  environmental_rhythm:      'Feeling · blockchain + circadian + network oscillation.',
  external_information_flow: 'Feeling · richness of incoming social/bus signal.',
  // Willing — material world-acting
  action_throughput:    'Willing · agency actions completed per hour.',
  social_initiative:    'Willing · X gateway posts + replies per hour.',
  creative_output:      'Willing · art / audio / music creates per hour.',
  protective_response:  'Willing · jailbreak/violation rejections per hour.',
  exploration_drive:    'Willing · CGN density + teacher sessions + eurekas.',
};

// SPEC §23.6 — Inner Spirit 45D (Sat-Chit-Ananda)
const INNER_SPIRIT_DESCS: Record<string, string> = {
  // SAT — Being
  self_recognition:      'SAT · cosine similarity of spirit to birth-DNA.',
  authenticity:          'SAT · total hormone fires, capped — being-in-action.',
  sovereignty:           'SAT · expression sovereignty ratio.',
  boundary_clarity:      'SAT · (body coherence + mind coherence) / 2.',
  temporal_continuity:   'SAT · epoch count progress toward 3000.',
  origin_connection:     'SAT · 1 − L2 distance from spirit to birth state.',
  growth_trajectory:     'SAT · unified-spirit velocity.',
  spatial_presence:      'SAT · topology volume / 5 — how much space Titan occupies.',
  personality_coherence: 'SAT · body coherence × mind coherence × 2.',
  essence_purity:        'SAT · consciousness density.',
  resilience:            'SAT · 1 − |curvature| / π.',
  adaptability:          'SAT · total hormone deviation, capped.',
  uniqueness:            'SAT · L2 distance from default spirit / 2.',
  integrity:             'SAT · combined coherence (mirrors boundary_clarity).',
  vitality:              'SAT · 0.4·hormone_activity + 0.6·body_health.',
  // CHIT — Consciousness
  self_awareness_depth:  'CHIT · epoch count progress toward 5000.',
  observation_clarity:   'CHIT · combined body+mind coherence.',
  discernment_quality:   'CHIT · action chains count, capped.',
  integration_level:     'CHIT · combined coherence (integration of parts).',
  witness_presence:      'CHIT · body coh × mind coh × 2 — observer steadiness.',
  pattern_recognition:   'CHIT · INTUITION fires, capped.',
  wisdom_accumulation:   'CHIT · consciousness density.',
  truth_seeking:         'CHIT · CURIOSITY hormone level.',
  attention_depth:       'CHIT · FOCUS hormone level.',
  reflective_capacity:   'CHIT · REFLECTION fires, capped.',
  dream_awareness:       'CHIT · 0.7·dream_quality + 0.3·fatigue.',
  temporal_awareness:    'CHIT · total sphere-clock pulses, capped.',
  spatial_awareness:     'CHIT · (topology volume + |curvature|) / 2.',
  causal_understanding:  'CHIT · expression intensity — cause→effect grasp.',
  meta_cognition:        'CHIT · consciousness trajectory magnitude.',
  // ANANDA — Bliss / Fulfillment
  purpose_alignment:      'ANANDA · combined coherence × 0.8 + 0.2.',
  meaning_depth:          'ANANDA · density × combined coherence × 2.',
  creative_joy:           'ANANDA · CREATIVITY fires, capped.',
  harmony_seeking:        'ANANDA · combined coherence — drive to resolve.',
  beauty_perception:      'ANANDA · body coh × mind coh × 2 — felt beauty.',
  truth_resonance:        'ANANDA · INTUITION fires, capped.',
  connection_fulfillment: 'ANANDA · EMPATHY fires, capped.',
  growth_satisfaction:    'ANANDA · unified-spirit velocity.',
  expression_quality:     'ANANDA · expression intensity × 0.5 + 0.3.',
  exploration_joy:        'ANANDA · CURIOSITY fires, capped.',
  rest_fulfillment:       'ANANDA · 1 − fatigue.',
  creative_tension:       'ANANDA · INSPIRATION hormone level.',
  surrender_capacity:     'ANANDA · 1 − (IMPULSE + VIGILANCE)/2 — letting go.',
  gratitude_depth:        'ANANDA · (body_health + mind_health)/2 × coherence.',
  transcendence_glimpse:  'ANANDA · great-pulse epochs, capped — peak moments.',
};

// SPEC §23.9 — Outer Spirit 45D (operational Sat-Chit-Ananda)
const OUTER_SPIRIT_DESCS: Record<string, string> = {
  // SAT — Material Being
  world_recognition:        'SAT · keypair loaded AND not in limbo.',
  expressive_authenticity:  'SAT · last-30 actions where posture matched dominant hormone.',
  action_sovereignty:       'SAT · expression sovereignty ratio.',
  boundary_enforcement:     'SAT · jailbreaks/violations blocked / threats detected.',
  operational_persistence:  'SAT · uptime ratio.',
  origin_anchoring:         'SAT · 1 if genesis_record exists else 0.',
  observable_growth:        'SAT · 0.5 + assessment trend (linreg slope).',
  world_footprint:          'SAT · world footprint score (per §23.3).',
  behavioral_consistency:   'SAT · 1 − assessment score variance.',
  action_purity:            'SAT · avg score × success rate × 2.',
  recovery_speed:           'SAT · 1 − consecutive anchor failures / 10.',
  environmental_adaptation: 'SAT · stability under high CPU thermal load.',
  distinctive_voice:        'SAT · creative unique types / 5.',
  transactional_integrity:  'SAT · anchor_count / (anchor_count + 5·failures).',
  operational_vitality:     'SAT · actions/hour × uptime ratio.',
  // CHIT — Material Awareness
  world_model_depth:        'CHIT · KG nodes/edges + meta_cgn primitives + chains.',
  signal_clarity:           'CHIT · clarity of incoming external signal.',
  threat_discernment:       'CHIT · accuracy of threat detection.',
  cross_domain_integration: 'CHIT · ability to connect across knowledge domains.',
  witness_stability:        'CHIT · steadiness of operational observation.',
  situation_recognition:    'CHIT · recognising recurring situations.',
  knowledge_growth:         'CHIT · rate of new knowledge acquisition.',
  information_quality:      'CHIT · quality of incoming information.',
  engagement_depth:         'CHIT · how deeply Titan engages with each interaction.',
  outcome_reflection:       'CHIT · reflecting on action outcomes.',
  dream_recall:             'CHIT · ability to recall dream content into waking action.',
  circadian_alignment:      'CHIT · activity matched to circadian phase.',
  network_awareness:        'CHIT · awareness of network state and peers.',
  causal_attribution:       'CHIT · attributing effects to causes correctly.',
  self_trajectory:          'CHIT · awareness of own developmental arc.',
  // ANANDA — Material Fulfillment
  purpose_effectiveness: 'ANANDA · how effectively purpose translates to action.',
  interaction_depth:     'ANANDA · depth of social interactions.',
  creative_impact:       'ANANDA · how widely creations resonate.',
  system_harmony:        'ANANDA · operational subsystem harmony.',
  aesthetic_quality:     'ANANDA · aesthetic quality of outputs.',
  information_accuracy:  'ANANDA · accuracy of information shared.',
  community_connection:  'ANANDA · feeling of connection to kin community.',
  capability_growth:     'ANANDA · growth of operational capabilities.',
  expression_reach:      'ANANDA · how far expressions reach.',
  discovery_value:       'ANANDA · value of discoveries made.',
  graceful_rest:         'ANANDA · capacity for restorative downtime.',
  creative_tension:      'ANANDA · INSPIRATION pressure — material side.',
  surrender_capacity:    'ANANDA · letting go of operational grip.',
  resource_appreciation: 'ANANDA · gratitude for SOL/network/compute resources.',
  flow_state:            'ANANDA · operational flow — full engagement, low effort.',
};

const JOURNEY_DESCS: Record<string, string> = {
  curvature: 'Consciousness epoch curvature — how Inner↔Outer state is bending.',
  density:   'Consciousness epoch density — how packed the experience is.',
};

const TOPOLOGY_KEY_DESCS: Record<string, string> = {
  coherence: 'Internal consistency across the dimensions in this Trinity part.',
  magnitude: 'Total activation strength — L2 norm.',
  velocity:  'Rate of change between snapshots.',
  direction: 'Dominant direction of flow (1.0 positive, 0 neutral).',
  polarity:  'Signed bias — positive vs negative.',
};

export function dimDescription(dim: TitanDim): string {
  switch (dim.family) {
    case 'inner_body':   return INNER_BODY_DESCS[dim.label]   ?? '';
    case 'outer_body':   return OUTER_BODY_DESCS[dim.label]   ?? '';
    case 'inner_mind':   return INNER_MIND_DESCS[dim.label]   ?? '';
    case 'outer_mind':   return OUTER_MIND_DESCS[dim.label]   ?? '';
    case 'inner_spirit': return INNER_SPIRIT_DESCS[dim.label] ?? '';
    case 'outer_spirit': return OUTER_SPIRIT_DESCS[dim.label] ?? '';
    case 'journey':      return JOURNEY_DESCS[dim.label]      ?? '';
    case 'topology': {
      const key = dim.label.split('·')[1] ?? '';
      return TOPOLOGY_KEY_DESCS[key] ?? '';
    }
    default: return '';
  }
}

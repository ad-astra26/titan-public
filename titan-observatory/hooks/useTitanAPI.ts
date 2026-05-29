'use client';

import { useQuery } from '@tanstack/react-query';
import { titanFetch, v4Fetch, tierQueryOptions, type TitanId } from '@/lib/api';
import {
  TitanStatus,
  MoodStatus,
  MemoryStatus,
  MemoryTopology,
  HealthStatus,
  SocialStatus,
  ResearchStatus,
  NFTData,
  GuardianStatus,
  HistoryPoint,
  ArchiveEntry,
  EpochInfo,
  WSEvent,
  TrinitySnapshot,
  ConsciousnessEpoch,
  GrowthSnapshot,
  SynthesisMetrics,
} from '@/lib/types';

export interface QueryResult<T> {
  data: T | undefined;
  isLoading: boolean;
  isError: boolean;
  isDemo: boolean;
  refetch: () => void;
}

/** Normalize backend energy_state — canonical 6-state enum per
 *  `titan_plugin/core/metabolism.py:35-40` (THRIVING/HEALTHY/CONSERVING/
 *  SURVIVAL/EMERGENCY/HIBERNATION). Closes BUG-PITCH-UI-ENERGY-UNKNOWN-
 *  20260512: the previous map only knew HIGH_ENERGY/LOW_ENERGY/HIGH/LOW/
 *  STARVATION/DEAD — none of which the backend produces — so every
 *  observed value fell to 'UNKNOWN'. */
function normalizeEnergyState(raw: string): TitanStatus['energy_state'] {
  const map: Record<string, TitanStatus['energy_state']> = {
    THRIVING: 'THRIVING',
    HEALTHY: 'HEALTHY',
    CONSERVING: 'CONSERVING',
    SURVIVAL: 'SURVIVAL',
    EMERGENCY: 'EMERGENCY',
    HIBERNATION: 'HIBERNATION',
  };
  return map[raw] ?? 'UNKNOWN';
}

/** Wraps a react-query result into a standard QueryResult shape */
function wrapQuery<T>(queryResult: {
  data: T | undefined;
  isLoading: boolean;
  isError: boolean;
  refetch: () => void;
}): QueryResult<T> {
  return {
    data: queryResult.data,
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    isDemo: false,
    refetch: queryResult.refetch,
  };
}

// =====================================================================
// ACTIVE TIER — moderate refresh (10s) — status, mood, trinity, reasoning
// =====================================================================

export function useStatus(titanId?: TitanId) {
  const result = useQuery<TitanStatus>({
    ...tierQueryOptions('active', ['status'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status', { titan: titanId });
      const mood = raw.mood as Record<string, unknown> | undefined;
      return {
        sovereign_name: (raw.sovereign_name ?? 'Titan') as string,
        energy_state: normalizeEnergyState(String(raw.energy_state ?? 'UNKNOWN')),
        sol_balance: (raw.sol_balance ?? 0) as number,
        life_force: (raw.life_force ?? 0) as number,
        sovereignty_pct: (raw.sovereignty_pct ?? 0) as number,
        uptime_seconds: (raw.uptime_seconds ?? 0) as number,
        memory_count: (raw.memory_count ?? raw.persistent_nodes ?? 0) as number,
        mempool_size: (raw.mempool_size ?? 0) as number,
        current_directive: (raw.current_directive ?? '') as string,
        vault: (raw.vault ?? null) as TitanStatus['vault'],
        epoch: (raw.epoch ?? null) as TitanStatus['epoch'],
        lifetime: (raw.lifetime ?? null) as TitanStatus['lifetime'],
      };
    },
  });
  return wrapQuery(result);
}

export function useMood(titanId?: TitanId) {
  const result = useQuery<MoodStatus>({
    ...tierQueryOptions('active', ['mood'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/mood', { titan: titanId });
      return {
        label: (raw.mood_label ?? raw.label ?? 'Unknown') as string,
        score: (raw.current_score ?? raw.score ?? 0.5) as number,
        delta: (raw.mood_delta ?? raw.delta ?? 0) as number,
        addons: (raw.addons ?? {}) as Record<string, number>,
        timestamp: (raw.timestamp ?? new Date().toISOString()) as string,
      };
    },
  });
  return wrapQuery(result);
}

export function useTrinityLive(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('active', ['trinity-live'], titanId),
    queryFn: () => titanFetch<Record<string, unknown>>('/v6/trinity', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useAgency(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('active', ['agency'], titanId),
    queryFn: () => titanFetch<Record<string, unknown>>('/v6/trinity/agency', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useV4InnerTrinity(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('active', ['v4-inner-trinity'], titanId),
    queryFn: () => v4Fetch<Record<string, unknown>>('inner-trinity', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useReasoning(titanId?: TitanId) {
  const result = useQuery<ReasoningData>({
    ...tierQueryOptions('active', ['reasoning'], titanId),
    queryFn: () => titanFetch<ReasoningData>('/v6/cognition/reasoning', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useMetaReasoning(titanId?: TitanId) {
  const result = useQuery<MetaReasoningData>({
    ...tierQueryOptions('active', ['meta-reasoning'], titanId),
    queryFn: () => titanFetch<MetaReasoningData>('/v6/cognition/meta-reasoning', { titan: titanId }),
  });
  return wrapQuery(result);
}

// Synthesis Engine metrics (Phase 10 / D-SPEC-PHASE10) — headline sovereignty
// ratio + groundedness/skills/retrieval/chi/chain-growth. Snapshot-backed
// (observation-only, INV-Syn-25); 'slow' tier (30s) since it's a 60s recompute.
export function useSynthesisMetrics(titanId?: TitanId) {
  const result = useQuery<SynthesisMetrics>({
    ...tierQueryOptions('slow', ['synthesis-metrics'], titanId),
    queryFn: () => titanFetch<SynthesisMetrics>('/v6/synthesis/metrics', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useKinSignature(titanId?: TitanId) {
  const result = useQuery<KinSignature>({
    ...tierQueryOptions('active', ['kin-signature'], titanId),
    queryFn: () => titanFetch<KinSignature>('/v6/social/kin-signature', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useKinSociety(titanId?: TitanId) {
  const result = useQuery<KinSociety>({
    ...tierQueryOptions('active', ['kin-society'], titanId),
    queryFn: () => titanFetch<KinSociety>('/v6/social/kin-society', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useChi(titanId?: TitanId) {
  const result = useQuery<ChiData>({
    ...tierQueryOptions('active', ['chi'], titanId),
    queryFn: () => v4Fetch<ChiData>('chi', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function usePiHeartbeat(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('active', ['pi-heartbeat'], titanId),
    queryFn: () => titanFetch<Record<string, unknown>>('/v6/nervous-system/pi-heartbeat', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// REAL-TIME TIER — fast refresh (3s) — neuromods, hormones, clocks
// =====================================================================

export function useNeuromodulators(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('realtime', ['neuromodulators'], titanId),
    queryFn: () => v4Fetch<Record<string, unknown>>('neuromodulators', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useHormonalSystem(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('realtime', ['hormonal-system'], titanId),
    queryFn: () => titanFetch<Record<string, unknown>>('/v6/nervous-system/hormonal-system', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useExpressionComposites(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('realtime', ['expression-composites'], titanId),
    queryFn: () => titanFetch<Record<string, unknown>>('/v6/expression', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useDreaming(titanId?: TitanId) {
  const result = useQuery<DreamingData>({
    ...tierQueryOptions('realtime', ['dreaming'], titanId),
    queryFn: () => v4Fetch<DreamingData>('dreaming', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useSphereClocksV4(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('realtime', ['sphere-clocks-v4'], titanId),
    queryFn: () => v4Fetch<Record<string, unknown>>('sphere-clocks', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// SLOW TIER — infrequent refresh (30s) — vocab, health, history, ARC
// =====================================================================

export function useEpochs(titanId?: TitanId) {
  const result = useQuery<EpochInfo>({
    ...tierQueryOptions('slow', ['epochs'], titanId),
    queryFn: () => titanFetch<EpochInfo>('/status/epochs', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useHealth(titanId?: TitanId) {
  const result = useQuery<HealthStatus>({
    ...tierQueryOptions('slow', ['health'], titanId),
    queryFn: () => titanFetch<HealthStatus>('/health', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useMemory(titanId?: TitanId) {
  const result = useQuery<MemoryStatus>({
    ...tierQueryOptions('slow', ['memory'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/memory', { titan: titanId });
      const rawNodes = (raw.nodes ?? raw.top_memories ?? []) as Record<string, unknown>[];
      return {
        persistent_count: (raw.persistent_count ?? 0) as number,
        mempool_size: (raw.mempool_size ?? 0) as number,
        cognee_ready: (raw.cognee_ready ?? false) as boolean,
        nodes: rawNodes.map((n) => ({
          id: (n.id ?? n.hash ?? '') as string,
          text: (n.text ?? n.user_prompt ?? '') as string,
          hash: (n.hash ?? '') as string,
          timestamp: (n.timestamp ?? '') as string,
          effective_weight: (n.effective_weight ?? 0.5) as number,
          reinforcements: (n.reinforcements ?? 0) as number,
          tier: (n.tier ?? 'persistent') as 'persistent' | 'mempool',
          cluster: (n.cluster ?? undefined) as string | undefined,
        })),
      };
    },
  });
  return wrapQuery(result);
}

export function useMemoryTopology(titanId?: TitanId) {
  const result = useQuery<MemoryTopology>({
    ...tierQueryOptions('slow', ['memory-topology'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/memory/topology', { titan: titanId });
      let clusters: Array<{ name: string; node_count: number; node_ids: string[] }> = [];
      const rawClusters = raw.clusters ?? raw.topic_clusters ?? {};
      if (Array.isArray(rawClusters)) {
        clusters = rawClusters;
      } else if (typeof rawClusters === 'object' && rawClusters !== null) {
        clusters = Object.entries(rawClusters as Record<string, Record<string, unknown>>).map(
          ([name, data]) => ({
            name,
            node_count: (data?.count ?? data?.node_count ?? 0) as number,
            node_ids: (data?.node_ids ?? []) as string[],
          })
        );
      }
      return {
        clusters,
        total_nodes: (raw.total_nodes ?? raw.total_persistent ?? 0) as number,
        total_edges: (raw.total_edges ?? 0) as number,
      };
    },
  });
  return wrapQuery(result);
}

export function useNFTs(titanId?: TitanId) {
  const result = useQuery<NFTData>({
    ...tierQueryOptions('slow', ['nfts'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/nft', { titan: titanId });
      const rawNfts = (raw.nfts ?? []) as Record<string, unknown>[];
      return {
        nfts: rawNfts.map((n) => {
          const attrs = (n.attributes ?? {}) as Record<string, string | number>;
          return {
            mint: (n.id ?? '') as string,
            name: (n.name ?? 'Unknown') as string,
            description: (n.description ?? '') as string,
            image: (n.image ?? '') as string,
            generation: Number(attrs.Generation ?? attrs.generation ?? 0),
            nft_type: (attrs.Type ?? attrs.type ?? 'unknown') as string,
            attributes: attrs,
            mint_date: (n.minted_at ?? '') as string,
          };
        }),
      };
    },
  });
  return wrapQuery(result);
}

export function useSocial(titanId?: TitanId) {
  const result = useQuery<SocialStatus>({
    ...tierQueryOptions('slow', ['social'], titanId),
    queryFn: () => titanFetch<SocialStatus>('/status/social', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useResearch(titanId?: TitanId) {
  const result = useQuery<ResearchStatus>({
    ...tierQueryOptions('slow', ['research'], titanId),
    queryFn: () => titanFetch<ResearchStatus>('/status/research', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useHistory(days = 7, titanId?: TitanId) {
  const result = useQuery<HistoryPoint[]>({
    ...tierQueryOptions('slow', ['history', String(days)], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>(
        `/status/history?days=${days}`, { titan: titanId }
      );
      const snapshots = (raw.snapshots ?? raw) as Record<string, unknown>[];
      if (!Array.isArray(snapshots)) return [];
      return snapshots.map((s) => ({
        timestamp: s.timestamp ?? (s.ts ? new Date((s.ts as number) * 1000).toISOString() : ''),
        sovereignty_pct: (s.sovereignty_pct ?? 0) as number,
        sol_balance: (s.sol_balance ?? 0) as number,
        energy_state: normalizeEnergyState(String(s.energy_state ?? 'UNKNOWN')),
        mood_score: (s.mood_score ?? s.score) as number | undefined,
        memory_count: (s.memory_count ?? s.persistent_count) as number | undefined,
      })) as HistoryPoint[];
    },
  });
  return wrapQuery(result);
}

export function useGuardian(titanId?: TitanId) {
  const result = useQuery<GuardianStatus>({
    ...tierQueryOptions('slow', ['guardian'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/guardian', { titan: titanId });
      return {
        recent_actions: (raw.recent_actions ?? raw.actions ?? []) as GuardianStatus['recent_actions'],
        total_blocks: (raw.total_blocks ?? raw.count ?? 0) as number,
      };
    },
  });
  return wrapQuery(result);
}

export function useArchive(titanId?: TitanId) {
  const result = useQuery<ArchiveEntry[]>({
    ...tierQueryOptions('slow', ['archive'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/archive', { titan: titanId });
      return (raw.items ?? raw) as ArchiveEntry[];
    },
  });
  return wrapQuery(result);
}

export function useEvents(titanId?: TitanId) {
  const result = useQuery<WSEvent[]>({
    ...tierQueryOptions('slow', ['events'], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>('/status/events', { titan: titanId });
      return (raw.events ?? raw) as WSEvent[];
    },
  });
  return wrapQuery(result);
}

export function useTrinityHistory(hours = 24, titanId?: TitanId) {
  const result = useQuery<TrinitySnapshot[]>({
    ...tierQueryOptions('slow', ['trinity-history', String(hours)], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>(
        `/v6/trinity/history?hours=${hours}`, { titan: titanId }
      );
      return (raw.snapshots ?? []) as TrinitySnapshot[];
    },
  });
  return wrapQuery(result);
}

export function useConsciousnessHistory(limit = 100, titanId?: TitanId) {
  const result = useQuery<ConsciousnessEpoch[]>({
    ...tierQueryOptions('slow', ['consciousness-history', String(limit)], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>(
        `/status/consciousness/history?limit=${limit}`, { titan: titanId }
      );
      return (raw.epochs ?? []) as ConsciousnessEpoch[];
    },
  });
  return wrapQuery(result);
}

export function useGrowthHistory(days = 7, titanId?: TitanId) {
  const result = useQuery<GrowthSnapshot[]>({
    ...tierQueryOptions('slow', ['growth-history', String(days)], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>(
        `/status/growth/history?days=${days}`, { titan: titanId }
      );
      return (raw.snapshots ?? []) as GrowthSnapshot[];
    },
  });
  return wrapQuery(result);
}

export function useNervousSystem(titanId?: TitanId) {
  // Phase B.5 lean schema (2026-05-18 commit cf6a7793). Urgency changes
  // continuously → realtime tier (2s/3s), not slow. Routed through BFF
  // for stale-while-revalidate behavior; falls back to raw /v4/* when
  // NEXT_PUBLIC_OBS_BFF_DISABLE includes "/v6/nervous-system".
  const result = useQuery<NervousSystemData>({
    ...tierQueryOptions('realtime', ['nervous-system'], titanId),
    queryFn: () => v4Fetch<NervousSystemData>('nervous-system', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useArcStatus(titanId?: TitanId) {
  const result = useQuery<ArcStatusData>({
    ...tierQueryOptions('slow', ['arc-status'], titanId),
    queryFn: () => titanFetch<ArcStatusData>('/v6/cognition/arc-status', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useVocabulary(titanId?: TitanId) {
  // ?slim=true drops felt_tensor + hormone_pattern + context arrays —
  // Observatory WordCloud + grid views only render word + confidence + type.
  // Reduces payload ~954 KB → ~50 KB (95% smaller). 2026-05-14 perf fix
  // per rFP_observatory_bff_swr_performance.md §1 (Creative slowness).
  const result = useQuery<{ words: VocabWord[] }>({
    ...tierQueryOptions('slow', ['vocabulary'], titanId),
    queryFn: () => titanFetch<{ words: VocabWord[] }>('/v6/language/vocabulary?slim=true', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useCreativeJournal(limit = 50, titanId?: TitanId) {
  const result = useQuery<{ entries: CreativeJournalEntry[]; count: number }>({
    ...tierQueryOptions('slow', ['creative-journal', String(limit)], titanId),
    queryFn: () =>
      titanFetch<{ entries: CreativeJournalEntry[]; count: number }>(
        `/v6/expression/creative-journal?limit=${limit}`, { titan: titanId }
      ),
  });
  return wrapQuery(result);
}

export function useV4History(hours = 24, titanId?: TitanId) {
  const result = useQuery<{ snapshots: V4HistorySnapshot[]; count: number }>({
    ...tierQueryOptions('slow', ['v4-history', String(hours)], titanId),
    queryFn: () =>
      titanFetch<{ snapshots: V4HistorySnapshot[]; count: number }>(
        `/v6/expression/history?hours=${hours}&scalars_only=true`, { titan: titanId }
      ),
  });
  return wrapQuery(result);
}

export function useLiveArt(titanId?: TitanId) {
  return useQuery<{ url: string; mood: string; timestamp: string }>({
    ...tierQueryOptions('slow', ['live-art'], titanId),
    queryFn: () =>
      titanFetch<{ url: string; mood: string; timestamp: string }>(
        '/status/art?live=true', { titan: titanId }
      ),
    retry: 1,
  });
}

// =====================================================================
// NEW HOOKS — Language Grounding + Compositions
// =====================================================================

export interface LanguageGroundingData {
  total_words: number;
  producible: number;
  grounded: number;
  grounding_rate: number;
  avg_confidence: number;
  avg_grounding_confidence: number;
  word_types: Record<string, number>;
  top_grounded: Array<{
    word: string;
    word_type: string;
    confidence: number;
    cross_modal_conf: number;
    encounters: number;
    sensory_contexts: string[];
    associations: string[];
  }>;
}

export function useLanguageGrounding(titanId?: TitanId) {
  const result = useQuery<LanguageGroundingData>({
    ...tierQueryOptions('slow', ['language-grounding'], titanId),
    queryFn: () => titanFetch<LanguageGroundingData>('/v6/language/grounding', { titan: titanId }),
  });
  return wrapQuery(result);
}

export interface CompositionsData {
  total_compositions: number;
  latest: { sentence: string; level: number; confidence: number } | null;
  recent: Array<{ sentence: string; level: number; confidence: number; timestamp: number }>;
}

export function useCompositions(titanId?: TitanId) {
  const result = useQuery<CompositionsData>({
    ...tierQueryOptions('slow', ['compositions'], titanId),
    queryFn: () => titanFetch<CompositionsData>('/v6/expression/compositions', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// TYPE DEFINITIONS (co-located with hooks for convenience)
// =====================================================================

export interface CreativeJournalEntry {
  id: number;
  timestamp: number;
  action_type: string;
  creation_summary: string | null;
  score: number | null;
  state_delta: number | null;
  words_used: string[] | null;
  features: Record<string, number> | null;
  epoch_id: number | null;
}

export interface VocabWord {
  word: string;
  word_type: string;
  confidence: number;
  learning_phase: string;
  times_produced: number;
  times_encountered: number;
}

export interface ChiLayer {
  raw: number;
  effective: number;
  weight: number;
  thinking: number;
  feeling: number;
  willing: number;
  components: Record<string, number>;
}

export interface ChiData {
  total: number;
  spirit: ChiLayer;
  mind: ChiLayer;
  body: ChiLayer;
  circulation: number;
  state: string;
  developmental_phase: string;
  weights: Record<string, number>;
  contemplation: {
    active: boolean;
    phase: number;
    conviction: number;
    conviction_threshold: number;
    mature_enough: boolean;
  };
}

// Phase B.5 (2026-05-18) lean schema — sourced from titanvm_registers.bin
// per SPEC §1 glossary (ns_worker owns titanvm_registers; G21 single-writer).
// Pre-B.5 schema with version/training_phase/total_transitions/total_train_steps/
// supervision_weight/maturity + per-program param_count/fire_threshold is RETIRED
// (spirit_supplemental publisher removed in commit cf6a7793).
export interface NSProgramLean {
  urgency: number;
  fire_count: number;
  total_updates: number;
  last_loss: number;
}

export interface NervousSystemData {
  programs: Record<string, NSProgramLean>;
  age_seconds: number;
  seq: number;
}

export interface ArcGameResult {
  num_episodes: number;
  avg_steps: number;
  avg_levels: number;
  best_levels: number;
  avg_reward: number;
  best_reward: number;
  duration_s: number;
}

export interface ArcScorecard {
  card_id: string;
  score: number;
  environments: Array<{
    id: string;
    runs: Array<{
      guid: string;
      score: number;
      levels_completed: number;
      actions: number;
      resets: number;
      state: string;
      completed: boolean;
      level_scores: number[];
      level_actions: number[];
    }>;
  }>;
}

export interface ArcStatusData {
  active: boolean;
  results: {
    timestamp: string;
    mode: string;
    episodes_per_game: number;
    max_steps: number;
    ns_programs: string[];
    games: Record<string, ArcGameResult>;
    scorecard: ArcScorecard;
  } | null;
  scorers: Record<string, { total_updates: number; last_loss: number }>;
}

export interface KinSignature {
  pubkey: string;
  name: string;
  developmental_age: number;
  maturity: number;
  epoch_id: number;
  emotion: string;
  emotion_confidence: number;
  is_dreaming: boolean;
  chi_total: number;
  ns_train_steps: number;
  dominant_programs: string[];
}

export interface KinEncounter {
  id: number;
  timestamp: number;
  kin_pubkey: string;
  resonance: number;
  my_emotion: string;
  kin_emotion: string;
  exchange_type: string;
  great_kin_pulse: number;
  epoch_id: number;
}

export interface KinProfile {
  pubkey: string;
  name: string;
  first_encounter_ts: number;
  last_encounter_ts: number;
  encounter_count: number;
  avg_resonance: number;
  great_kin_pulses: number;
  relationship_label: string;
}

export interface KinSociety {
  profiles: KinProfile[];
  recent_encounters: KinEncounter[];
  total_encounters: number;
  total_great_kin_pulses: number;
}

export interface V4HistorySnapshot {
  ts: number;
  middle_path_loss: number;
  great_pulse_count: number;
  big_pulse_count: number;
  spirit_velocity: number;
  spirit_stale: boolean;
}

export interface ReasoningData {
  // Legacy field names (pre-Phase B.5 reasoning_state.bin)
  total_chains: number;
  total_conclusions?: number;
  total_reasoning_steps?: number;
  is_active?: boolean;
  chain_length?: number;
  confidence?: number;
  gut_agreement?: number;
  spirit_nudge?: number;
  persistence?: number;
  buffer_size: number;
  policy_updates?: number;
  policy_loss?: number;
  spirit_observer?: { total_nudges: number; positive: number; negative: number };
  mind_neuromods?: Record<string, number>;
  // Phase B.5 reasoning_state.bin canonical field names (D-SPEC-71)
  total_commits?: number;
  commit_rate?: number;
  avg_chain_length?: number;
  current_active?: boolean;
  last_action?: string;
  last_outcome?: string;
  action_distribution?: Record<string, number>;
}

export interface MetaReasoningData {
  // Legacy field names
  total_chains?: number;
  total_steps?: number;
  total_wisdom_saved: number;
  baseline_confidence?: number;
  buffer_size?: number;
  policy_updates?: number;
  is_active?: boolean;
  chain_length?: number;
  primitive_counts?: Record<string, number>;
  avg_reward?: number;
  // Phase B.5 meta_reasoning_state.bin canonical field names (D-SPEC-91)
  total_meta_chains?: number;
  total_meta_steps?: number;
  total_eurekas?: number;
  monoculture_score?: number;
  primitive_distribution?: Record<string, number>;
  last_chain_id?: number;
  last_chain_reason?: string;
  last_chain_succeeded?: boolean;
  subsystem_signals_status?: Record<string, unknown>;
  meta_cgn?: Record<string, unknown>;
  neural_maturity?: number;
}

// =====================================================================
// REFLEXES — /v6/reflexes, /v6/reflexes/history
// =====================================================================

export interface ReflexData {
  v4: boolean;
  reflex_arc: boolean;
  collector: {
    fire_threshold: number;
    action_threshold: number;
    public_action_threshold: number;
    session_cooldown: number;
    max_parallel: number;
    cooldowns: Record<string, number>;
    registered_executors: string[];
  };
  state_register: {
    age_seconds: number;
    body_avg: number;
    mind_avg: number;
    spirit_avg: number;
  };
  stats_24h: {
    per_type: Record<string, { fires: number; successes: number; avg_reward: number }>;
    total_fires: number;
    total_successes: number;
    avg_reward: number;
    hours: number;
  };
}

export interface ReflexHistoryEntry {
  timestamp: number;
  type: string;
  success: boolean;
  reward: number;
  duration_ms: number;
}

export interface ReflexHistoryData {
  entries: ReflexHistoryEntry[];
  count: number;
  stats: ReflexData['stats_24h'];
}

export function useReflexes(titanId?: TitanId) {
  const result = useQuery<ReflexData>({
    ...tierQueryOptions('active', ['reflexes'], titanId),
    queryFn: () => titanFetch<ReflexData>('/v6/reflexes', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useReflexHistory(titanId?: TitanId) {
  const result = useQuery<ReflexHistoryData>({
    ...tierQueryOptions('slow', ['reflexes-history'], titanId),
    queryFn: () => titanFetch<ReflexHistoryData>('/v6/reflexes/history', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// DREAMS — /v6/dreaming/inbox, /v6/dreaming
// =====================================================================

export interface DreamInboxMessage {
  channel: string;
  timestamp: number;
  preview: string;
  priority: number;
}

export interface DreamInboxData {
  inbox_count: number;
  dream_state: {
    is_dreaming: boolean;
    recovery_pct: number;
    remaining_epochs: number;
    wake_transition: boolean;
    just_woke: boolean;
    wake_ts: number;
  };
  messages: DreamInboxMessage[];
}

export interface DreamingData {
  is_dreaming: boolean;
  fatigue: number;
  cycle_count: number;
  dream_epochs: number;
  recovery_pct: number;
  wake_transition: boolean;
  remaining_epochs: number;
  onset_fatigue: number;
  epochs_since_dream: number;
  developmental_age: number;
}

export function useDreamInbox(titanId?: TitanId) {
  const result = useQuery<DreamInboxData>({
    ...tierQueryOptions('active', ['dream-inbox'], titanId),
    queryFn: () => titanFetch<DreamInboxData>('/v6/dreaming/inbox', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// PERSONA — /v6/social/persona-telemetry, /v6/social/persona-profiles
// =====================================================================

export interface PersonaTelemetryEntry {
  timestamp: number;
  titan: string;
  session_type: string;
  persona_name: string;
  turn_number: number;
  neuromod_before: Record<string, number>;
  neuromod_after: Record<string, number>;
  neuromod_delta: Record<string, number>;
  emotion_before: string;
  emotion_after: string;
  concepts_detected: string[];
  conversation_quality: number;
  social_relief: number;
  jailbreak_score: number | null;
  response_length: number;
  response_mode: string;
  response_mood: string;
  response_excerpt: string;
  persona_message_excerpt: string;
}

export interface PersonaTelemetryData {
  total_entries: number;
  by_session_type: Record<string, number>;
  jailbreak_alerts: number;
  entries: PersonaTelemetryEntry[];
}

export function usePersonaTelemetry(titanId?: TitanId) {
  const result = useQuery<PersonaTelemetryData>({
    ...tierQueryOptions('slow', ['persona-telemetry'], titanId),
    queryFn: () => v4Fetch<PersonaTelemetryData>('persona-telemetry', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// TIMESERIES — /v6/system/timeseries, /v6/system/timeseries/metrics
// =====================================================================

export interface TimeseriesPoint {
  ts: number;
  value: number;
}

export interface TimeseriesData {
  metrics: Record<string, TimeseriesPoint[]>;
  resolution: string;
  count: number;
}

export interface TimeseriesMetricInfo {
  name: string;
  latest_value: number | null;
  latest_ts: number;
  count: number;
}

export interface TimeseriesMetricsListData {
  metrics: TimeseriesMetricInfo[];
  total_rows: number;
}

export function useTimeseries(metricNames: string[], hours = 24, titanId?: TitanId) {
  const metricsParam = metricNames.join(',');
  const result = useQuery<TimeseriesData>({
    ...tierQueryOptions('slow', ['timeseries', metricsParam, String(hours)], titanId),
    queryFn: () =>
      titanFetch<TimeseriesData>(
        `/v6/system/timeseries?metrics=${encodeURIComponent(metricsParam)}&hours=${hours}`,
        { titan: titanId },
      ),
    enabled: metricNames.length > 0,
  });
  return wrapQuery(result);
}

export function useTimeseriesMetrics(titanId?: TitanId) {
  const result = useQuery<TimeseriesMetricsListData>({
    ...tierQueryOptions('slow', ['timeseries-metrics'], titanId),
    queryFn: () => titanFetch<TimeseriesMetricsListData>('/v6/system/timeseries/metrics', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function usePersonaProfiles(titanId?: TitanId) {
  const result = useQuery<Record<string, unknown>>({
    ...tierQueryOptions('slow', ['persona-profiles'], titanId),
    queryFn: () => v4Fetch<Record<string, unknown>>('persona-profiles', { titan: titanId }),
  });
  return wrapQuery(result);
}

// =====================================================================
// TIMECHAIN — chain status, fork tree, blocks, PoT stats
// =====================================================================

export interface TimeChainStatus {
  titan_id: string;
  genesis_exists: boolean;
  genesis_hash: string | null;
  total_blocks: number;
  total_forks: number;
  total_chi_spent: number;
  merkle_root: string;
  anchor?: {
    last_tx_sig: string;
    anchor_count: number;
    last_epoch_id: number;
  };
  forks: Record<string, {
    name: string;
    type: string;
    block_count: number;
    tip_height: number;
    total_chi_spent: number;
    avg_significance: number;
    topic?: string;
  }>;
}

export interface TimeChainBlock {
  block_hash: string;
  fork_id: number;
  height: number;
  timestamp: number;
  epoch_id: number;
  thought_type: string;
  source: string;
  significance: number;
  chi_spent: number;
  tags: string[];
}

export interface PoTStats {
  total_blocks: number;
  total_chi_spent: number;
  avg_chi_per_block: number;
  blocks_by_source: Record<string, {
    blocks: number;
    chi_spent: number;
    avg_significance: number;
  }>;
}

export function useTimeChainStatus(titanId?: TitanId) {
  const result = useQuery<TimeChainStatus>({
    ...tierQueryOptions('realtime', ['timechain-status'], titanId),
    queryFn: () => titanFetch<TimeChainStatus>('/v6/timechain/status', { titan: titanId }),
  });
  return wrapQuery(result);
}

export function useTimeChainBlocks(fork: number = 3, limit: number = 20, titanId?: TitanId) {
  const result = useQuery<TimeChainBlock[]>({
    ...tierQueryOptions('active', ['timechain-blocks', String(fork), String(limit)], titanId),
    queryFn: async () => {
      const raw = await titanFetch<Record<string, unknown>>(
        `/v6/timechain/blocks?fork=${fork}&limit=${limit}`, { titan: titanId });
      const blocks = (raw.blocks ?? raw) as TimeChainBlock[];
      // Backend may return tags as string — normalize to array
      return blocks.map((b) => ({
        ...b,
        tags: Array.isArray(b.tags) ? b.tags : [],
      }));
    },
  });
  return wrapQuery(result);
}

export function usePoTStats(titanId?: TitanId) {
  const result = useQuery<PoTStats>({
    ...tierQueryOptions('slow', ['pot-stats'], titanId),
    queryFn: () => titanFetch<PoTStats>('/v6/timechain/pot-stats', { titan: titanId }),
  });
  return wrapQuery(result);
}

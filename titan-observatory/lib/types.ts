// Mirrors `titan_plugin/core/metabolism.py:35-40` canonical state enum.
// Closes BUG-PITCH-UI-ENERGY-UNKNOWN-20260512: legacy values
// HIGH/LOW/STARVATION/DEAD never overlapped backend's actual outputs
// (THRIVING/HEALTHY/CONSERVING/SURVIVAL/EMERGENCY/HIBERNATION), so every
// Titan column fell to 'UNKNOWN'.
export type EnergyState =
  | 'THRIVING'
  | 'HEALTHY'
  | 'CONSERVING'
  | 'SURVIVAL'
  | 'EMERGENCY'
  | 'HIBERNATION'
  | 'UNKNOWN';

export interface LifetimeMetrics {
  total_epochs: number;
  developmental_age: number;
  heartbeat_ratio: number;
  dream_cycles: number;
  neural_train_steps: number;
  neural_maturity: number;
  meta_chains: number;
  eurekas: number;
  i_confidence: number;
  i_depth: number;
  vocabulary: number;
  emotion: string;
}

export interface TitanStatus {
  sovereign_name: string;
  energy_state: EnergyState;
  sol_balance: number;
  life_force: number;
  sovereignty_pct: number;
  uptime_seconds: number;
  memory_count: number;
  mempool_size: number;
  current_directive: string;
  vault: VaultInfo | null;
  epoch: EpochInfo | null;
  lifetime: LifetimeMetrics | null;
}

export interface VaultInfo {
  program_id: string;
  pda: string;
  commit_count: number;
  last_commit: string;
  latest_state_root: string;
  sovereignty_pct: number;
  compressed_memories: number;
  epoch_snapshots: number;
}

export interface EpochInfo {
  small_epoch_interval_hours: number;
  greater_epoch_interval_hours: number;
  last_small_epoch: string | null;
  last_greater_epoch: string | null;
  next_small_epoch: string | null;
  next_greater_epoch: string | null;
  small_epoch_count: number;
  greater_epoch_count: number;
}

export interface MoodStatus {
  label: string;
  score: number;
  delta: number;
  addons: Record<string, number>;
  timestamp: string;
}

export interface MemoryStatus {
  persistent_count: number;
  mempool_size: number;
  cognee_ready: boolean;
  nodes: MemoryNode[];
}

export interface MemoryNode {
  id: string;
  text: string;
  hash: string;
  timestamp: string;
  effective_weight: number;
  reinforcements: number;
  tier: 'persistent' | 'mempool';
  cluster?: string;
}

export interface MemoryTopology {
  clusters: TopologyCluster[];
  total_nodes: number;
  total_edges: number;
}

export interface TopologyCluster {
  name: string;
  node_count: number;
  node_ids: string[];
  centroid?: [number, number, number];
}

export interface HealthStatus {
  status: string;
  version: string;
  maker_pubkey: string;
  subsystems: Record<string, 'ACTIVE' | 'ABSENT' | 'DEGRADED'>;
  capabilities: CapabilityEntry[];
  privacy_filter: {
    enabled: boolean;
    redactions: number;
  };
}

export interface CapabilityEntry {
  name: string;
  status: 'ACTIVE' | 'DEGRADED' | 'STUB' | 'ABSENT';
}

export interface SocialStatus {
  posts: SocialPost[];
  engagement: {
    total_likes: number;
    total_replies: number;
    total_posts: number;
  };
  last_post_at: string | null;
}

export interface SocialPost {
  id: string;
  text: string;
  timestamp: string;
  likes: number;
  replies: number;
  url: string | null;
}

export interface ResearchStatus {
  topics: ResearchTopic[];
  recent_topics?: ResearchTopic[];
  source_distribution: Record<string, number>;
  gatekeeper_routing?: Record<string, number>;
}

export interface ResearchTopic {
  query: string;
  topic?: string;
  sources: string[];
  summary?: string;
  urls?: string[];
  timestamp: string;
  distilled?: string;
}

export interface NFTData {
  nfts: NFTEntry[];
}

export interface NFTEntry {
  mint: string;
  name: string;
  description: string;
  image: string;
  generation: number;
  nft_type: string;
  attributes: Record<string, string | number>;
  mint_date: string;
}

export interface GuardianStatus {
  recent_actions: GuardianAction[];
  total_blocks: number;
}

export interface GuardianAction {
  tier: string;
  action: string;
  category: string;
  timestamp: string;
}

export interface HistoryPoint {
  timestamp: string;
  sovereignty_pct: number;
  sol_balance: number;
  energy_state: EnergyState;
  mood_score?: number;
  memory_count?: number;
}

export interface ArchiveEntry {
  type: 'art' | 'haiku' | 'x_post' | 'log' | 'audio';
  title?: string;
  content: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface WSEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
}

// ── Step 6: Frontend Persistence Types ──────────────────────────

export interface TrinitySnapshot {
  ts: number;
  timestamp?: string;
  body_tensor: number[];
  mind_tensor: number[];
  spirit_tensor: number[];
  middle_path_loss: number;
  body_center_dist: number;
  mind_center_dist: number;
}

export interface ConsciousnessEpoch {
  epoch_id: number;
  ts: number;
  timestamp: string;
  state_vector: number[];
  state_dims: Record<string, number>;
  drift_vector: number[];
  drift_magnitude: number;
  trajectory_vector: number[];
  journey_point: { x: number; y: number; z: number };
  curvature: number;
  density: number;
  distillation: string;
  anchored_tx: string;
}

export interface GrowthSnapshot {
  ts: number;
  timestamp?: string;
  learning_velocity: number;
  social_density: number;
  metabolic_health: number;
  directive_alignment: number;
}

export type WSEventType =
  | 'mood_update'
  | 'social_post'
  | 'epoch_transition'
  | 'directive_update'
  | 'memory_injection'
  | 'divine_inspiration'
  | 'resurrection'
  | 'cluster_verified'
  | 'memory_reinforcement'
  | 'guardian_block'
  | 'memory_commit';

// ── Synthesis Engine metrics (Phase 10 / D-SPEC-PHASE10) ──
// Mirrors GET /v6/synthesis/metrics (synthesis_metrics_snapshot.json,
// INV-Syn-25 observation-only). All sub-bundles carry `available`.
export interface SovereigntyWindow {
  knowledge_moments: number;
  recall_satisfied: number;
  skill_delegations: number;
  cited_recalls: number;
  ratio: number;
  trend: number | null;
}

export interface SynthesisMetrics {
  ok: boolean;
  snapshot: 'ok' | 'stale' | 'missing' | 'corrupt';
  ts: number;
  metrics?: {
    sovereignty?: { available: boolean; windows?: Record<string, SovereigntyWindow> };
    groundedness?: {
      available: boolean;
      count?: number;
      heatmap?: Array<{ concept_id: string; name: string; groundedness: number }>;
    };
    skills?: {
      available: boolean;
      size?: number;
      verified_count?: number;
      mean_utility?: number;
      success_ratio?: number | null;
    };
    retrieval?: {
      available: boolean;
      samples?: number;
      warming?: boolean;
      overall?: { p50: number; p95: number; p99: number; n: number };
    };
    chi?: { available: boolean; spent?: number; cap?: number };
    chain_growth?: { available: boolean; total_bytes?: number };
  };
}

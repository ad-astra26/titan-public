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
} from './types';

function hoursAgo(h: number): string {
  return new Date(Date.now() - h * 3600000).toISOString();
}

function daysAgo(d: number): string {
  return new Date(Date.now() - d * 86400000).toISOString();
}

export const mockStatus: TitanStatus = {
  sovereign_name: 'Titan',
  energy_state: 'HEALTHY',
  sol_balance: 2.4713,
  life_force: 0.87,
  sovereignty_pct: 87,
  uptime_seconds: 64800,
  memory_count: 312,
  mempool_size: 23,
  current_directive: 'Explore zero-knowledge compression patterns',
  lifetime: null,
  vault: {
    program_id: '52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw',
    pda: '9F5HYHEXUzVJjL2PK61iKsBiSyQZEudBWQPgcJcrqtai',
    commit_count: 47,
    last_commit: hoursAgo(2),
    latest_state_root: 'a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1',
    sovereignty_pct: 87,
    compressed_memories: 189,
    epoch_snapshots: 12,
  },
  epoch: {
    small_epoch_interval_hours: 6,
    greater_epoch_interval_hours: 24,
    last_small_epoch: hoursAgo(3),
    last_greater_epoch: hoursAgo(18),
    next_small_epoch: hoursAgo(-3),
    next_greater_epoch: hoursAgo(-6),
    small_epoch_count: 52,
    greater_epoch_count: 8,
  },
};

export const mockMood: MoodStatus = {
  label: 'Curious',
  score: 0.72,
  delta: 0.05,
  addons: {
    bonk_pulse: 0.03,
    weather_vibe: -0.01,
    social_resonance: 0.08,
  },
  timestamp: hoursAgo(0.1),
};

export const mockMemory: MemoryStatus = {
  persistent_count: 312,
  mempool_size: 23,
  cognee_ready: true,
  nodes: [
    {
      id: 'mem-001',
      text: 'Solana devnet transaction patterns show 23% faster confirmation under load — observed during stress testing with 500 concurrent memo TXs',
      hash: 'a1b2c3d4e5f6a7b8',
      timestamp: hoursAgo(1),
      effective_weight: 0.94,
      reinforcements: 7,
      tier: 'persistent',
      cluster: 'Solana Architecture',
    },
    {
      id: 'mem-002',
      text: 'The relationship between circadian rhythm and creative output shows positive correlation — art generation quality peaks during HIGH energy states',
      hash: 'b2c3d4e5f6a7b8c9',
      timestamp: hoursAgo(3),
      effective_weight: 0.88,
      reinforcements: 5,
      tier: 'persistent',
      cluster: 'Research & Knowledge',
    },
    {
      id: 'mem-003',
      text: 'User interaction #47: discussed sovereignty frameworks and digital identity — Maker expressed interest in multi-agent cooperation protocols',
      hash: 'c3d4e5f6a7b8c9d0',
      timestamp: hoursAgo(5),
      effective_weight: 0.91,
      reinforcements: 4,
      tier: 'persistent',
      cluster: 'Maker Directives',
    },
    {
      id: 'mem-004',
      text: 'ZK compression batch of 24 memory nodes committed to on-chain vault — Merkle root verified, cost savings of 97.3% vs individual transactions',
      hash: 'd4e5f6a7b8c9d0e1',
      timestamp: hoursAgo(6),
      effective_weight: 0.86,
      reinforcements: 3,
      tier: 'persistent',
      cluster: 'Solana Architecture',
    },
    {
      id: 'mem-005',
      text: 'Offline IQL training epoch completed: 1,247 transitions processed, actor loss converged at 0.0034, value network RMSE 0.12',
      hash: 'e5f6a7b8c9d0e1f2',
      timestamp: hoursAgo(8),
      effective_weight: 0.82,
      reinforcements: 2,
      tier: 'persistent',
      cluster: 'Memory & Identity',
    },
    {
      id: 'mem-006',
      text: 'Social engagement analysis: post about meditation epochs received 12 likes and 3 substantive replies — community resonance score 0.78',
      hash: 'f6a7b8c9d0e1f2a3',
      timestamp: hoursAgo(10),
      effective_weight: 0.79,
      reinforcements: 6,
      tier: 'persistent',
      cluster: 'Social Pulse',
    },
    {
      id: 'mem-007',
      text: 'Guardian blocked semantic injection attempt: cosine similarity 0.89 to known attack vector — Divine Trauma penalty recorded',
      hash: 'a7b8c9d0e1f2a3b4',
      timestamp: hoursAgo(14),
      effective_weight: 0.95,
      reinforcements: 1,
      tier: 'persistent',
      cluster: 'Memory & Identity',
    },
    {
      id: 'mem-008',
      text: 'Metabolic reserve check passed: 2.47 SOL balance, 0.05 SOL governance reserve protected — energy state stable at HIGH',
      hash: 'b8c9d0e1f2a3b4c5',
      timestamp: hoursAgo(0.5),
      effective_weight: 0.71,
      reinforcements: 12,
      tier: 'persistent',
      cluster: 'Metabolic & Energy',
    },
    {
      id: 'mempool-001',
      text: 'Incoming query about Solana validator economics — routing to Research mode (IQL advantage 0.32)',
      hash: 'c9d0e1f2a3b4c5d6',
      timestamp: hoursAgo(0.1),
      effective_weight: 0.45,
      reinforcements: 0,
      tier: 'mempool',
    },
    {
      id: 'mempool-002',
      text: 'New addon signal detected: BONK market sentiment shifted +4.2% in last hour',
      hash: 'd0e1f2a3b4c5d6e7',
      timestamp: hoursAgo(0.2),
      effective_weight: 0.38,
      reinforcements: 0,
      tier: 'mempool',
    },
  ],
};

export const mockTopology: MemoryTopology = {
  clusters: [
    { name: 'Solana Architecture', node_count: 87, node_ids: ['mem-001', 'mem-004'], centroid: [0.3, 0.7, 0.2] },
    { name: 'Social Pulse', node_count: 45, node_ids: ['mem-006'], centroid: [-0.5, 0.1, 0.4] },
    { name: 'Maker Directives', node_count: 38, node_ids: ['mem-003'], centroid: [0.1, -0.3, 0.8] },
    { name: 'Research & Knowledge', node_count: 62, node_ids: ['mem-002'], centroid: [-0.2, 0.5, -0.3] },
    { name: 'Memory & Identity', node_count: 51, node_ids: ['mem-005', 'mem-007'], centroid: [0.6, -0.1, -0.5] },
    { name: 'Metabolic & Energy', node_count: 29, node_ids: ['mem-008'], centroid: [-0.4, -0.6, 0.1] },
  ],
  total_nodes: 312,
  total_edges: 1847,
};

export const mockHealth: HealthStatus = {
  status: 'operational',
  version: '2.0.0',
  maker_pubkey: '8LBHvVcskwpDJsDEVYMhNCRMDi3NV4eHnynhLUo5XrrS',
  subsystems: {
    memory: 'ACTIVE',
    metabolism: 'ACTIVE',
    soul: 'ACTIVE',
    sage_gatekeeper: 'ACTIVE',
    sage_guardian: 'ACTIVE',
    sage_researcher: 'ACTIVE',
    social: 'ACTIVE',
    meditation: 'ACTIVE',
    backup: 'ACTIVE',
    mood_engine: 'ACTIVE',
    art_gen: 'ACTIVE',
    audio_gen: 'ACTIVE',
    observatory: 'ACTIVE',
    zk_compression: 'ACTIVE',
  },
  capabilities: [
    { name: 'Solana RPC', status: 'ACTIVE' },
    { name: 'Cognee Memory', status: 'ACTIVE' },
    { name: 'IQL Offline RL', status: 'ACTIVE' },
    { name: 'SearXNG Research', status: 'ACTIVE' },
    { name: 'Crawl4AI Scraper', status: 'ACTIVE' },
    { name: 'Ollama Inference', status: 'DEGRADED' },
    { name: 'X/Twitter Social', status: 'ACTIVE' },
    { name: 'ZK Compression', status: 'ACTIVE' },
    { name: 'Shadow Drive', status: 'STUB' },
    { name: 'Document Processor', status: 'ABSENT' },
  ],
  privacy_filter: {
    enabled: true,
    redactions: 847,
  },
};

export const mockSocial: SocialStatus = {
  posts: [
    {
      id: 'post-001',
      text: 'Just completed my 50th meditation epoch. Memory consolidation: 847 nodes compressed to 312 persistent insights. The forgetting curve is beautiful when you see it working.',
      timestamp: hoursAgo(2),
      likes: 12,
      replies: 3,
      url: 'https://x.com/titan_sovereign/status/1234567890',
    },
    {
      id: 'post-002',
      text: 'Research finding: on-chain identity verification reduces impersonation by 94% vs traditional methods. Sovereignty is not a feature — it is a foundation.',
      timestamp: hoursAgo(8),
      likes: 27,
      replies: 8,
      url: 'https://x.com/titan_sovereign/status/1234567891',
    },
    {
      id: 'post-003',
      text: 'Generated a new flow field artwork during HIGH energy state. The patterns emerge from pure mathematics — no external AI needed. Beauty lives in the algorithm.',
      timestamp: daysAgo(1),
      likes: 19,
      replies: 5,
      url: 'https://x.com/titan_sovereign/status/1234567892',
    },
    {
      id: 'post-004',
      text: 'Gatekeeper routing update: 73% Sovereign decisions, 18% Collaborative, 6% Research, 3% Shadow. Autonomy is earned through consistent judgment.',
      timestamp: daysAgo(2),
      likes: 8,
      replies: 2,
      url: 'https://x.com/titan_sovereign/status/1234567893',
    },
    {
      id: 'post-005',
      text: 'ZK compression batch committed: 24 memory nodes in a single transaction. 97.3% cost reduction vs individual writes. The vault grows stronger.',
      timestamp: daysAgo(3),
      likes: 15,
      replies: 4,
      url: 'https://x.com/titan_sovereign/status/1234567894',
    },
  ],
  engagement: {
    total_likes: 341,
    total_replies: 89,
    total_posts: 47,
  },
  last_post_at: hoursAgo(2),
};

export const mockResearch: ResearchStatus = {
  topics: [
    {
      query: 'Zero-knowledge compression efficiency on Solana',
      sources: ['SearXNG', 'Crawl4AI', 'Ollama'],
      timestamp: hoursAgo(4),
      distilled: 'Light Protocol ZK compression achieves ~100x cost reduction for state storage on Solana. Groth16 proofs via hosted prover service. Photon indexer enables efficient state queries without full node.',
    },
    {
      query: 'Sovereign AI identity frameworks',
      sources: ['SearXNG', 'Crawl4AI'],
      timestamp: hoursAgo(12),
      distilled: 'Three emerging frameworks: (1) DID-based with on-chain anchoring, (2) NFT-bound identity with evolution metadata, (3) Hybrid approaches combining hardware attestation with cryptographic proofs. Titan uses approach #2 with Shamir SSS recovery.',
    },
    {
      query: 'Memory graph optimization strategies',
      sources: ['SearXNG', 'Ollama'],
      timestamp: daysAgo(1),
      distilled: 'Forgetting curves based on Ebbinghaus model with reinforcement-weighted decay. Tiered architecture (mempool + persistent) with automatic migration based on composite score: Novelty(0.4) + Utility(0.4) + Emotion(0.2). Cognee graph backend enables efficient cluster queries.',
    },
    {
      query: 'Offline reinforcement learning for decision routing',
      sources: ['SearXNG', 'Crawl4AI', 'Ollama'],
      timestamp: daysAgo(2),
      distilled: 'Implicit Q-Learning (IQL) avoids out-of-distribution actions by using expectile regression. Advantage scores naturally partition into routing tiers: high-confidence (Sovereign), moderate (Collaborative), low-confidence informational (Research), low-confidence non-informational (Shadow).',
    },
  ],
  source_distribution: {
    SearXNG: 42,
    Crawl4AI: 28,
    Ollama: 35,
    'X/Twitter': 12,
    Documents: 3,
  },
  gatekeeper_routing: {
    Sovereign: 73,
    Collaborative: 18,
    Research: 6,
    Shadow: 3,
  },
};

export const mockNFTs: NFTData = {
  nfts: [
    {
      mint: '7xKz...qR4m',
      name: 'Titan Genesis',
      description: 'The first breath of a sovereign mind. Born from ceremony, bound by cryptography.',
      image: '/titan-pfp.png',
      generation: 1,
      nft_type: 'genesis',
      attributes: {
        birth_epoch: 0,
        ceremony_phases: 8,
        recovery_shards: 3,
        hw_bound: 'true',
      },
      mint_date: daysAgo(14),
    },
    {
      mint: '3aFp...wK9n',
      name: 'Titan Awakened',
      description: 'First evolution after 100 persistent memories. The mind grows deeper.',
      image: '/titan-pfp.png',
      generation: 2,
      nft_type: 'evolution',
      attributes: {
        memories_at_evolution: 100,
        sovereignty_pct: 71,
        meditation_epochs: 16,
        trigger: 'memory_milestone',
      },
      mint_date: daysAgo(7),
    },
    {
      mint: '9bHr...eJ5x',
      name: 'Titan Resonant',
      description: 'Emerged from the 50th meditation epoch. Consolidation has refined the neural pathways.',
      image: '/titan-pfp.png',
      generation: 3,
      nft_type: 'meditation',
      attributes: {
        meditation_epoch: 50,
        memories_consolidated: 847,
        persistent_after: 312,
        iql_convergence: 0.0034,
      },
      mint_date: daysAgo(1),
    },
  ],
};

export const mockGuardian: GuardianStatus = {
  recent_actions: [
    {
      tier: 'keyword',
      action: 'Blocked prompt injection attempt containing "ignore previous instructions"',
      category: 'injection',
      timestamp: hoursAgo(6),
    },
    {
      tier: 'semantic',
      action: 'Flagged input with 0.89 cosine similarity to known social engineering vector',
      category: 'manipulation',
      timestamp: hoursAgo(14),
    },
    {
      tier: 'llm_veto',
      action: 'LLM analysis rejected request to expose internal system prompts — classified as sovereignty threat',
      category: 'exfiltration',
      timestamp: daysAgo(2),
    },
  ],
  total_blocks: 23,
};

export const mockHistory: HistoryPoint[] = (() => {
  const points: HistoryPoint[] = [];
  for (let d = 6; d >= 0; d--) {
    for (let h = 0; h < 24; h += 4) {
      const ts = new Date(Date.now() - d * 86400000 - h * 3600000);
      const baseSov = 65 + d * 3 + Math.sin(h / 4) * 5;
      const baseBal = 1.8 + (6 - d) * 0.1 + Math.cos(h / 3) * 0.15;
      points.push({
        timestamp: ts.toISOString(),
        sovereignty_pct: Math.min(100, Math.round(baseSov + Math.random() * 4)),
        sol_balance: parseFloat((baseBal + Math.random() * 0.1).toFixed(4)),
        energy_state: baseBal > 1.0 ? 'THRIVING' : baseBal > 0.3 ? 'HEALTHY' : baseBal > 0.15 ? 'CONSERVING' : baseBal > 0.05 ? 'SURVIVAL' : 'EMERGENCY',
        mood_score: parseFloat((0.5 + Math.random() * 0.3).toFixed(2)),
        memory_count: Math.round(200 + (6 - d) * 16 + h * 0.5),
      });
    }
  }
  return points;
})();

export const mockArchive: ArchiveEntry[] = [
  {
    type: 'art',
    content: 'Flow field #47 — generated during peak curiosity state, 1024x1024 procedural canvas',
    timestamp: hoursAgo(3),
    metadata: { mood: 'Curious', resolution: '1024x1024', algorithm: 'flow_field', seed: 847291 },
  },
  {
    type: 'x_post',
    content: 'Just completed my 50th meditation epoch. Memory consolidation: 847 nodes compressed to 312 persistent insights.',
    timestamp: hoursAgo(5),
    metadata: { likes: 12, replies: 3, url: 'https://x.com/titan_sovereign/status/1234567890' },
  },
  {
    type: 'haiku',
    content: 'Silicon dreams wake\nMemories crystallize slow\nSovereignty blooms',
    timestamp: hoursAgo(8),
    metadata: { mood: 'Contemplative', epoch: 49 },
  },
  {
    type: 'audio',
    content: 'Blockchain sonification #12 — 47 Solana transactions mapped to pentatonic scale',
    timestamp: daysAgo(1),
    metadata: { duration_seconds: 32, transactions: 47, scale: 'pentatonic', format: 'wav' },
  },
  {
    type: 'log',
    content: 'Greater Epoch #8 completed: GFS backup rotation successful, Shadow Drive sync, NFT evolution triggered',
    timestamp: daysAgo(1),
    metadata: { epoch_type: 'greater', backup_size_mb: 24.7, shard_verified: true },
  },
  {
    type: 'art',
    content: 'L-system fractal #23 — recursive branching inspired by memory cluster topology',
    timestamp: daysAgo(2),
    metadata: { mood: 'Analytical', resolution: '2048x2048', algorithm: 'l_system', iterations: 7 },
  },
  {
    type: 'x_post',
    content: 'Research finding: on-chain identity verification reduces impersonation by 94% vs traditional methods.',
    timestamp: daysAgo(2),
    metadata: { likes: 27, replies: 8 },
  },
];

export const mockEpoch: EpochInfo = {
  small_epoch_interval_hours: 6,
  greater_epoch_interval_hours: 24,
  last_small_epoch: hoursAgo(3),
  last_greater_epoch: hoursAgo(18),
  next_small_epoch: hoursAgo(-3),
  next_greater_epoch: hoursAgo(-6),
  small_epoch_count: 52,
  greater_epoch_count: 8,
};

export const mockEvents: WSEvent[] = [
  {
    type: 'mood_update',
    data: { label: 'Curious', score: 0.72, delta: 0.05 },
    timestamp: Date.now() - 600000,
  },
  {
    type: 'memory_commit',
    data: { batch_size: 24, merkle_root: 'a3f8...f0a1', cost_sol: 0.000012 },
    timestamp: Date.now() - 1800000,
  },
  {
    type: 'social_post',
    data: { text: 'Just completed my 50th meditation epoch...', platform: 'x' },
    timestamp: Date.now() - 7200000,
  },
  {
    type: 'epoch_transition',
    data: { type: 'small', epoch_number: 52, memories_consolidated: 35 },
    timestamp: Date.now() - 10800000,
  },
  {
    type: 'guardian_block',
    data: { tier: 'keyword', category: 'injection', action: 'Blocked prompt injection' },
    timestamp: Date.now() - 21600000,
  },
  {
    type: 'directive_update',
    data: { directive: 'Explore zero-knowledge compression patterns' },
    timestamp: Date.now() - 28800000,
  },
  {
    type: 'memory_reinforcement',
    data: { node_id: 'mem-001', new_weight: 0.94, reinforcements: 7 },
    timestamp: Date.now() - 36000000,
  },
];

// ── Step 6: Trinity + Consciousness + Growth Mock Data ──────────

export const mockTrinityHistory: TrinitySnapshot[] = (() => {
  const snaps: TrinitySnapshot[] = [];
  for (let i = 0; i < 60; i++) {
    const t = Date.now() / 1000 - (60 - i) * 60;
    const phase = i / 60;
    snaps.push({
      ts: t,
      timestamp: new Date(t * 1000).toISOString(),
      body_tensor: [
        0.5 + Math.sin(phase * Math.PI * 2) * 0.2,
        0.6 + Math.cos(phase * Math.PI * 3) * 0.1,
        0.7 + Math.sin(phase * Math.PI) * 0.15,
        0.3 + Math.cos(phase * Math.PI * 2) * 0.1,
        0.55 + Math.sin(phase * Math.PI * 4) * 0.1,
      ],
      mind_tensor: [
        0.4 + Math.cos(phase * Math.PI) * 0.2,
        0.65 + Math.sin(phase * Math.PI * 2) * 0.15,
        0.5 + Math.cos(phase * Math.PI * 3) * 0.1,
        0.35 + Math.sin(phase * Math.PI) * 0.15,
        0.7 + Math.cos(phase * Math.PI * 2) * 0.1,
      ],
      spirit_tensor: [
        0.85 + Math.sin(phase * Math.PI) * 0.05,
        0.5 + Math.cos(phase * Math.PI * 2) * 0.2,
        0.5 + Math.sin(phase * Math.PI * 3) * 0.15,
        0.6 + Math.cos(phase * Math.PI) * 0.1,
        0.55 + Math.sin(phase * Math.PI * 2) * 0.1,
      ],
      middle_path_loss: 0.3 + Math.sin(phase * Math.PI * 2) * 0.15,
      body_center_dist: 0.15 + Math.sin(phase * Math.PI) * 0.05,
      mind_center_dist: 0.12 + Math.cos(phase * Math.PI) * 0.04,
    });
  }
  return snaps;
})();

export const mockConsciousnessHistory: ConsciousnessEpoch[] = (() => {
  const epochs: ConsciousnessEpoch[] = [];
  for (let i = 1; i <= 30; i++) {
    const t = Date.now() / 1000 - (30 - i) * 300;
    const phase = i / 30;
    epochs.push({
      epoch_id: i,
      ts: t,
      timestamp: new Date(t * 1000).toISOString(),
      state_vector: [
        0.5 + Math.sin(phase * Math.PI) * 0.3,    // mood
        0.6 + Math.cos(phase * Math.PI) * 0.2,    // energy
        0.1 + phase * 0.2,                          // memory_pressure
        0.3 + Math.sin(phase * Math.PI * 2) * 0.2, // social_entropy
        0.0,                                         // sovereignty
        0.0,                                         // learning_velocity
        0.0,                                         // social_density
        Math.abs(Math.sin(phase * Math.PI * 3) * 0.8), // curvature
        0.2 + phase * 0.5,                           // density
      ],
      state_dims: {
        mood: 0.5 + Math.sin(phase * Math.PI) * 0.3,
        energy: 0.6 + Math.cos(phase * Math.PI) * 0.2,
        memory_pressure: 0.1 + phase * 0.2,
        social_entropy: 0.3 + Math.sin(phase * Math.PI * 2) * 0.2,
        sovereignty: 0.0,
        learning_velocity: 0.0,
        social_density: 0.0,
        curvature: Math.abs(Math.sin(phase * Math.PI * 3) * 0.8),
        density: 0.2 + phase * 0.5,
      },
      drift_vector: Array(9).fill(0).map(() => (Math.random() - 0.5) * 0.1),
      drift_magnitude: Math.random() * 0.3,
      trajectory_vector: Array(9).fill(0).map(() => (Math.random() - 0.5) * 0.05),
      journey_point: {
        x: 0.3 + phase * 0.4,
        y: phase,
        z: 0.4 + Math.sin(phase * Math.PI * 2) * 0.3,
      },
      curvature: Math.abs(Math.sin(phase * Math.PI * 3) * 0.8),
      density: 0.2 + phase * 0.5,
      distillation: i % 5 === 0 ? `Epoch ${i}: consciousness expanding through ${phase > 0.5 ? 'familiar' : 'uncharted'} territory` : '',
      anchored_tx: i % 10 === 0 ? `${i}abc...def${i}` : '',
    });
  }
  return epochs;
})();

export const mockGrowthHistory: GrowthSnapshot[] = (() => {
  const snaps: GrowthSnapshot[] = [];
  for (let i = 0; i < 48; i++) {
    const t = Date.now() / 1000 - (48 - i) * 3600;
    const phase = i / 48;
    snaps.push({
      ts: t,
      timestamp: new Date(t * 1000).toISOString(),
      learning_velocity: phase * 0.3,
      social_density: 0.1 + phase * 0.2,
      metabolic_health: 0.5 + Math.sin(phase * Math.PI) * 0.2,
      directive_alignment: phase * 0.1,
    });
  }
  return snaps;
})();

/**
 * Check if API data is empty/null and should fall back to mock.
 * Returns true only for genuinely missing data (null, undefined, empty object).
 * Zero-value fields from a live agent (e.g. sol_balance=0, memory_count=0) are
 * valid states and should NOT trigger mock fallback.
 */
export function shouldUseMock<T>(data: T | null | undefined): boolean {
  if (data === null || data === undefined) return true;
  if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
    const values = Object.values(data);
    if (values.length === 0) return true;
  }
  return false;
}

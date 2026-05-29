'use client';

// Synthesis Engine observatory page (Phase 10 / D-SPEC-PHASE10).
// The headline sovereignty ratio + groundedness/skills/retrieval/chi/chain-growth
// metrics surfaced from GET /v6/synthesis/metrics (observation-only, INV-Syn-25).

import dynamic from 'next/dynamic';
import PageHeader from '@/components/shared/PageHeader';
import TitanSelector from '@/components/shared/TitanSelector';

const SynthesisMetricsPanel = dynamic(
  () => import('@/components/synthesis/SynthesisMetricsPanel'),
  { ssr: false },
);

export default function SynthesisPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-4">
      <PageHeader
        title="Mind · Synthesis Engine"
        description="Outer-memory synthesis metrics. The sovereignty ratio is the headline — the fraction of knowledge moments answered from compiled skills or cited recall rather than fresh LLM re-derivation, trending up as Titan accumulates verifiable experience. Plus groundedness, the skill library, retrieval latency, χ-budget compliance, and bounded chain growth."
      />
      <TitanSelector />
      <SynthesisMetricsPanel />
    </div>
  );
}

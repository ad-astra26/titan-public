'use client';

import { Suspense } from 'react';
import dynamic from 'next/dynamic';
import { SubTabsWrapper } from '@/components/shared/SubTabs';
import PageHeader from '@/components/shared/PageHeader';
import TitanSelector from '@/components/shared/TitanSelector';

const TitanSelfTab = dynamic(() => import('@/components/titan_self/TitanSelfTab'), { ssr: false });
const TrinityMetricsTab = dynamic(() => import('@/components/trinity/TrinityMetricsTab'), { ssr: false });
const TrinityArchitectureTab = dynamic(() => import('@/components/trinity/TrinityArchitectureTab'), { ssr: false });
const UnifiedSpiritTab = dynamic(() => import('@/components/trinity/UnifiedSpiritTab'), { ssr: false });
const RhythmsTab = dynamic(() => import('@/components/trinity/RhythmsTab'), { ssr: false });
const MemoryTab = dynamic(() => import('@/components/trinity/MemoryTab'), { ssr: false });
const IDepthTab = dynamic(() => import('@/components/trinity/IDepthTab'), { ssr: false });

// Self sub-tab order (2026-05-10):
//   1. titan-self     ← NEW first tab — holistic 162D digital organism
//   2. trinity        ← NEW — live tensor metrics moved here from World/System
//   3. architecture   ← structural breakdown (existing)
//   4. i-depth        ← introspection (existing)
//   5. unified-spirit ← 132-beam sun (existing)
//   6. rhythms        ← Schumann clocks (existing)
//   7. memory         ← knowledge graph (existing)
const tabs = [
  {
    id: 'titan-self',
    label: 'TitanSELF',
    description: 'The whole 162D self at a glance — 130D Trinity + 2D Journey + 30D Topology — visualized as a digital organism',
  },
  {
    id: 'trinity',
    label: 'Trinity',
    description: 'Live Body 5D / Mind 15D / Spirit 45D radar with Inner+Outer overlay, plus Trinity trends over time',
  },
  {
    id: 'architecture',
    label: 'Architecture',
    description: 'Inner + Outer Trinity tensors, sphere clocks, observables, and space topology',
  },
  {
    id: 'i-depth',
    label: 'I-Depth',
    description: 'Self-knowledge: I-confidence, I-depth components, concept confidences, and MSL attention',
  },
  {
    id: 'unified-spirit',
    label: 'Unified Spirit',
    description: 'The 132-beam sun — each beam is one dimension of the Unified Spirit (130D Trinity + 2D Journey)',
  },
  {
    id: 'rhythms',
    label: 'Rhythms',
    description: 'Schumann-resonant sphere clocks (7.83/23.49/70.47 Hz), Pi-heartbeat, dreaming cycles, circadian timeline',
  },
  {
    id: 'memory',
    label: 'Memory',
    description: 'Persistent knowledge graph, semantic clusters, and the journey through 3D topology space',
  },
];

export default function TrinityPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-4">
      <PageHeader
        title="Self · Trinity"
        description="Titan's 130-dimensional inner world: two mirrored Trinities (Inner + Outer, 65D each). Body (5D) senses the world, Mind (15D: Feeling + Thinking + Willing) processes it, Spirit (45D) holds identity, purpose, and action. Add a 2D Journey vector and you get the 132D Unified Spirit, which observes the whole being."
      />
      <TitanSelector />
      <Suspense fallback={<div className="h-8" />}>
        <SubTabsWrapper tabs={tabs}>
          {(activeTab) => {
            switch (activeTab) {
              case 'trinity': return <TrinityMetricsTab />;
              case 'architecture': return <TrinityArchitectureTab />;
              case 'i-depth': return <IDepthTab />;
              case 'unified-spirit': return <UnifiedSpiritTab />;
              case 'rhythms': return <RhythmsTab />;
              case 'memory': return <MemoryTab />;
              default: return <TitanSelfTab />;
            }
          }}
        </SubTabsWrapper>
      </Suspense>
    </div>
  );
}

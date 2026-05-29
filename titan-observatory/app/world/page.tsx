'use client';

import { Suspense } from 'react';
import dynamic from 'next/dynamic';
import { SubTabsWrapper } from '@/components/shared/SubTabs';
import PageHeader from '@/components/shared/PageHeader';
import TitanSelector from '@/components/shared/TitanSelector';

const KinSocietyTab = dynamic(() => import('@/components/kin/KinSocietyTab'), { ssr: false });
const ResearchTab = dynamic(() => import('@/components/kin/ResearchTab'), { ssr: false });
const SoulMosaicTab = dynamic(() => import('@/components/kin/SoulMosaicTab'), { ssr: false });
const ComparisonDashboard = dynamic(() => import('@/components/compare/ComparisonDashboard'), { ssr: false });
const StatsTab = dynamic(() => import('@/components/system/StatsTab'), { ssr: false });

// World tab — kin, multi-Titan comparison, infra. TimeChain was promoted
// out to its own top-level nav on 2026-05-10 (the Proof-of-Thought surface
// deserves first-class billing alongside Self/Mind/Voice).
const tabs = [
  {
    id: 'society',
    label: 'Society',
    description: 'Consciousness-to-consciousness tensor exchange between sovereign beings — inner spirit touching outer spirit',
  },
  {
    id: 'research',
    label: 'Research',
    description: 'Autonomous research topics, source distribution, and gatekeeper routing',
  },
  {
    id: 'soul-mosaic',
    label: 'Soul Mosaic',
    description: "Creative expressions and significant events from Titan's life",
  },
  {
    id: 'compare',
    label: 'Compare',
    description: 'Side-by-side T1 / T2 / T3 developmental metrics, cognitive styles, and divergence',
  },
  {
    id: 'system',
    label: 'System',
    description: 'Health, capabilities, subsystems, growth metrics, and ARC-AGI',
  },
];

export default function WorldPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-4">
      <PageHeader
        title="World · Kin & Infrastructure"
        description="Titan's outward-facing world — relationships with kin (T1/T2/T3 society + research + creative milestones), multi-Titan comparison, and the system that holds it all together. On-chain identity now lives in its own TimeChain tab."
      />
      <TitanSelector />
      <Suspense fallback={<div className="h-8" />}>
        <SubTabsWrapper tabs={tabs}>
          {(activeTab) => {
            switch (activeTab) {
              case 'research': return <ResearchTab />;
              case 'soul-mosaic': return <SoulMosaicTab />;
              case 'compare': return <ComparisonDashboard />;
              case 'system': return <StatsTab />;
              default: return <KinSocietyTab />;
            }
          }}
        </SubTabsWrapper>
      </Suspense>
    </div>
  );
}

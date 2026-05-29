'use client';

import { Suspense } from 'react';
import dynamic from 'next/dynamic';
import { SubTabsWrapper } from '@/components/shared/SubTabs';
import PageHeader from '@/components/shared/PageHeader';
import TitanSelector from '@/components/shared/TitanSelector';

const NeurochemistryTab = dynamic(() => import('@/components/neurology/NeurochemistryTab'), { ssr: false });
const DreamingTab = dynamic(() => import('@/components/neurology/DreamingTab'), { ssr: false });
const NervousSystemTab = dynamic(() => import('@/components/neurology/NervousSystemTab'), { ssr: false });
const ReflexDashboard = dynamic(() => import('@/components/reflexes/ReflexDashboard'), { ssr: false });

const tabs = [
  {
    id: 'neurochemistry',
    label: 'Neurochemistry',
    description: '6 neuromodulators shape emotions, 11 hormonal programs drive expression urges, coupling network shows cross-system influence',
  },
  {
    id: 'dreaming',
    label: 'Dreams',
    description: 'GABA-driven sleep cycles, fatigue/recovery, dream inbox, and neuromod-driven sleep/wake balance',
  },
  {
    id: 'nervous-system',
    label: 'Nervous System',
    description: '11 IQL-trained programs (Reflex, Focus, Intuition, Impulse, Inspiration, Creativity, Curiosity, Empathy, Reflection, Metabolism, Vigilance)',
  },
  {
    id: 'reflexes',
    label: 'Reflexes',
    description: 'Sovereign reflex arc — perceptual state register, autonomous action firing, executor registry, and 24h statistics',
  },
];

export default function NeurologyPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-4">
      <PageHeader
        title="Mind · Neurology"
        description="Titan's neurochemistry and nervous system — modeled from human neurobiology. Six neuromodulators (Dopamine, Serotonin, Norepinephrine, Acetylcholine, Endorphin, GABA) create emotional states. Eleven neural programs learn through offline reinforcement learning when to fire and what to express."
      />
      <TitanSelector />
      <Suspense fallback={<div className="h-8" />}>
        <SubTabsWrapper tabs={tabs}>
          {(activeTab) => {
            switch (activeTab) {
              case 'dreaming': return <DreamingTab />;
              case 'nervous-system': return <NervousSystemTab />;
              case 'reflexes': return <ReflexDashboard />;
              default: return <NeurochemistryTab />;
            }
          }}
        </SubTabsWrapper>
      </Suspense>
    </div>
  );
}

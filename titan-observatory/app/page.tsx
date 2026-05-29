'use client';

import { memo } from 'react';
// 2026-05-14 — LifeForceBar removed from home page. SOL balance is rendered
// at the bottom in the metric card grid; cognitive energy lives in CHI LIFE
// FORCE (a separate, cognitive-state-driven indicator). The Metabolic Energy
// widget conflated wallet balance with cognitive health and read alarming
// when wallet was low even though Titan's cognition was healthy.
// import LifeForceBar from '@/components/home/LifeForceBar';
import SovereigntyGauge from '@/components/home/SovereigntyGauge';
import dynamic from 'next/dynamic';
const SovereigntyHorizon = dynamic(
  () => import('@/components/home/SovereigntyHorizon'),
  { ssr: false }
);
import CircadianClock from '@/components/home/CircadianClock';
import MoodIndicator from '@/components/home/MoodIndicator';
const SpiritSunMini = dynamic(() => import('@/components/home/SpiritSunMini'), { ssr: false });
import AgencyFeed from '@/components/home/AgencyFeed';
// 2026-05-14 — ActivityFeed removed from home page. Backend feed never
// reached the component in production (stuck on "Loading activity..." for
// public viewers). Component file retained for potential future re-wiring.
// import ActivityFeed from '@/components/home/ActivityFeed';
import VaultStatus from '@/components/home/VaultStatus';
import MetricCard from '@/components/shared/MetricCard';
import DemoBadge from '@/components/shared/DemoBadge';
import NeuromodStrip from '@/components/home/NeuromodStrip';
import DreamingIndicator from '@/components/home/DreamingIndicator';
import HormonalMini from '@/components/home/HormonalMini';
import ChiLifeForce from '@/components/home/ChiLifeForce';
import NeuralNSMini from '@/components/home/NeuralNSMini';
import TitanDescription from '@/components/home/TitanDescription';
import TitanSelector from '@/components/shared/TitanSelector';
import { useTitanStore } from '@/store/titanStore';
import { useStatus, useVocabulary, useNervousSystem, useCreativeJournal } from '@/hooks/useTitanAPI';
import { formatSOL, formatDuration } from '@/lib/formatters';

const MemoizedSovereigntyGauge = memo(SovereigntyGauge);
const MemoizedCircadianClock = memo(CircadianClock);
const MemoizedVaultStatus = memo(VaultStatus);

export default function HomePage() {
  const status = useTitanStore((s) => s.status);
  const { isDemo } = useStatus();
  const { data: vocabData } = useVocabulary();
  const { data: nsData } = useNervousSystem();
  const { data: journalData } = useCreativeJournal(100);

  const vocabCount = vocabData?.words?.length ?? 0;
  // Phase B.5 lean schema (2026-05-18): titanvm_registers.bin → {programs, age_seconds, seq}.
  // Pre-B.5 total_transitions / total_train_steps are GONE — derive surface metrics
  // from the present-moment program inventory instead.
  const nsPrograms = nsData?.programs ?? {};
  const nsProgramEntries = Object.values(nsPrograms);
  const nsProgramCount = nsProgramEntries.length;
  const nsTotalFires = nsProgramEntries.reduce((s, p) => s + (p.fire_count ?? 0), 0);
  const creativeCount = journalData?.count ?? 0;
  const speakCount = journalData?.entries?.filter(e => e.action_type === 'speak_composition').length ?? 0;

  return (
    <div className="space-y-6">
      {/* Demo indicator */}
      {isDemo && (
        <div className="flex justify-end">
          <DemoBadge />
        </div>
      )}

      {/* Description */}
      <TitanDescription />

      {/* Titan selector */}
      <TitanSelector />

      {/* Top bar: Life Force — removed 2026-05-14 (see import comment) */}

      {/* Neuromod strip + Dreaming + Hormonal mini */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <NeuromodStrip />
        <DreamingIndicator />
        <HormonalMini />
      </div>

      {/* Main grid — three portals */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MemoizedSovereigntyGauge />
        <SpiritSunMini />
        <MemoizedCircadianClock />
      </div>

      {/* Chi Life Force + Neural NS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChiLifeForce />
        <NeuralNSMini />
      </div>

      {/* Quick stats row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="SOL Balance"
          value={status ? formatSOL(status.sol_balance) : '--'}
          sublabel={`Memories: ${status?.memory_count ?? 0}`}
          accent="growth"
        />
        <MetricCard
          label="Vocabulary"
          value={`${vocabCount}`}
          sublabel={`${speakCount} compositions`}
          accent="haze"
        />
        <MetricCard
          label="Neural NS"
          value={nsProgramCount > 0 ? `${nsProgramCount} progs` : '--'}
          sublabel={nsTotalFires > 0 ? `${(nsTotalFires / 1000).toFixed(0)}K fires` : '0 fires'}
          accent="pulse"
        />
        <MetricCard
          label="Creations"
          value={`${creativeCount}`}
          sublabel="speak + art + music"
          accent="haze"
        />
        <MetricCard
          label="Total Epochs"
          value={status?.lifetime?.total_epochs ? `${(status.lifetime.total_epochs / 1000).toFixed(0)}K` : '--'}
          sublabel={status?.lifetime ? `${(status.lifetime.dream_cycles ?? 0).toLocaleString()} dream cycles` : '130D Unified Spirit'}
          accent="metal"
        />
        <MetricCard
          label="Cognitive Depth"
          value={status?.lifetime?.i_depth ? status.lifetime.i_depth.toFixed(3) : '--'}
          sublabel={status?.lifetime ? `I=${(status.lifetime.i_confidence ?? 0).toFixed(2)} · ${(status.lifetime.eurekas ?? 0).toLocaleString()} insights` : 'I-depth · EUREKAs'}
          accent="pulse"
        />
      </div>

      {/* Activity feed removed 2026-05-14 — see import comment */}

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SovereigntyHorizon />
        <MemoizedVaultStatus />
      </div>
    </div>
  );
}

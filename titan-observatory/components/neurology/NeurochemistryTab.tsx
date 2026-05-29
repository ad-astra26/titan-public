'use client';

import dynamic from 'next/dynamic';
import NeuromodGauges from '@/components/neurology/NeuromodGauges';
import HormonalGrid from '@/components/neurology/HormonalGrid';
import ExpressionUrges from '@/components/neurology/ExpressionUrges';
import CouplingNetwork from '@/components/neurology/CouplingNetwork';

const TimeseriesChart = dynamic(() => import('@/components/shared/TimeseriesChart'), { ssr: false });

const NEUROMOD_METRICS = [
  'neuromod.DA', 'neuromod.5HT', 'neuromod.NE',
  'neuromod.ACh', 'neuromod.Endorphin', 'neuromod.GABA',
];

export default function NeurochemistryTab() {
  return (
    <div className="flex flex-col gap-4">
      <NeuromodGauges />
      <TimeseriesChart
        metrics={NEUROMOD_METRICS}
        hours={24}
        title="Neuromodulator History (24h)"
        yDomain={[0, 1]}
      />
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3">
          <HormonalGrid />
        </div>
        <div className="lg:col-span-2">
          <ExpressionUrges />
        </div>
      </div>
      <CouplingNetwork />
    </div>
  );
}

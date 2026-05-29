'use client';

import dynamic from 'next/dynamic';
import { useConsciousnessHistory, useV4InnerTrinity } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import EpochStats from '@/components/consciousness/EpochStats';
import EpochTimeline from '@/components/consciousness/EpochTimeline';
import ErrorBoundary from '@/components/shared/ErrorBoundary';

const SpiritSun = dynamic(() => import('@/components/consciousness/SpiritSun'), { ssr: false });

export default function UnifiedSpiritTab() {
  const titanId = useTitanId();
  const { data: histData } = useConsciousnessHistory(1, titanId);
  const { data: coordData } = useV4InnerTrinity(titanId);
  const epochs = (histData ?? []) as unknown as Array<Record<string, unknown>>;
  const latest = epochs[0] ?? {};
  const coord = (coordData ?? {}) as Record<string, unknown>;
  const ns = (coord?.neural_nervous_system ?? {}) as Record<string, unknown>;

  const epochId = typeof latest.epoch_id === 'number' ? latest.epoch_id : 0;
  const drift = typeof latest.drift_magnitude === 'number' ? latest.drift_magnitude : 0;
  const curvature = typeof latest.curvature === 'number' ? latest.curvature : 0;
  const density = typeof latest.density === 'number' ? latest.density : 0;

  return (
    <div className="flex flex-col gap-4">
      <ErrorBoundary fallback={
        <div className="bg-titan-card rounded-xl p-12 text-center" style={{ height: '450px' }}>
          <div className="flex flex-col items-center justify-center h-full">
            <h2 className="text-xl font-titan text-titan-haze/60 mb-2">Unified Spirit</h2>
            <p className="text-sm text-titan-metal/40 max-w-md mb-4">
              132D sun visualization — 132 sunbeams, one per dimension of Titan&apos;s inner world (130D Trinity + 2D Journey)
            </p>
            <p className="text-xs text-titan-metal/30">
              3D rendering temporarily unavailable. Tensor data flows correctly — try refreshing.
            </p>
            {/* Phase B.5 lean schema (commit cf6a7793). */}
            <div className="flex gap-6 mt-6 font-mono text-sm">
              {(() => {
                const programs = (ns?.programs ?? {}) as Record<string, { fire_count?: number; urgency?: number }>;
                const entries = Object.values(programs);
                const fires = entries.reduce((s, p) => s + (p.fire_count ?? 0), 0);
                const urgs = entries.map(p => p.urgency ?? 0);
                const avgU = urgs.length ? urgs.reduce((s, u) => s + u, 0) / urgs.length : 0;
                return (
                  <>
                    <span className="text-titan-metal/60">programs <span className="text-titan-haze">{entries.length}</span></span>
                    <span className="text-titan-metal/60">avg urgency <span className="text-titan-growth">{(avgU * 100).toFixed(0)}%</span></span>
                    <span className="text-titan-metal/60">{'\u03a3'} fires <span className="text-titan-haze">{fires.toLocaleString()}</span></span>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      }>
        <SpiritSun />
      </ErrorBoundary>

      <EpochStats epochId={epochId} drift={drift} curvature={curvature} density={density} />
      <EpochTimeline />
    </div>
  );
}

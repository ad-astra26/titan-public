'use client';

import { useConsciousnessHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

function curvatureColor(c: number): string {
  if (c < -0.01) return '#4488FF';
  if (c > 0.01) return '#E5C79E';
  return '#8E9AAF';
}

export default function EpochTimeline() {
  const titanId = useTitanId();
  const { data, isLoading } = useConsciousnessHistory(50, titanId);
  const epochs = (data ?? []) as unknown as Array<Record<string, unknown>>;

  if (isLoading || epochs.length === 0) {
    return <div className="bg-titan-card rounded-xl p-4 text-center text-titan-metal/40">Loading epochs...</div>;
  }

  return (
    <div className="bg-titan-card rounded-xl p-4">
      <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-3">Consciousness Timeline</h3>
      <div className="overflow-x-auto pb-2">
        <div className="flex items-end gap-1 min-w-max px-2" style={{ height: '60px' }}>
          {epochs.map((e, i) => {
            const curvature = typeof e.curvature === 'number' ? e.curvature : 0;
            const epochId = typeof e.epoch_id === 'number' ? e.epoch_id : i;
            const anchored = !!e.anchored_tx;
            const isLatest = i === epochs.length - 1;
            const size = isLatest ? 10 : 6;

            return (
              <div key={epochId} className="flex flex-col items-center gap-0.5" title={`Epoch ${epochId} · curv=${curvature.toFixed(4)}`}>
                <div
                  className={`transition-all ${anchored ? 'rotate-45' : 'rounded-full'}`}
                  style={{
                    width: `${size}px`,
                    height: `${size}px`,
                    backgroundColor: anchored ? 'var(--titan-pulse)' : curvatureColor(curvature),
                    boxShadow: isLatest ? `0 0 6px ${curvatureColor(curvature)}` : 'none',
                  }}
                />
              </div>
            );
          })}
        </div>
      </div>
      <div className="flex justify-between text-[10px] text-titan-metal/30 mt-1 px-2">
        <span>oldest</span>
        <span className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" /> expanding
          <span className="w-2 h-2 rounded-full bg-titan-haze inline-block" /> contracting
          <span className="w-2 h-2 rotate-45 bg-titan-pulse inline-block" /> anchored
        </span>
        <span>latest</span>
      </div>
    </div>
  );
}

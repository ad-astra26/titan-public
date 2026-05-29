'use client';

import { useHormonalSystem } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function HormonalMini() {
  const titanId = useTitanId();
  const { data } = useHormonalSystem(titanId);
  const hs = (data ?? {}) as Record<string, unknown>;
  const programs = (hs?.programs ?? {}) as Record<string, Record<string, unknown>>;
  const maturity = typeof hs?.maturity === 'number' ? hs.maturity : 0;
  const transitions = typeof hs?.total_transitions === 'number' ? hs.total_transitions : 0;

  const firing = Object.entries(programs).filter(
    ([, p]) => typeof (p as Record<string, unknown>)?.fire_count === 'number' && ((p as Record<string, unknown>).fire_count as number) > 0
  ).length;
  const total = Object.keys(programs).length || 11;

  return (
    <div className="bg-titan-card rounded-xl px-4 py-3 flex items-center gap-4">
      <span className="text-xs text-titan-metal/40 uppercase whitespace-nowrap">Neural NS</span>
      <div className="flex items-center gap-3 flex-1">
        <div className="flex items-center gap-1">
          <span className="font-mono text-sm text-titan-haze">{firing}/{total}</span>
          <span className="text-xs text-titan-metal/40">firing</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-16 h-1.5 bg-titan-bg rounded-full overflow-hidden">
            <div className="h-full bg-titan-growth rounded-full transition-all" style={{ width: `${maturity * 100}%` }} />
          </div>
          <span className="font-mono text-xs text-titan-metal/40">{maturity.toFixed(2)}</span>
        </div>
        <span className="font-mono text-xs text-titan-metal/40">{transitions.toLocaleString()} trans</span>
      </div>
    </div>
  );
}

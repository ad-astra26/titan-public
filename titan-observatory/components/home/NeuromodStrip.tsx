'use client';

import { useNeuromodulators } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const MODS = ['DA', '5HT', 'NE', 'ACh', 'Endorphin', 'GABA'];
const COLORS: Record<string, string> = {
  DA: '#E5C79E', '5HT': '#4488FF', NE: '#FF6644',
  ACh: '#77CCCC', Endorphin: '#FF88CC', GABA: '#9945FF',
};

export default function NeuromodStrip() {
  const titanId = useTitanId();
  const { data } = useNeuromodulators(titanId);
  const nm = (data ?? {}) as Record<string, unknown>;
  const modulators = (nm?.modulators ?? {}) as Record<string, Record<string, unknown>>;
  const emotion = nm?.current_emotion as string ?? '—';

  return (
    <div className="bg-titan-card rounded-xl px-4 py-3 flex items-center gap-4">
      <span className="text-xs text-titan-metal/40 uppercase whitespace-nowrap">Neuromods</span>
      <div className="flex-1 flex gap-2">
        {MODS.map(name => {
          const level = typeof (modulators[name] as Record<string, unknown>)?.level === 'number'
            ? (modulators[name] as Record<string, unknown>).level as number : 0;
          return (
            <div key={name} className="flex-1 flex flex-col gap-0.5" title={`${name}: ${level.toFixed(3)}`}>
              <div className="h-2 bg-titan-bg rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(level, 1) * 100}%`, backgroundColor: COLORS[name] }} />
              </div>
              <span className="text-[8px] font-mono text-titan-metal/40 text-center">{name}</span>
            </div>
          );
        })}
      </div>
      <span className="text-sm text-titan-haze font-titan whitespace-nowrap">{emotion}</span>
    </div>
  );
}

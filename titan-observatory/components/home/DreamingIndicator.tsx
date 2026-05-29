'use client';

import { useDreaming } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function DreamingIndicator() {
  const titanId = useTitanId();
  const { data } = useDreaming(titanId);
  const isDreaming = data?.is_dreaming === true;
  const fatigue = data?.fatigue ?? 0;
  const cycles = data?.cycle_count ?? 0;
  const devAge = data?.developmental_age ?? 0;

  return (
    <div className="bg-titan-card rounded-xl px-4 py-3 flex items-center gap-4">
      <div className="flex items-center gap-2">
        <span className={`text-lg ${isDreaming ? 'animate-pulse' : ''}`}>
          {isDreaming ? '🌙' : '☀️'}
        </span>
        <span className={`text-sm font-titan ${isDreaming ? 'text-blue-400' : 'text-titan-haze'}`}>
          {isDreaming ? 'Dreaming' : 'Awake'}
        </span>
      </div>
      <div className="flex-1 flex items-center gap-3">
        <div className="flex-1">
          <div className="h-1.5 bg-titan-bg rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all duration-1000"
              style={{
                width: `${fatigue * 100}%`,
                backgroundColor: isDreaming ? '#4488FF' : 'var(--titan-haze)',
              }} />
          </div>
        </div>
        <span className="font-mono text-xs text-titan-metal/60">{(fatigue * 100).toFixed(0)}%</span>
      </div>
      <div className="flex gap-3 text-xs font-mono text-titan-metal/40">
        <span>cycles: {cycles}</span>
        <span>age: {devAge}</span>
      </div>
    </div>
  );
}

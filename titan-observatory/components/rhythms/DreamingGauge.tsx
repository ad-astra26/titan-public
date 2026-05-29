'use client';

import { useDreaming } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function DreamingGauge() {
  const titanId = useTitanId();
  const { data, isLoading } = useDreaming(titanId);
  const isDreaming = data?.is_dreaming === true;
  const fatigue = data?.fatigue ?? 0;
  const cycleCount = data?.cycle_count ?? 0;
  const epochsSince = data?.epochs_since_dream ?? 0;
  const devAge = data?.developmental_age ?? 0;
  const recoveryPct = data?.recovery_pct ?? 0;

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 text-center text-titan-metal/40">Loading dreaming...</div>;
  }

  // SVG circular gauge
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const progress = isDreaming ? recoveryPct / 100 : fatigue;
  const dashOffset = circumference * (1 - progress);
  const arcColor = isDreaming ? '#4488FF' : '#E5C79E';

  // Estimate time to dream (fatigue path)
  const threshold = devAge * 50 + 100;
  const remaining = Math.max(0, threshold - epochsSince);
  const estMinutes = Math.round(remaining * 7.2 / 60);

  return (
    <div className="bg-titan-card rounded-xl p-6 flex flex-col items-center">
      <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-4">Dreaming Cycle</h3>

      <div className="relative" style={{ width: '160px', height: '160px' }}>
        <svg viewBox="0 0 160 160" className="w-full h-full">
          {/* Background circle */}
          <circle cx="80" cy="80" r={radius} fill="none" stroke="var(--titan-bg)" strokeWidth="8" />
          {/* Progress arc */}
          <circle cx="80" cy="80" r={radius} fill="none"
            stroke={arcColor} strokeWidth="8" strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={dashOffset}
            transform="rotate(-90 80 80)"
            style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.5s' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-2xl" style={{ color: arcColor }}>
            {isDreaming ? `${recoveryPct.toFixed(0)}%` : `${(fatigue * 100).toFixed(0)}%`}
          </span>
          <span className="text-xs text-titan-metal/40">
            {isDreaming ? 'recovering' : 'fatigue'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 mt-4 text-center">
        <div>
          <span className="text-xs text-titan-metal/40">State</span>
          <p className={`font-mono text-sm ${isDreaming ? 'text-blue-400' : 'text-titan-haze'}`}>
            {isDreaming ? 'Dreaming' : 'Awake'}
          </p>
        </div>
        <div>
          <span className="text-xs text-titan-metal/40">Cycles</span>
          <p className="font-mono text-sm text-titan-metal">{cycleCount}</p>
        </div>
        <div>
          <span className="text-xs text-titan-metal/40">Epochs Awake</span>
          <p className="font-mono text-sm text-titan-metal">{epochsSince.toLocaleString()}</p>
        </div>
        <div>
          <span className="text-xs text-titan-metal/40">Est. to Dream</span>
          <p className="font-mono text-sm text-titan-metal">{estMinutes > 0 ? `~${estMinutes}m` : 'soon'}</p>
        </div>
      </div>
    </div>
  );
}

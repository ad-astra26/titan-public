'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useHormonalSystem, useNeuromodulators, useStatus } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function SovereigntyGauge() {
  const [hovered, setHovered] = useState(false);
  const router = useRouter();
  const titanId = useTitanId();
  const { data: hsData } = useHormonalSystem(titanId);
  const { data: nmData } = useNeuromodulators(titanId);
  const { data: statusData } = useStatus(titanId);
  const hs = (hsData ?? {}) as Record<string, unknown>;
  const nm = (nmData ?? {}) as Record<string, unknown>;
  const lifetime = statusData?.lifetime as Record<string, unknown> | undefined;

  const maturity = typeof hs?.maturity === 'number' ? hs.maturity : 0;
  const transitions = typeof hs?.total_transitions === 'number' ? hs.total_transitions : 0;
  const trainSteps = typeof hs?.total_train_steps === 'number' ? hs.total_train_steps : 0;
  const programs = (hs?.programs ?? {}) as Record<string, Record<string, unknown>>;
  const firing = Object.values(programs).filter(
    p => typeof p?.fire_count === 'number' && (p.fire_count as number) > 0
  ).length;
  const evals = typeof (nm as Record<string, unknown>)?.total_evaluations === 'number'
    ? (nm as Record<string, unknown>).total_evaluations as number : 0;

  const radius = 60;
  const stroke = 8;
  const circumference = 2 * Math.PI * radius;
  const dashoffset = circumference - maturity * circumference;

  // Pie chart segments for the inner ring
  const segments = [
    { label: 'Programs', value: firing / 11, color: '#E5C79E' },
    { label: 'Maturity', value: maturity, color: '#77CCCC' },
    { label: 'Training', value: Math.min(trainSteps / 600000, 1), color: '#9945FF' },
  ];

  return (
    <div
      className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5 flex flex-col items-center cursor-pointer relative overflow-hidden transition-all hover:border-titan-growth/30 hover:shadow-growth_glow"
      onClick={() => router.push('/neurology')}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4 self-start">
        Neural Maturity
      </h3>
      <div className="relative">
        <svg width={160} height={160} className="-rotate-90">
          <circle cx={80} cy={80} r={radius} fill="none"
            stroke="currentColor" strokeWidth={stroke} className="text-titan-metal/10" />
          <circle cx={80} cy={80} r={radius} fill="none"
            stroke="currentColor" strokeWidth={stroke} strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={dashoffset}
            className="text-titan-growth transition-all duration-1000 ease-out"
            style={{ filter: 'drop-shadow(0 0 8px rgba(119, 204, 204, 0.4))' }}
          />
          {/* Inner pie segments */}
          {segments.map((seg, i) => {
            const innerR = 40;
            const innerCirc = 2 * Math.PI * innerR;
            const segLen = seg.value * innerCirc * 0.3;
            const offset = innerCirc - segLen;
            const rotation = i * 120 - 90;
            return (
              <circle key={seg.label} cx={80} cy={80} r={innerR} fill="none"
                stroke={seg.color} strokeWidth={4} strokeLinecap="round"
                strokeDasharray={`${segLen} ${innerCirc}`}
                strokeDashoffset={0} opacity={0.6}
                style={{ transform: `rotate(${rotation}deg)`, transformOrigin: '80px 80px',
                  transition: 'all 1s' }}
              />
            );
          })}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center rotate-0">
          <span className="text-2xl font-bold text-titan-growth">
            {(maturity * 100).toFixed(0)}
          </span>
          <span className="text-[10px] text-titan-metal/50 uppercase tracking-wider">
            maturity %
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 mt-3 text-center w-full">
        <div>
          <span className="text-[9px] text-titan-haze/50">Epochs</span>
          <p className="font-mono text-xs text-titan-metal">{lifetime?.total_epochs ? `${((lifetime.total_epochs as number) / 1000).toFixed(0)}K` : `${firing}/11`}</p>
        </div>
        <div>
          <span className="text-[9px] text-titan-growth/50">Train Steps</span>
          <p className="font-mono text-xs text-titan-metal">{(trainSteps / 1000000).toFixed(1)}M</p>
        </div>
        <div>
          <span className="text-[9px] text-titan-pulse/50">Insights</span>
          <p className="font-mono text-xs text-titan-metal">{lifetime?.eurekas ? (lifetime.eurekas as number).toLocaleString() : evals}</p>
        </div>
      </div>
      <p className="text-[10px] text-titan-metal/30 mt-1">click to explore →</p>
      {hovered && (
        <div className="absolute inset-0 bg-titan-bg backdrop-blur-sm rounded-xl p-4 flex flex-col justify-center z-20">
          <h4 className="text-sm font-titan text-titan-growth mb-2">Neural Maturity</h4>
          <p className="text-xs text-titan-metal/70 leading-relaxed">
            Titan&apos;s nervous system has 11 autonomous programs that learn from experience.
            Maturity grows as Titan accumulates epochs, GREAT PULSEs, and hormonal fires.
            Higher maturity = faster recovery, larger capacity, deeper self-regulation.
          </p>
          <p className="text-[10px] text-titan-growth/50 mt-2">Click for full neurology dashboard →</p>
        </div>
      )}
    </div>
  );
}

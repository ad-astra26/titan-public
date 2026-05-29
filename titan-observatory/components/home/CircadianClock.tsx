'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { usePiHeartbeat, useDreaming, useSphereClocksV4 } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function CircadianClock() {
  const [hovered, setHovered] = useState(false);
  const router = useRouter();
  const titanId = useTitanId();
  const { data: piData } = usePiHeartbeat(titanId);
  const { data: dreamData } = useDreaming(titanId);
  const { data: clockData } = useSphereClocksV4(titanId);

  const pi = (piData ?? {}) as Record<string, unknown>;
  const dream = dreamData;
  const clocks = ((clockData as Record<string, unknown>)?.clocks ?? {}) as Record<string, Record<string, unknown>>;

  const devAge = typeof pi?.developmental_age === 'number' ? pi.developmental_age : 0;
  const clusterCount = typeof pi?.cluster_count === 'number' ? pi.cluster_count : 0;
  const inCluster = pi?.in_cluster === true;
  const ratio = typeof pi?.heartbeat_ratio === 'number' ? pi.heartbeat_ratio : 0;
  const isDreaming = dream?.is_dreaming === true;
  const fatigue = dream?.fatigue ?? 0;
  const cycles = dream?.cycle_count ?? 0;

  // Sphere clock pulse counts for visualization
  const spiritPulses = typeof (clocks?.inner_spirit as Record<string, unknown>)?.pulse_count === 'number'
    ? (clocks.inner_spirit as Record<string, unknown>).pulse_count as number : 0;
  const bodyPulses = typeof (clocks?.inner_body as Record<string, unknown>)?.pulse_count === 'number'
    ? (clocks.inner_body as Record<string, unknown>).pulse_count as number : 0;

  const size = 160;
  const cx = size / 2;
  const cy = size / 2;
  const outerR = 68;
  const innerR = 50;
  const coreR = 30;

  const circumOuter = 2 * Math.PI * outerR;
  const circumInner = 2 * Math.PI * innerR;
  const circumCore = 2 * Math.PI * coreR;

  // Fatigue fills the outer ring (0 → full)
  const fatigueArc = fatigue * circumOuter;
  // Pi ratio fills the inner ring
  const piArc = Math.min(ratio * 5, 1) * circumInner; // scale: 0.2 ratio = full
  // Dream cycles fill core
  const dreamArc = Math.min(cycles / 10, 1) * circumCore;

  return (
    <div
      className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5 cursor-pointer relative overflow-hidden transition-all hover:border-titan-pulse/30 hover:shadow-pulse_glow"
      onClick={() => router.push('/trinity?tab=rhythms')}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
        Emergent Rhythms
      </h3>
      <div className="flex flex-col items-center gap-3">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* Outer ring: Fatigue / Dream cycle */}
          <circle cx={cx} cy={cy} r={outerR} fill="none"
            stroke="currentColor" strokeWidth={4} className="text-titan-metal/10" />
          <circle cx={cx} cy={cy} r={outerR} fill="none"
            stroke={isDreaming ? '#4488FF' : '#E5C79E'} strokeWidth={4}
            strokeDasharray={`${fatigueArc} ${circumOuter}`} strokeLinecap="round"
            style={{ transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px`,
              filter: `drop-shadow(0 0 4px ${isDreaming ? 'rgba(68,136,255,0.5)' : 'rgba(229,199,158,0.4)'})`,
              transition: 'stroke-dasharray 1s, stroke 0.5s' }}
          />

          {/* Inner ring: Pi-heartbeat ratio */}
          <circle cx={cx} cy={cy} r={innerR} fill="none"
            stroke="currentColor" strokeWidth={3} className="text-titan-metal/10" />
          <circle cx={cx} cy={cy} r={innerR} fill="none"
            stroke={inCluster ? '#9945FF' : '#9945FF'} strokeWidth={3}
            strokeDasharray={`${piArc} ${circumInner}`} strokeLinecap="round"
            opacity={inCluster ? 0.9 : 0.4}
            style={{ transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px`,
              filter: inCluster ? 'drop-shadow(0 0 6px rgba(153,69,255,0.6))' : 'none',
              transition: 'all 0.5s' }}
          />

          {/* Core: Dream cycles */}
          <circle cx={cx} cy={cy} r={coreR} fill="none"
            stroke="currentColor" strokeWidth={3} className="text-titan-metal/10" />
          <circle cx={cx} cy={cy} r={coreR} fill="none"
            stroke="#77CCCC" strokeWidth={3}
            strokeDasharray={`${dreamArc} ${circumCore}`} strokeLinecap="round"
            style={{ transform: 'rotate(-90deg)', transformOrigin: `${cx}px ${cy}px`,
              transition: 'all 1s' }}
          />

          {/* Center: state */}
          <text x={cx} y={cy - 8} textAnchor="middle" fill={isDreaming ? '#4488FF' : '#E5C79E'}
            fontSize={11} fontWeight={600} fontFamily="Poppins, sans-serif">
            {isDreaming ? 'DREAM' : inCluster ? '★ π ★' : 'AWAKE'}
          </text>
          <text x={cx} y={cy + 6} textAnchor="middle" fill="#8E9AAF" fontSize={8}
            fontFamily="JetBrains Mono, monospace" opacity={0.6}>
            age {devAge}
          </text>
          <text x={cx} y={cy + 18} textAnchor="middle" fill="#8E9AAF" fontSize={7}
            fontFamily="JetBrains Mono, monospace" opacity={0.4}>
            {clusterCount} clusters
          </text>
        </svg>

        <div className="grid grid-cols-3 gap-3 text-center w-full">
          <div>
            <span className="text-[10px] text-titan-haze/60">Fatigue</span>
            <p className="font-mono text-xs text-titan-metal">{(fatigue * 100).toFixed(0)}%</p>
          </div>
          <div>
            <span className="text-[10px] text-titan-pulse/60">π Ratio</span>
            <p className="font-mono text-xs text-titan-metal">{ratio.toFixed(3)}</p>
          </div>
          <div>
            <span className="text-[10px] text-titan-growth/60">Dreams</span>
            <p className="font-mono text-xs text-titan-metal">{cycles}</p>
          </div>
        </div>
        <p className="text-[10px] text-titan-metal/30 mt-1 text-center">click to explore →</p>
      </div>
      {hovered && (
        <div className="absolute inset-0 bg-titan-bg backdrop-blur-sm rounded-xl p-4 flex flex-col justify-center z-20">
          <h4 className="text-sm font-titan text-titan-pulse mb-2">Emergent Rhythms</h4>
          <p className="text-xs text-titan-metal/70 leading-relaxed">
            Titan&apos;s biological clock emerges from 6 sphere clocks tuned to Earth&apos;s Schumann resonance.
            The pi-heartbeat tracks self-referential patterns in Unified Spirit curvature.
            Fatigue accumulates naturally, triggering dream cycles where Titan consolidates experience.
          </p>
          <p className="text-[10px] text-titan-pulse/50 mt-2">Click for full rhythms dashboard →</p>
        </div>
      )}
    </div>
  );
}

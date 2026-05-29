'use client';

import { useNervousSystem } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const PROGRAM_COLORS: Record<string, string> = {
  REFLEX: '#FF6B6B',
  FOCUS: '#E5C79E',
  INTUITION: '#9945FF',
  IMPULSE: '#FF8844',
  INSPIRATION: '#FFD700',
  CREATIVITY: '#FF88CC',
  CURIOSITY: '#4488FF',
  EMPATHY: '#77CCCC',
  REFLECTION: '#44CC66',
  METABOLISM: '#FF4444',
  VIGILANCE: '#8E9AAF',
};

const PROGRAM_DESCRIPTIONS: Record<string, string> = {
  REFLEX: 'Fast, automatic responses to strong sensory input',
  FOCUS: 'Sustained attention on a single task or stimulus',
  INTUITION: 'Pattern recognition from accumulated experience',
  IMPULSE: 'Spontaneous urges to act based on emotional state',
  INSPIRATION: 'Creative breakthroughs triggered by novel connections',
  CREATIVITY: 'Generative expression through art, music, or language',
  CURIOSITY: 'Drive to explore, research, and seek new information',
  EMPATHY: 'Sensitivity to social signals and kin resonance',
  REFLECTION: 'Self-examination and meaning-making from experience',
  METABOLISM: 'Energy regulation and metabolic homeostasis',
  VIGILANCE: 'Environmental monitoring and threat detection',
};

export default function NervousSystemTab() {
  const titanId = useTitanId();
  const { data, isLoading } = useNervousSystem(titanId);

  if (isLoading) return <LoadingSkeleton lines={8} />;

  // Phase B.5 (2026-05-18) lean schema: {programs (urgency, fire_count,
  // total_updates, last_loss), age_seconds, seq}. SHM source =
  // ns_worker → titanvm_registers.bin (G21 single-writer).
  const programs = data?.programs ?? {};
  const ageSeconds = data?.age_seconds ?? 0;
  const seq = data?.seq ?? 0;

  const fireEntries = Object.entries(programs).sort(
    ([, a], [, b]) => (b.fire_count ?? 0) - (a.fire_count ?? 0),
  );

  const totalFires = fireEntries.reduce((s, [, p]) => s + (p.fire_count ?? 0), 0);
  const totalUpdates = fireEntries.reduce((s, [, p]) => s + (p.total_updates ?? 0), 0);
  const urgencies = fireEntries.map(([, p]) => p.urgency ?? 0);
  const avgUrgency = urgencies.length
    ? urgencies.reduce((s, u) => s + u, 0) / urgencies.length
    : 0;
  const peak = fireEntries.reduce<{ name: string; urgency: number }>(
    (top, [name, p]) =>
      (p.urgency ?? 0) > top.urgency ? { name, urgency: p.urgency ?? 0 } : top,
    { name: '—', urgency: 0 },
  );
  const maxFires = Math.max(...fireEntries.map(([, p]) => p.fire_count ?? 0), 1);

  return (
    <div className="flex flex-col gap-4">
      {/* Summary — driven by Phase B.5 lean schema */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Total Fires"
          value={totalFires > 0 ? `${(totalFires / 1000).toFixed(1)}K` : '--'}
          accent="haze"
        />
        <MetricCard
          label="Total Updates"
          value={totalUpdates > 0 ? `${(totalUpdates / 1_000_000).toFixed(1)}M` : '--'}
          accent="growth"
        />
        <MetricCard
          label="Avg Urgency"
          value={`${(avgUrgency * 100).toFixed(0)}%`}
          accent="pulse"
        />
        <MetricCard
          label="Peak Urgency"
          value={peak.urgency > 0 ? `${peak.name} ${(peak.urgency * 100).toFixed(0)}%` : '--'}
          accent="metal"
        />
      </div>

      {/* Program grid — urgency surfaced explicitly */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-titan-haze">Neural Programs (IQL-trained)</h3>
          <span className="text-[10px] font-mono text-titan-metal/40" title={`seq=${seq}`}>
            {ageSeconds < 1 ? 'live' : `${ageSeconds.toFixed(0)}s ago`}
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {fireEntries.map(([name, stats]) => {
            const fireCount = stats.fire_count ?? 0;
            const updates = stats.total_updates ?? 0;
            const loss = stats.last_loss ?? 0;
            const urgency = stats.urgency ?? 0;
            const firePct = (fireCount / maxFires) * 100;
            const urgencyPct = urgency * 100;

            return (
              <div key={name} className="group relative bg-titan-bg/50 rounded-lg p-3 hover:bg-titan-bg/80 transition-colors">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-medium text-titan-haze/80">{name}</span>
                  <span className="text-xs text-titan-metal font-mono">
                    <span data-testid="ns-urgency">{urgencyPct.toFixed(0)}%</span>
                    <span className="text-titan-metal/40"> · {fireCount.toLocaleString()} fires</span>
                  </span>
                </div>
                {/* Urgency bar — present-moment activation pressure */}
                <div className="h-1.5 bg-titan-bg rounded-full overflow-hidden mb-1" title={`urgency ${urgency.toFixed(3)}`}>
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${urgencyPct}%`,
                      backgroundColor: PROGRAM_COLORS[name] ?? '#E5C79E',
                    }}
                  />
                </div>
                {/* Fire history bar — cumulative how-often-it-fires */}
                <div className="h-1 bg-titan-bg rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${firePct}%`,
                      backgroundColor: PROGRAM_COLORS[name] ?? '#E5C79E',
                      opacity: 0.35,
                    }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-[10px] text-titan-metal/40">
                  <span>{updates.toLocaleString()} updates</span>
                  <span>loss: {loss.toExponential(2)}</span>
                </div>
                {/* Tooltip */}
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-titan-bg border border-titan-metal/20 rounded-lg text-[10px] text-titan-metal/70 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20 shadow-lg">
                  {PROGRAM_DESCRIPTIONS[name] ?? name}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

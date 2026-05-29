'use client';

import { useNervousSystem } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import InfoTooltip from '@/components/shared/InfoTooltip';

const PROGRAM_COLORS: Record<string, string> = {
  REFLEX: 'bg-red-400/60',
  FOCUS: 'bg-blue-400/60',
  INTUITION: 'bg-purple-400/60',
  IMPULSE: 'bg-orange-400/60',
  INSPIRATION: 'bg-yellow-400/60',
  CREATIVITY: 'bg-pink-400/60',
  CURIOSITY: 'bg-cyan-400/60',
  EMPATHY: 'bg-green-400/60',
  REFLECTION: 'bg-indigo-400/60',
  METABOLISM: 'bg-emerald-400/60',
  VIGILANCE: 'bg-amber-400/60',
};

export default function NeuralNSMini() {
  const titanId = useTitanId();
  const { data: ns } = useNervousSystem(titanId);

  if (!ns || !ns.programs) {
    return (
      <div className="bg-titan-card rounded-xl p-4 animate-pulse">
        <div className="h-4 bg-titan-metal/10 rounded w-32 mb-3" />
        <div className="grid grid-cols-5 gap-2">
          {Array.from({ length: 11 }).map((_, i) => (
            <div key={i} className="h-10 bg-titan-metal/10 rounded" />
          ))}
        </div>
      </div>
    );
  }

  const programs = Object.entries(ns.programs);
  const maxFires = Math.max(...programs.map(([, p]) => p.fire_count || 1), 1);
  const totalFires = programs.reduce((s, [, p]) => s + (p.fire_count ?? 0), 0);
  const totalUpdates = programs.reduce((s, [, p]) => s + (p.total_updates ?? 0), 0);
  const ageS = ns.age_seconds ?? 0;

  return (
    <div className="bg-titan-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <InfoTooltip text="11 neural programs that learn autonomously through experience. Each program has a tiny neural network that decides when to fire. Bars show relative fire frequency. Color intensity tracks urgency (how strongly the program wants to fire right now). Hover for details.">
            <h3 className="text-xs font-mono uppercase tracking-wider text-titan-metal/60">Neural Nervous System</h3>
          </InfoTooltip>
          <span className="text-[10px] font-mono text-titan-metal/40">
            {programs.length} programs · {totalFires.toLocaleString()} fires · {(totalUpdates / 1000).toFixed(0)}K updates
          </span>
        </div>
        <span className="text-[10px] font-mono text-titan-growth" title={`seq=${ns.seq}`}>
          {ageS < 1 ? 'live' : `${ageS.toFixed(0)}s ago`}
        </span>
      </div>

      <div className="grid grid-cols-5 gap-1.5">
        {programs.map(([name, prog]) => {
          const firePct = Math.max(5, (prog.fire_count / maxFires) * 100);
          const urgency = prog.urgency ?? 0;
          // Urgency drives color OPACITY (0.25 baseline → 1.0 at urgency=1).
          // Bar height tracks fire count (history); color saturation tracks
          // urgency (present-moment "wanting to fire").
          const urgencyOpacity = (0.25 + urgency * 0.75).toFixed(2);
          const bgColor = PROGRAM_COLORS[name] ?? 'bg-titan-metal/40';

          return (
            <div key={name} className="relative group" title={`${name}: urgency=${urgency.toFixed(3)}, ${prog.fire_count} fires`}>
              <div className="bg-titan-bg rounded-lg p-1.5 text-center">
                <div className="w-full h-6 bg-titan-bg/50 rounded overflow-hidden mb-1 relative">
                  <div
                    className={`absolute bottom-0 left-0 right-0 ${bgColor} rounded transition-all duration-1000`}
                    style={{ height: `${firePct}%`, opacity: urgencyOpacity }}
                  />
                </div>
                <span className="text-[8px] font-mono text-titan-metal/60 leading-none block truncate">
                  {name.slice(0, 4)}
                </span>
                <span className="text-[9px] font-mono text-titan-haze/70">
                  u{urgency.toFixed(2)}
                </span>
              </div>
              {/* Tooltip on hover — lean schema only */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-20
                             bg-titan-bg border border-titan-metal/20 rounded-lg p-2 min-w-[140px] shadow-lg">
                <div className="text-[10px] font-mono text-titan-haze mb-1">{name}</div>
                <div className="text-[9px] font-mono text-titan-metal/60 space-y-0.5">
                  <div>urgency: {urgency.toFixed(4)}</div>
                  <div>fires: {(prog.fire_count ?? 0).toLocaleString()}</div>
                  <div>updates: {(prog.total_updates ?? 0).toLocaleString()}</div>
                  <div>last loss: {(prog.last_loss ?? 0).toExponential(2)}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

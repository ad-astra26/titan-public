'use client';

import { useHormonalSystem } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import InfoTooltip from '@/components/shared/InfoTooltip';

// 11 NS programs (post-Phase B.5 — METABOLISM added; was 10 in legacy schema).
// Order matches titanvm_registers.bin row order from ns_worker.
const PROGRAMS = ['REFLEX', 'FOCUS', 'INTUITION', 'IMPULSE', 'METABOLISM',
                  'CREATIVITY', 'CURIOSITY', 'EMPATHY', 'REFLECTION',
                  'INSPIRATION', 'VIGILANCE'];

function ProgramCell({ name, prog }: { name: string; prog: Record<string, unknown> }) {
  // Phase B.5 NS program shape (titanvm_registers.bin):
  //   { urgency, fire_count, total_updates, last_loss }
  // Pre-B.5 shape was { level, threshold, refractory, fire_count } — fall
  // back to those if a future producer surfaces them again.
  const urgency = typeof prog?.urgency === 'number' ? prog.urgency : 0;
  const level = typeof prog?.level === 'number' ? prog.level : urgency;
  const threshold = typeof prog?.threshold === 'number' ? prog.threshold : 1;
  const refractory = typeof prog?.refractory === 'number' ? prog.refractory : 0;
  const fireCount = typeof prog?.fire_count === 'number' ? prog.fire_count : 0;
  const totalUpdates = typeof prog?.total_updates === 'number' ? prog.total_updates : 0;
  // Pressure = urgency directly (0-1 normalized) under B.5; fall back to
  // level/threshold ratio if those are present (legacy producers).
  const pressure = prog?.urgency !== undefined
    ? Math.min(Math.max(urgency, 0), 1)
    : (threshold > 0 ? Math.min(level / threshold, 1) : 0);
  const isNew = fireCount > 0 && fireCount < 20;

  return (
    <div className="bg-titan-bg rounded-lg p-3 relative overflow-hidden">
      {/* Refractory overlay (legacy producers only) */}
      {refractory > 0.01 && (
        <div
          className="absolute inset-0 bg-titan-bg/70 transition-opacity duration-300 pointer-events-none"
          style={{ opacity: refractory * 0.7 }}
        />
      )}
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-xs text-titan-metal/80 uppercase">{name}</span>
          <span className={`font-mono text-xs ${fireCount > 0 ? 'text-titan-haze' : 'text-titan-metal/30'}`}>
            {isNew && '★ '}{fireCount.toLocaleString()}
          </span>
        </div>
        <div className="h-2 bg-titan-card rounded-sm overflow-hidden">
          <div
            className="h-full rounded-sm transition-all duration-300"
            style={{
              width: `${pressure * 100}%`,
              backgroundColor: pressure > 0.8 ? 'var(--titan-haze)' : 'var(--titan-metal)',
              opacity: 0.7,
            }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="font-mono text-[10px] text-titan-metal/40">u:{pressure.toFixed(2)}</span>
          <span className="font-mono text-[10px] text-titan-metal/40">{totalUpdates > 0 ? `${(totalUpdates/1000).toFixed(0)}K upd` : `r:${refractory.toFixed(2)}`}</span>
        </div>
      </div>
    </div>
  );
}

export default function HormonalGrid() {
  const titanId = useTitanId();
  const { data, isLoading } = useHormonalSystem(titanId);
  const hs = (data ?? {}) as Record<string, unknown>;
  const programs = (hs?.programs ?? {}) as Record<string, Record<string, unknown>>;
  const maturity = typeof hs?.maturity === 'number' ? hs.maturity : 0;
  const transitions = typeof hs?.total_transitions === 'number'
    ? hs.total_transitions
    : (typeof hs?.total_train_steps === 'number' ? hs.total_train_steps : 0);

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 text-center text-titan-metal/40">Loading hormonal system...</div>;
  }

  return (
    <div className="bg-titan-card rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <InfoTooltip text="10 hormonal programs that accumulate over time and fire when they exceed a threshold. Each is driven by neuromodulator combinations. When a program fires, it triggers an expression (Speak, Art, Music, or Social).">
          <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider">Hormonal Programs</h3>
        </InfoTooltip>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="text-xs text-titan-metal/40">maturity</span>
            <div className="w-16 h-1.5 bg-titan-bg rounded-full overflow-hidden">
              <div className="h-full bg-titan-growth rounded-full transition-all" style={{ width: `${maturity * 100}%` }} />
            </div>
            <span className="font-mono text-xs text-titan-metal/60">{maturity.toFixed(2)}</span>
          </div>
          <span className="font-mono text-xs text-titan-metal/40">{transitions.toLocaleString()} trans</span>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
        {PROGRAMS.map(name => (
          <ProgramCell key={name} name={name} prog={(programs[name] ?? {}) as Record<string, unknown>} />
        ))}
      </div>
    </div>
  );
}

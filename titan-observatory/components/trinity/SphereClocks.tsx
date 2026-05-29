'use client';

import { useSphereClocksV4 } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const CLOCK_CONFIG = [
  { name: 'inner_spirit', label: 'Spirit', freq: '0.383s', row: 'Inner' },
  { name: 'inner_mind', label: 'Mind', freq: '1.15s', row: 'Inner' },
  { name: 'inner_body', label: 'Body', freq: '3.45s', row: 'Inner' },
  { name: 'outer_spirit', label: 'Spirit', freq: '0.383s', row: 'Outer' },
  { name: 'outer_mind', label: 'Mind', freq: '1.15s', row: 'Outer' },
  { name: 'outer_body', label: 'Body', freq: '3.45s', row: 'Outer' },
];

const SCHUMANN_DURATIONS: Record<string, string> = {
  spirit: '0.383s',
  mind: '1.15s',
  body: '3.45s',
};

function ClockCircle({ clock, data }: { clock: typeof CLOCK_CONFIG[0]; data: Record<string, unknown> }) {
  const layer = clock.name.split('_')[1]; // spirit, mind, body
  const duration = SCHUMANN_DURATIONS[layer] ?? '1s';
  const radius = typeof data?.radius === 'number' ? data.radius : 0.5;
  const streak = typeof (data?.consecutive_balanced ?? data?.streak) === 'number' ? (data?.consecutive_balanced ?? data?.streak) as number : 0;
  const pulseCount = typeof data?.pulse_count === 'number' ? data.pulse_count : 0;
  const balanced = streak > 100;
  const scale = 0.4 + (1.0 - radius) * 1.2; // smaller radius = bigger visual (more mature)

  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="relative rounded-full border-2 flex items-center justify-center"
        style={{
          width: `${48 * scale}px`,
          height: `${48 * scale}px`,
          borderColor: balanced ? 'var(--titan-haze)' : 'var(--titan-metal)',
          boxShadow: balanced ? '0 0 12px -2px rgba(229,199,158,0.4)' : 'none',
          animation: `pulse ${duration} ease-in-out infinite`,
        }}
      >
        <span className="font-mono text-xs" style={{ color: balanced ? 'var(--titan-haze)' : 'var(--titan-metal)' }}>
          {radius.toFixed(2)}
        </span>
      </div>
      <span className="text-xs text-titan-metal">{clock.label}</span>
      <span className="font-mono text-xs text-titan-metal/60">
        {pulseCount.toLocaleString()} · {streak > 999 ? `${(streak / 1000).toFixed(1)}k` : streak}
      </span>
    </div>
  );
}

export default function SphereClocks() {
  const titanId = useTitanId();
  const { data, isLoading } = useSphereClocksV4(titanId);
  // Backend /v6/trinity/sphere-clocks returns the clocks dict FLAT (each layer key
  // at top level: {inner_body:{...}, inner_mind:{...}, ...}) — there is no
  // wrapping `clocks` field. Pre-fix this component read data.clocks
  // (always undefined) so all 6 ClockCircles defaulted to radius=0.5,
  // streak=0 → "0.50" with no resonance. Read flat first, fall back to
  // .clocks shape for safety. Per rFP_observatory_data_loading_v1 §3.2.
  const raw = (data ?? {}) as Record<string, unknown>;
  const wrapped = raw.clocks as Record<string, Record<string, unknown>> | undefined;
  const clocks = (wrapped && Object.keys(wrapped).length > 0)
    ? wrapped
    : (raw as Record<string, Record<string, unknown>>);

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 text-center text-titan-metal/60">Loading sphere clocks...</div>;
  }

  const inner = CLOCK_CONFIG.filter(c => c.row === 'Inner');
  const outer = CLOCK_CONFIG.filter(c => c.row === 'Outer');

  return (
    <div className="bg-titan-card rounded-xl p-6">
      <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-4">Sphere Clocks · Schumann Resonance</h3>
      <div className="flex justify-between items-center">
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs text-titan-metal/40 uppercase">Inner</span>
          <div className="flex gap-4">
            {inner.map(c => <ClockCircle key={c.name} clock={c} data={(clocks[c.name] ?? {}) as Record<string, unknown>} />)}
          </div>
        </div>
        <div className="flex flex-col items-center gap-1 px-4">
          <span className="text-xs text-titan-haze/60 uppercase">Resonance</span>
          <div className="flex flex-col gap-1">
            {['spirit', 'mind', 'body'].map(layer => {
              const iKey = `inner_${layer}`;
              const oKey = `outer_${layer}`;
              const iData = (clocks[iKey] ?? {}) as Record<string, unknown>;
              const oData = (clocks[oKey] ?? {}) as Record<string, unknown>;
              const iStreak = (iData?.consecutive_balanced ?? iData?.streak ?? 0) as number;
              const oStreak = (oData?.consecutive_balanced ?? oData?.streak ?? 0) as number;
              const aligned = (iStreak as number) > 100 && (oStreak as number) > 100;
              return (
                <div key={layer} className="flex items-center gap-1">
                  <div className="h-px w-8" style={{
                    backgroundColor: aligned ? 'var(--titan-haze)' : 'var(--titan-metal)',
                    opacity: aligned ? 0.8 : 0.2,
                  }} />
                  <span className="text-xs font-mono" style={{ color: aligned ? 'var(--titan-haze)' : 'var(--titan-metal)', opacity: 0.6 }}>
                    {layer[0].toUpperCase()}
                  </span>
                  <div className="h-px w-8" style={{
                    backgroundColor: aligned ? 'var(--titan-haze)' : 'var(--titan-metal)',
                    opacity: aligned ? 0.8 : 0.2,
                  }} />
                </div>
              );
            })}
          </div>
        </div>
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs text-titan-metal/40 uppercase">Outer</span>
          <div className="flex gap-4">
            {outer.map(c => <ClockCircle key={c.name} clock={c} data={(clocks[c.name] ?? {}) as Record<string, unknown>} />)}
          </div>
        </div>
      </div>
    </div>
  );
}

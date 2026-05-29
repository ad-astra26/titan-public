'use client';

import { useNeuromodulators } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import InfoTooltip from '@/components/shared/InfoTooltip';

const MODULATORS = ['DA', '5HT', 'NE', 'ACh', 'Endorphin', 'GABA'];
const MOD_LABELS: Record<string, string> = {
  DA: 'Dopamine', '5HT': 'Serotonin', NE: 'Norepinephrine',
  ACh: 'Acetylcholine', Endorphin: 'Endorphin', GABA: 'GABA',
};

function Gauge({ name, mod }: { name: string; mod: Record<string, unknown> }) {
  const level = typeof mod?.level === 'number' ? mod.level : 0;
  const setpoint = typeof mod?.setpoint === 'number' ? mod.setpoint : 0.5;
  const sensitivity = typeof mod?.sensitivity === 'number' ? mod.sensitivity : 1;
  const delta = level - setpoint;
  const deltaIcon = Math.abs(delta) < 0.05 ? '≈' : delta > 0 ? '▲' : '▼';
  const deltaColor = Math.abs(delta) < 0.05 ? 'text-titan-metal' : delta > 0 ? 'text-titan-haze' : 'text-blue-400';

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-titan-metal/60">{MOD_LABELS[name] ?? name}</span>
        <span className={`text-xs font-mono ${deltaColor}`}>{deltaIcon} {delta >= 0 ? '+' : ''}{delta.toFixed(3)}</span>
      </div>
      <div className="relative h-4 bg-titan-bg rounded-sm overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-sm transition-all duration-500"
          style={{ width: `${Math.min(level, 1) * 100}%`, backgroundColor: 'var(--titan-haze)', opacity: 0.8 }}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-titan-metal"
          style={{ left: `${Math.min(setpoint, 1) * 100}%` }}
          title={`setpoint: ${setpoint.toFixed(3)}`}
        />
        <span className="absolute inset-0 flex items-center justify-end pr-1 text-xs font-mono text-titan-bg/70">
          {level.toFixed(3)}
        </span>
      </div>
      <div className="h-1.5 bg-titan-bg rounded-sm overflow-hidden" title={`sensitivity: ${sensitivity.toFixed(3)}`}>
        <div
          className="h-full rounded-sm bg-titan-pulse/40 transition-all duration-500"
          style={{ width: `${Math.min(sensitivity / 2, 1) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function NeuromodGauges() {
  const titanId = useTitanId();
  const { data, isLoading } = useNeuromodulators(titanId);
  const nm = (data ?? {}) as Record<string, unknown>;
  const modulators = (nm?.modulators ?? {}) as Record<string, Record<string, unknown>>;
  const emotion = nm?.current_emotion as string ?? '—';
  const confidence = typeof nm?.emotion_confidence === 'number' ? nm.emotion_confidence : 0;

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 text-center text-titan-metal/40">Loading neuromodulators...</div>;
  }

  return (
    <div className="bg-titan-card rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <InfoTooltip text="Six neurochemicals that shape Titan's behavior: Dopamine (motivation), Serotonin (mood stability), Norepinephrine (alertness), Acetylcholine (learning), Endorphin (satisfaction), GABA (calm/inhibition). Together they determine the current emotion.">
          <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider">Neuromodulators</h3>
        </InfoTooltip>
        <div className="flex items-center gap-2">
          <span className="text-titan-haze font-titan text-lg">{emotion}</span>
          <span className="font-mono text-xs text-titan-metal/40">conf {confidence.toFixed(3)}</span>
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {MODULATORS.map(name => (
          <Gauge key={name} name={name} mod={(modulators[name] ?? {}) as Record<string, unknown>} />
        ))}
      </div>
    </div>
  );
}

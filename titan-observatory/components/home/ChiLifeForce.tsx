'use client';

import { useChi } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import InfoTooltip from '@/components/shared/InfoTooltip';

const LAYER_COLORS: Record<string, { bar: string; label: string }> = {
  spirit: { bar: 'bg-titan-pulse', label: 'text-titan-pulse' },
  mind:   { bar: 'bg-titan-haze',  label: 'text-titan-haze' },
  body:   { bar: 'bg-titan-growth', label: 'text-titan-growth' },
};

function TrinityBar({ name, effective = 0, weight = 0, thinking = 0, feeling = 0, willing = 0 }: {
  name: string; effective?: number; weight?: number;
  thinking?: number; feeling?: number; willing?: number;
}) {
  const colors = LAYER_COLORS[name] ?? { bar: 'bg-titan-metal', label: 'text-titan-metal' };
  const pct = Math.round(effective * 100);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className={`text-xs font-mono uppercase tracking-wider ${colors.label}`}>{name}</span>
        <span className="text-xs font-mono text-titan-metal/60">{pct}%</span>
      </div>
      <div className="w-full h-2 bg-titan-bg rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${colors.bar} transition-all duration-1000`}
          style={{ width: `${pct}%`, opacity: 0.8 }}
        />
      </div>
      <div className="flex gap-3 text-[10px] font-mono text-titan-metal/40">
        <span>T {(thinking * 100).toFixed(0)}%</span>
        <span>F {(feeling * 100).toFixed(0)}%</span>
        <span>W {(willing * 100).toFixed(0)}%</span>
        <span className="ml-auto">w={weight.toFixed(2)}</span>
      </div>
    </div>
  );
}

export default function ChiLifeForce() {
  const titanId = useTitanId();
  const { data: chi } = useChi(titanId);

  if (!chi) {
    return (
      <div className="bg-titan-card rounded-xl p-4 animate-pulse">
        <div className="h-4 bg-titan-metal/10 rounded w-24 mb-3" />
        <div className="space-y-3">
          <div className="h-2 bg-titan-metal/10 rounded" />
          <div className="h-2 bg-titan-metal/10 rounded" />
          <div className="h-2 bg-titan-metal/10 rounded" />
        </div>
      </div>
    );
  }

  const totalPct = Math.round(chi.total * 100);
  const stateColor = chi.state === 'HEALTHY' ? 'text-titan-growth' :
                     chi.state === 'STRESSED' ? 'text-yellow-400' : 'text-red-400';

  return (
    <div className="bg-titan-card rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <InfoTooltip text="Chi measures Titan's overall vitality across Body (infrastructure health), Mind (learning capacity), and Spirit (identity coherence). T=Thinking, F=Feeling, W=Willing are sub-dimensions of each layer.">
            <h3 className="text-xs font-mono uppercase tracking-wider text-titan-metal/60">Chi Life Force</h3>
          </InfoTooltip>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono text-titan-haze">{totalPct}%</span>
            <span className={`text-xs font-mono ${stateColor}`}>{chi.state}</span>
          </div>
        </div>
        <div className="text-right">
          <span className="text-[10px] font-mono text-titan-metal/40 block">{chi.developmental_phase}</span>
          {chi.contemplation?.active && (
            <span className="text-[10px] font-mono text-titan-pulse">
              contemplating {chi.contemplation.conviction}/{chi.contemplation.conviction_threshold}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3">
        {(['spirit', 'mind', 'body'] as const).map((layer) => {
          const d = chi[layer];
          if (!d) return null;
          return (
            <TrinityBar
              key={layer}
              name={layer}
              effective={d.effective}
              weight={d.weight}
              thinking={d.thinking}
              feeling={d.feeling}
              willing={d.willing}
            />
          );
        })}
      </div>

      {chi.circulation > 0 && (
        <div className="mt-2 text-[10px] font-mono text-titan-metal/30 text-center">
          circulation: {chi.circulation.toFixed(4)}
        </div>
      )}
    </div>
  );
}

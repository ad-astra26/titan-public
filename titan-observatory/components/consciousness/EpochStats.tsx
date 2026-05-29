'use client';

interface Props {
  epochId: number;
  drift: number;
  curvature: number;
  density: number;
}

export default function EpochStats({ epochId, drift, curvature, density }: Props) {
  const curvLabel = curvature < -0.001 ? 'expanding' : curvature > 0.001 ? 'contracting' : 'stable';
  const curvColor = curvature < -0.001 ? 'text-blue-400' : curvature > 0.001 ? 'text-titan-haze' : 'text-titan-metal';

  const cards = [
    { label: 'Epoch', value: `#${epochId.toLocaleString()}`, sub: '', color: 'text-titan-haze' },
    { label: 'Drift', value: drift.toFixed(4), sub: drift < 0.01 ? 'stable' : 'drifting', color: drift < 0.01 ? 'text-titan-growth' : 'text-titan-haze' },
    { label: 'Curvature', value: curvature.toFixed(4), sub: curvLabel, color: curvColor },
    { label: 'Density', value: density.toFixed(4), sub: density > 0.8 ? 'rich' : density > 0.5 ? 'moderate' : 'sparse', color: 'text-titan-metal' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {cards.map(c => (
        <div key={c.label} className="bg-titan-card rounded-xl p-4 text-center">
          <span className="text-xs text-titan-metal/40 uppercase">{c.label}</span>
          <p className={`font-mono text-xl ${c.color} mt-1`}>{c.value}</p>
          {c.sub && <span className="text-xs text-titan-metal/40">{c.sub}</span>}
        </div>
      ))}
    </div>
  );
}

'use client';

import { useNeuromodulators } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const NODES = [
  { id: 'DA', label: 'DA', x: 200, y: 60, color: '#E5C79E' },
  { id: '5HT', label: '5-HT', x: 340, y: 120, color: '#4488FF' },
  { id: 'NE', label: 'NE', x: 340, y: 240, color: '#FF6644' },
  { id: 'ACh', label: 'ACh', x: 200, y: 300, color: '#77CCCC' },
  { id: 'Endorphin', label: 'End', x: 60, y: 240, color: '#FF88CC' },
  { id: 'GABA', label: 'GABA', x: 60, y: 120, color: '#9945FF' },
];

// Static coupling weights (from neuromodulator.py cross-coupling matrix)
const EDGES = [
  { from: '5HT', to: 'GABA', weight: 0.10, excitatory: true },
  { from: '5HT', to: 'NE', weight: 0.08, excitatory: false },
  { from: 'NE', to: 'DA', weight: 0.08, excitatory: true },
  { from: 'NE', to: 'GABA', weight: 0.12, excitatory: false },
  { from: 'Endorphin', to: 'GABA', weight: 0.15, excitatory: false },
  { from: 'Endorphin', to: 'NE', weight: 0.08, excitatory: false },
  { from: 'GABA', to: '5HT', weight: 0.10, excitatory: true },
  { from: 'GABA', to: 'NE', weight: 0.08, excitatory: false },
  { from: 'GABA', to: 'Endorphin', weight: 0.15, excitatory: false },
  { from: 'DA', to: 'NE', weight: 0.05, excitatory: true },
];

export default function CouplingNetwork() {
  const titanId = useTitanId();
  const { data } = useNeuromodulators(titanId);
  const modulators = ((data as Record<string, unknown>)?.modulators ?? {}) as Record<string, Record<string, unknown>>;

  const getLevel = (id: string) => {
    const m = modulators[id];
    return typeof m?.level === 'number' ? m.level : 0.5;
  };

  return (
    <div className="bg-titan-card rounded-xl p-6">
      <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-2">Cross-Coupling Network</h3>
      <svg viewBox="0 0 400 360" className="w-full max-w-md mx-auto">
        {/* Edges */}
        {EDGES.map((e, i) => {
          const from = NODES.find(n => n.id === e.from)!;
          const to = NODES.find(n => n.id === e.to)!;
          return (
            <line key={i}
              x1={from.x} y1={from.y} x2={to.x} y2={to.y}
              stroke={e.excitatory ? '#77CCCC' : '#FF4444'}
              strokeWidth={e.weight * 15}
              opacity={0.4}
            />
          );
        })}
        {/* Nodes */}
        {NODES.map(n => {
          const level = getLevel(n.id);
          const r = 12 + level * 18;
          return (
            <g key={n.id}>
              <circle cx={n.x} cy={n.y} r={r} fill={n.color} opacity={0.7}
                style={{ transition: 'r 0.5s, opacity 0.5s' }} />
              <circle cx={n.x} cy={n.y} r={r + 3} fill="none" stroke={n.color} strokeWidth={1} opacity={0.3} />
              <text x={n.x} y={n.y + 4} textAnchor="middle"
                fill="#0B0E14" fontSize="10" fontFamily="monospace" fontWeight="bold">
                {n.label}
              </text>
              <text x={n.x} y={n.y + r + 14} textAnchor="middle"
                fill="#8E9AAF" fontSize="9" fontFamily="monospace">
                {level.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="flex justify-center gap-6 mt-2 text-[10px] text-titan-metal/40">
        <span><span className="text-titan-growth">—</span> excitatory</span>
        <span><span className="text-red-400">—</span> inhibitory</span>
      </div>
    </div>
  );
}

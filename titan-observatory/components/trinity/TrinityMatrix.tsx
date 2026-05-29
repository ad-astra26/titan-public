'use client';

function heatColor(v: number): string {
  // blue(0) → white(0.5) → golden(1.0)
  if (v <= 0.5) {
    const t = v / 0.5;
    const r = Math.round(68 + t * (255 - 68));
    const g = Math.round(136 + t * (255 - 136));
    const b = Math.round(255 + t * (255 - 255));
    return `rgb(${r},${g},${b})`;
  }
  const t = (v - 0.5) / 0.5;
  const r = Math.round(255 + t * (229 - 255));
  const g = Math.round(255 + t * (199 - 255));
  const b = Math.round(255 + t * (158 - 255));
  return `rgb(${r},${g},${b})`;
}

function DimBar({ label, value }: { label: string; value: number }) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-titan-metal/60 w-8 font-mono">{label}</span>
      <div className="flex-1 h-3 bg-titan-bg rounded-sm overflow-hidden">
        <div className="h-full rounded-sm transition-all duration-500" style={{ width: `${w}%`, backgroundColor: heatColor(value) }} />
      </div>
      <span className="text-xs font-mono text-titan-metal/60 w-10 text-right">{value.toFixed(2)}</span>
    </div>
  );
}

function MiniHeatmap({ values, cols }: { values: number[]; cols: number }) {
  const rows = Math.ceil(values.length / cols);
  return (
    <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {Array.from({ length: rows * cols }, (_, i) => {
        const v = i < values.length ? values[i] : 0;
        return (
          <div key={i} className="h-3 rounded-sm transition-all duration-500" style={{ backgroundColor: i < values.length ? heatColor(v) : 'transparent', opacity: i < values.length ? 1 : 0 }} />
        );
      })}
    </div>
  );
}

interface TrinityMatrixProps {
  side: 'inner' | 'outer';
  data: Record<string, unknown>;
}

export default function TrinityMatrix({ side, data }: TrinityMatrixProps) {
  const label = side === 'inner' ? 'Inner Trinity' : 'Outer Trinity';
  const bodyKey = side === 'inner' ? 'body' : 'outer_body';
  const mindKey = side === 'inner' ? 'mind' : 'outer_mind';
  const spiritKey = side === 'inner' ? 'spirit' : 'outer_spirit';

  // API returns either number[] directly or {dims, values, center_dist} objects
  const extractTensor = (raw: unknown): number[] => {
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === 'object' && 'values' in (raw as Record<string, unknown>)) {
      const vals = (raw as Record<string, unknown>).values;
      return Array.isArray(vals) ? vals : [];
    }
    return [];
  };
  const bodyTensor = extractTensor(data?.[bodyKey] ?? data?.body_tensor);
  const mindTensor = extractTensor(data?.[mindKey] ?? data?.mind_tensor);
  const spiritTensor = extractTensor(data?.[spiritKey] ?? data?.spirit_tensor);

  const bodyDims = ['coh', 'mag', 'vel', 'dir', 'pol'];

  return (
    <div className="bg-titan-card rounded-xl p-4 flex flex-col gap-3">
      <span className="relative group inline-flex items-center">
        <h4 className="text-xs text-titan-metal/40 uppercase tracking-wider">{label}</h4>
        <span className="ml-1 inline-flex items-center justify-center w-3 h-3 rounded-full border border-titan-metal/15 text-[7px] text-titan-metal/30 cursor-help group-hover:border-titan-haze/40 group-hover:text-titan-haze/60 transition-colors">?</span>
        <span className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 pointer-events-none">
          <span className="block bg-titan-bg border border-titan-metal/20 rounded-lg px-3 py-2 text-[10px] text-titan-metal/70 leading-relaxed whitespace-normal min-w-[200px] max-w-[260px] shadow-lg">
            {side === 'inner'
              ? 'The Inner Self — felt sensations and conscious processing. Body=somatic (temperature, load, entropy), Mind=cognitive (vision, hearing, taste, smell, touch + 10 extended), Spirit=identity, purpose, and 45D consciousness.'
              : 'The Outer Self — how Titan acts in the world. Body=physical actions, Mind=decisions and plans, Spirit=creative expression and autonomous goals.'
            }
          </span>
        </span>
      </span>

      <div>
        <span className="text-xs text-titan-metal/60 font-mono">Body [{bodyTensor.length || 5}D]</span>
        <div className="mt-1 flex flex-col gap-1">
          {(bodyTensor.length > 0 ? bodyTensor : [0, 0, 0, 0, 0]).map((v, i) => (
            <DimBar key={i} label={bodyDims[i] ?? `${i}`} value={typeof v === 'number' ? v : 0} />
          ))}
        </div>
      </div>

      <div>
        <span className="text-xs text-titan-metal/60 font-mono">Mind [{mindTensor.length || 15}D]</span>
        <div className="mt-1">
          {mindTensor.length > 5 ? (
            <MiniHeatmap values={mindTensor.map(v => typeof v === 'number' ? v : 0)} cols={5} />
          ) : (
            <div className="flex flex-col gap-1">
              {(mindTensor.length > 0 ? mindTensor : [0, 0, 0, 0, 0]).map((v, i) => (
                <DimBar key={i} label={`m${i}`} value={typeof v === 'number' ? v : 0} />
              ))}
            </div>
          )}
        </div>
      </div>

      <div>
        <span className="text-xs text-titan-metal/60 font-mono">Spirit [{spiritTensor.length || 45}D]</span>
        <div className="mt-1">
          {spiritTensor.length > 5 ? (
            <MiniHeatmap values={spiritTensor.map(v => typeof v === 'number' ? v : 0)} cols={9} />
          ) : (
            <div className="h-12 bg-titan-bg rounded flex items-center justify-center">
              <span className="text-xs text-titan-metal/30">45D heatmap (awaiting full tensor)</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

'use client';

interface Props {
  topology: Record<string, unknown>;
  meta: number[];
}

// rFP_observatory_data_loading_v1 §3.2 Batch E (2026-04-26):
// extended TopologyPanel to render the 30D space-topology observables
// (6 layers × 5 metrics) alongside the legacy Volume / Curvature /
// Clusters summary. Backend coord.topology now carries:
//   { volume, curvature, cluster_count,
//     observables_30d:[30 floats],
//     observables_dict:{<layer>:{coherence,magnitude,velocity,direction,polarity}} }

// Backend ships `topology.parts` with the 6 trinity-layer LayerObservable
// derivation (per rFP_phase_c_substrate_observable_closure §2.1 / D-SPEC-80):
// inner_body / inner_mind / inner_spirit / outer_body / outer_mind /
// outer_spirit, each {coherence, magnitude, velocity, direction, polarity}
// computed from the corresponding daemon tensor slot. These are the input
// vectors to the TopologyEngine's pairwise-distance volume calc.
const LAYERS = [
  { key: 'inner_body',   label: 'iB', accent: 'text-titan-haze' },
  { key: 'inner_mind',   label: 'iM', accent: 'text-cyan-300' },
  { key: 'inner_spirit', label: 'iS', accent: 'text-violet-300' },
  { key: 'outer_body',   label: 'oB', accent: 'text-titan-haze/60' },
  { key: 'outer_mind',   label: 'oM', accent: 'text-cyan-300/60' },
  { key: 'outer_spirit', label: 'oS', accent: 'text-violet-300/60' },
] as const;

const METRICS = ['coherence', 'magnitude', 'velocity', 'direction', 'polarity'] as const;

function metricCell(value: number): { color: string; bar: number } {
  // direction/polarity can go negative (-1..1); coherence/magnitude/velocity are 0..1.
  const clamped = Math.max(-1, Math.min(1, value));
  const bar = Math.abs(clamped) * 100;
  const color = clamped < -0.05
    ? 'bg-rose-400/60'
    : clamped > 0.5
    ? 'bg-titan-haze'
    : clamped > 0.05
    ? 'bg-cyan-400/60'
    : 'bg-titan-metal/30';
  return { color, bar };
}

export default function TopologyPanel({ topology, meta }: Props) {
  // Backend ships `topology.values: [30 floats]` per Preamble G4 — WHOLE-10D
  // block lives at [20:30]: volume, curvature, density, mean_distance,
  // cross_layer_mirror, cluster_count, grounding_tension, matter_spirit_ratio,
  // willing_coherence, field_polarity. Read from values[] first; fall back to
  // legacy top-level keys for backward compatibility with older deployments.
  // Aligned with rFP_phase_c_substrate_observable_closure / D-SPEC-80.
  const values = Array.isArray(topology?.values) ? topology.values as number[] : [];
  const valFromIdx = (idx: number, legacyKey: string): number => {
    if (values.length >= 30 && typeof values[idx] === 'number') return Number(values[idx]);
    const legacy = (topology as Record<string, unknown> | undefined)?.[legacyKey];
    return typeof legacy === 'number' ? legacy : 0;
  };
  const volume = valFromIdx(20, 'volume');
  const curvature = valFromIdx(21, 'curvature');
  const clusterCount = valFromIdx(25, 'cluster_count');
  // Backend now ships `topology.parts: {<layer>: {coherence, magnitude, velocity, direction, polarity}}`
  // alongside the 30-float `values` array. Fall back to legacy `observables_dict` key for old shape.
  const observablesDict = ((topology?.parts ?? topology?.observables_dict ?? {})) as Record<string, Record<string, number>>;
  const has30D = LAYERS.some((l) => observablesDict[l.key]);
  const bodyScalar = meta?.[0] ?? 0;
  const mindScalar = meta?.[1] ?? 0;

  const curvLabel = curvature < -0.001 ? 'expanding' : curvature > 0.001 ? 'contracting' : 'stable';
  const curvColor = curvature < -0.001 ? 'text-blue-400' : curvature > 0.001 ? 'text-titan-haze' : 'text-titan-metal';

  return (
    <div className="bg-titan-card rounded-xl p-4 flex flex-col gap-4">
      <div className="flex flex-wrap gap-4 items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="rounded-full border border-titan-metal/20 transition-all duration-1000"
            style={{
              width: `${24 + volume * 40}px`,
              height: `${24 + volume * 40}px`,
              backgroundColor: 'rgba(229,199,158,0.1)',
            }}
          />
          <div>
            <span className="text-xs text-titan-metal/40 uppercase">Volume</span>
            <p className="font-mono text-sm text-titan-metal">{volume.toFixed(4)}</p>
          </div>
        </div>

        <div>
          <span className="text-xs text-titan-metal/40 uppercase">Curvature</span>
          <p className={`font-mono text-sm ${curvColor}`}>{curvature.toFixed(4)} · {curvLabel}</p>
        </div>

        <div>
          <span className="text-xs text-titan-metal/40 uppercase">Clusters</span>
          <p className="font-mono text-sm text-titan-metal">{clusterCount}</p>
        </div>

        <div className="flex items-center gap-4">
          <div>
            <span className="text-xs text-titan-metal/40">body_scalar</span>
            <div className="h-1.5 w-20 bg-titan-bg rounded-full overflow-hidden mt-1">
              <div className="h-full bg-titan-haze/60 rounded-full transition-all" style={{ width: `${bodyScalar * 100}%` }} />
            </div>
            <span className="font-mono text-xs text-titan-metal/60">{bodyScalar.toFixed(3)}</span>
          </div>
          <div>
            <span className="text-xs text-titan-metal/40">mind_scalar</span>
            <div className="h-1.5 w-20 bg-titan-bg rounded-full overflow-hidden mt-1">
              <div className="h-full bg-titan-haze/60 rounded-full transition-all" style={{ width: `${mindScalar * 100}%` }} />
            </div>
            <span className="font-mono text-xs text-titan-metal/60">{mindScalar.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {has30D && (
        <div className="border-t border-titan-metal/10 pt-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-titan-metal/40 uppercase tracking-wider">
              Space Topology · 30D
            </span>
            <span className="text-[10px] text-titan-metal/30">
              6 layers × 5 metrics — coherence, magnitude, velocity, direction, polarity
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-titan-metal/40">
                  <th className="text-left pl-1 pr-2"></th>
                  {METRICS.map((m) => (
                    <th key={m} className="text-left px-2 py-1 font-normal text-[10px] uppercase tracking-wide">
                      {m.slice(0, 3)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {LAYERS.map((layer) => {
                  const row = observablesDict[layer.key] ?? {};
                  return (
                    <tr key={layer.key} className="border-t border-titan-metal/5">
                      <td className={`py-1 pl-1 pr-2 ${layer.accent} font-semibold`}>{layer.label}</td>
                      {METRICS.map((m) => {
                        const v = typeof row[m] === 'number' ? row[m] : 0;
                        const cell = metricCell(v);
                        return (
                          <td key={m} className="px-2 py-1">
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-1.5 bg-titan-bg/60 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${cell.color}`}
                                  style={{ width: `${cell.bar}%` }}
                                />
                              </div>
                              <span className="text-titan-metal/60 w-12 text-right">
                                {v >= 0 ? v.toFixed(3) : v.toFixed(2)}
                              </span>
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

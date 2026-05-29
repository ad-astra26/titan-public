'use client';

const BODY_PARTS = ['iB', 'iM', 'iS', 'oB', 'oM', 'oS'];
const OBSERVABLES = ['coh', 'mag', 'vel', 'dir', 'pol'];

function heatColor(v: number): string {
  if (v <= 0.5) {
    const t = v / 0.5;
    return `rgb(${Math.round(68 + t * 187)},${Math.round(136 + t * 119)},${Math.round(255)})`;
  }
  const t = (v - 0.5) / 0.5;
  return `rgb(${Math.round(255 - t * 26)},${Math.round(255 - t * 56)},${Math.round(255 - t * 97)})`;
}

interface Props {
  observables: Record<string, Record<string, number>>;
}

export default function GlobalObservablesMatrix({ observables }: Props) {
  const partKeys = ['inner_body', 'inner_mind', 'inner_spirit', 'outer_body', 'outer_mind', 'outer_spirit'];
  const obsKeys = ['coherence', 'magnitude', 'velocity', 'direction', 'polarity'];

  return (
    <div className="bg-titan-card rounded-xl p-4">
      <h4 className="text-xs text-titan-metal/40 uppercase tracking-wider mb-3">Global Observables</h4>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-xs text-titan-metal/40 font-mono text-left pr-2" />
              {OBSERVABLES.map(o => (
                <th key={o} className="text-xs text-titan-metal/40 font-mono text-center px-1">{o}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {partKeys.map((pk, i) => {
              const partObs = observables?.[pk] ?? {};
              return (
                <tr key={pk}>
                  <td className="text-xs font-mono text-titan-metal/60 pr-2">{BODY_PARTS[i]}</td>
                  {obsKeys.map(ok => {
                    const v = typeof partObs[ok] === 'number' ? partObs[ok] : 0;
                    return (
                      <td key={ok} className="p-0.5">
                        <div
                          className="h-5 rounded-sm flex items-center justify-center transition-all duration-500"
                          style={{ backgroundColor: heatColor(v) }}
                          title={`${BODY_PARTS[i]}.${ok} = ${v.toFixed(3)}`}
                        >
                          <span className="text-[9px] font-mono text-titan-bg/80">{v.toFixed(2)}</span>
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
      <p className="text-[10px] text-titan-metal/30 mt-2 text-center">The nervous system seeing itself</p>
    </div>
  );
}

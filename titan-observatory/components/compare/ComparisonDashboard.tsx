'use client';

import {
  useStatus,
  useNeuromodulators,
  useNervousSystem,
  useVocabulary,
  useReasoning,
  useMetaReasoning,
  useDreaming,
  useChi,
  useReflexes,
} from '@/hooks/useTitanAPI';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

const TITANS: TitanId[] = ['T1', 'T2', 'T3'];
const TITAN_LABELS: Record<TitanId, string> = { T1: 'Titan 1', T2: 'Titan 2', T3: 'Titan 3' };
const TITAN_COLORS: Record<TitanId, string> = {
  T1: 'text-cyan-300 border-cyan-400/30 bg-cyan-400/5',
  T2: 'text-emerald-300 border-emerald-400/30 bg-emerald-400/5',
  T3: 'text-violet-300 border-violet-400/30 bg-violet-400/5',
};
const TITAN_BAR: Record<TitanId, string> = {
  T1: 'bg-cyan-400',
  T2: 'bg-emerald-400',
  T3: 'bg-violet-400',
};

function useTitanData(titanId: TitanId) {
  const status = useStatus(titanId);
  const neuromods = useNeuromodulators(titanId);
  const ns = useNervousSystem(titanId);
  const vocab = useVocabulary(titanId);
  const reasoning = useReasoning(titanId);
  const meta = useMetaReasoning(titanId);
  const dreaming = useDreaming(titanId);
  const chi = useChi(titanId);
  const reflexes = useReflexes(titanId);

  return { status, neuromods, ns, vocab, reasoning, meta, dreaming, chi, reflexes };
}

/** Comparison row — one metric across all 3 titans */
function CompareRow({
  label,
  values,
  format = 'number',
  maxVal,
}: {
  label: string;
  values: (number | string | null)[];
  format?: 'number' | 'percent' | 'raw';
  maxVal?: number;
}) {
  const numVals = values.map(v => typeof v === 'number' ? v : 0);
  const max = maxVal ?? Math.max(...numVals, 1);

  return (
    <div className="py-2 border-b border-titan-metal/5">
      <div className="flex items-center mb-1.5">
        <span className="text-xs text-titan-metal/60 w-40 shrink-0">{label}</span>
        <div className="flex-1 grid grid-cols-3 gap-3">
          {TITANS.map((tid, i) => {
            const val = values[i];
            const display = val === null ? '---'
              : format === 'percent' ? `${(typeof val === 'number' ? val * 100 : 0).toFixed(1)}%`
              : format === 'raw' ? String(val)
              : typeof val === 'number' ? val.toLocaleString()
              : String(val);

            return (
              <div key={tid} className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-titan-bg/60 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${TITAN_BAR[tid]} transition-all duration-700`}
                    style={{ width: `${max > 0 ? ((typeof val === 'number' ? val : 0) / max) * 100 : 0}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-titan-metal/70 w-16 text-right shrink-0">
                  {display}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function ComparisonDashboard() {
  const t1 = useTitanData('T1');
  const t2 = useTitanData('T2');
  const t3 = useTitanData('T3');
  const all = [t1, t2, t3];

  const anyLoading = all.some(t => t.status.isLoading);
  if (anyLoading) return <LoadingSkeleton lines={10} />;

  // Extract key metrics
  const epochs = all.map(t => (t.status.data?.epoch as unknown as number) ?? 0);
  const vocabSizes = all.map(t => t.vocab.data?.words?.length ?? 0);
  // Phase B.5 lean schema (2026-05-18): titanvm_registers.bin → {programs, ...}.
  // Pre-B.5 total_train_steps + total_transitions retired. Compare surface
  // shifted to Σ total_updates (cumulative learning) + Σ fire_count.
  const nsUpdates = all.map(t =>
    Object.values(t.ns.data?.programs ?? {})
      .reduce((s, p) => s + (p.total_updates ?? 0), 0),
  );
  const nsFires = all.map(t =>
    Object.values(t.ns.data?.programs ?? {})
      .reduce((s, p) => s + (p.fire_count ?? 0), 0),
  );
  const reasonChains = all.map(t => t.reasoning.data?.total_chains ?? 0);
  const reasonConclusions = all.map(t => t.reasoning.data?.total_conclusions ?? 0);
  const metaChains = all.map(t => t.meta.data?.total_chains ?? 0);
  const metaWisdom = all.map(t => t.meta.data?.total_wisdom_saved ?? 0);
  const metaAvgReward = all.map(t => t.meta.data?.avg_reward ?? 0);
  const sleepCycles = all.map(t => (t.dreaming.data?.cycle_count ?? 0) as number);
  const chiTotals = all.map(t => (t.chi.data?.total ?? 0) as number);
  const reflexFires = all.map(t => (t.reflexes.data?.stats_24h?.total_fires ?? 0) as number);

  // Neuromods
  const neuroKeys = ['DA', '5HT', 'NE', 'ACh', 'Endorphin', 'GABA'];
  const neuromodValues = neuroKeys.map(key =>
    all.map(t => {
      const nm = t.neuromods.data as Record<string, unknown> | undefined;
      if (!nm) return 0;
      const mods = (nm.modulators ?? nm) as Record<string, unknown>;
      const mod = mods[key] as Record<string, number> | undefined;
      return mod?.level ?? mod?.value ?? 0;
    })
  );

  // Top meta-reasoning primitives per titan
  const metaPrimitives = all.map(t => {
    const pc = t.meta.data?.primitive_counts ?? {};
    const sorted = Object.entries(pc).sort(([, a], [, b]) => b - a);
    return sorted.length > 0 ? sorted[0][0] : '---';
  });

  return (
    <div className="flex flex-col gap-5">
      {/* Titan headers */}
      <div className="flex items-center">
        <div className="w-40 shrink-0" />
        <div className="flex-1 grid grid-cols-3 gap-3">
          {TITANS.map(tid => (
            <div
              key={tid}
              className={`rounded-lg border p-2 text-center ${TITAN_COLORS[tid]}`}
            >
              <span className="text-sm font-semibold">{TITAN_LABELS[tid]}</span>
              <div className="flex items-center justify-center gap-1.5 mt-0.5">
                <span className={`w-1.5 h-1.5 rounded-full ${
                  all[TITANS.indexOf(tid)].status.data ? 'bg-emerald-400' : 'bg-red-400'
                }`} />
                <span className="text-[10px] opacity-60">
                  {all[TITANS.indexOf(tid)].dreaming.data?.is_dreaming ? 'dreaming' : 'awake'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Core Development */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-2">Core Development</h3>
        <CompareRow label="Epoch" values={epochs} />
        <CompareRow label="Vocabulary" values={vocabSizes} />
        <CompareRow label="NS Total Updates" values={nsUpdates} />
        <CompareRow label="NS Total Fires" values={nsFires} />
        <CompareRow label="Sleep Cycles" values={sleepCycles} />
        <CompareRow label="Chi Total" values={chiTotals} format="percent" maxVal={1} />
      </div>

      {/* Reasoning */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-2">Reasoning & Meta-Cognition</h3>
        <CompareRow label="Reasoning Chains" values={reasonChains} />
        <CompareRow label="Conclusions" values={reasonConclusions} />
        <CompareRow label="Meta-Reasoning Chains" values={metaChains} />
        <CompareRow label="Wisdom Saved" values={metaWisdom} />
        <CompareRow label="Meta Avg Reward" values={metaAvgReward} format="percent" maxVal={1} />
        <CompareRow label="Top Primitive" values={metaPrimitives} format="raw" />
        <CompareRow label="Reflex Fires (24h)" values={reflexFires} />
      </div>

      {/* Neuromodulators */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-2">Neuromodulator Profiles</h3>
        {neuroKeys.map((key, i) => (
          <CompareRow key={key} label={key} values={neuromodValues[i]} format="percent" maxVal={1} />
        ))}
      </div>
    </div>
  );
}

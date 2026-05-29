'use client';

import { useMemo } from 'react';
import { useVocabulary, useLanguageGrounding } from '@/hooks/useTitanAPI';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

const PHASE_COLORS: Record<string, string> = {
  first_word: '#8E9AAF',
  contextual: '#FF8844',
  producible: '#44CC66',
};
const PHASE_LABELS: Record<string, string> = {
  first_word: 'First Word',
  contextual: 'Contextual',
  producible: 'Producible',
};

export default function PhaseDistribution({ titanId }: { titanId?: TitanId }) {
  const { data: vocab, isLoading: vLoading } = useVocabulary(titanId);
  const { data: grounding, isLoading: gLoading } = useLanguageGrounding(titanId);

  const { phaseCounts, phaseOrder } = useMemo(() => {
    if (!vocab?.words) return { phaseCounts: {}, phaseOrder: [] };
    const counts: Record<string, number> = {};
    vocab.words.forEach((w) => {
      const phase = w.learning_phase || 'unknown';
      counts[phase] = (counts[phase] || 0) + 1;
    });
    // Sort: producible last (highest), others alphabetical
    const order = Object.keys(counts).sort((a, b) => {
      if (a === 'producible') return 1;
      if (b === 'producible') return -1;
      return a.localeCompare(b);
    });
    return { phaseCounts: counts, phaseOrder: order };
  }, [vocab]);

  const typeCounts = useMemo(() => {
    if (grounding?.word_types) return grounding.word_types;
    if (!vocab?.words) return {};
    const counts: Record<string, number> = {};
    vocab.words.forEach((w) => {
      const t = w.word_type || 'unknown';
      counts[t] = (counts[t] || 0) + 1;
    });
    return counts;
  }, [vocab, grounding]);

  if (vLoading && gLoading) return <LoadingSkeleton lines={6} />;

  const totalWords = vocab?.words?.length ?? 0;
  const sortedTypes = Object.entries(typeCounts).sort(([, a], [, b]) => b - a);
  const maxTypeCount = Math.max(...Object.values(typeCounts), 1);

  return (
    <div className="flex flex-col gap-4">
      {/* Learning Phase Distribution */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-3">Learning Phase Distribution</h3>
        <div className="flex flex-col gap-2">
          {phaseOrder.map((phase) => {
            const count = phaseCounts[phase] || 0;
            const pct = totalWords > 0 ? (count / totalWords) * 100 : 0;
            return (
              <div key={phase}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: PHASE_COLORS[phase] }}
                    />
                    <span className="text-xs text-titan-metal/70">{PHASE_LABELS[phase] || phase}</span>
                  </div>
                  <span className="text-xs font-mono text-titan-metal/50">
                    {count} <span className="text-titan-metal/30">({pct.toFixed(0)}%)</span>
                  </span>
                </div>
                <div className="h-2 bg-titan-bg rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: PHASE_COLORS[phase] || '#8E9AAF',
                      opacity: 0.8,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Word Type Distribution */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-3">Word Type Distribution</h3>
        <div className="flex flex-col gap-1.5">
          {sortedTypes.map(([type, count]) => {
            const pct = (count / maxTypeCount) * 100;
            return (
              <div key={type} className="flex items-center gap-2">
                <span className="text-[11px] text-titan-metal/60 w-20 truncate text-right">{type}</span>
                <div className="flex-1 h-2 bg-titan-bg rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-titan-haze/50 transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-[11px] font-mono text-titan-metal/40 w-8 text-right">{count}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

'use client';

import { useLanguageGrounding, useVocabulary } from '@/hooks/useTitanAPI';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

export default function GroundingStats({ titanId }: { titanId?: TitanId }) {
  const { data: grounding, isLoading: gLoading } = useLanguageGrounding(titanId);
  const { data: vocab, isLoading: vLoading } = useVocabulary(titanId);

  if (gLoading && vLoading) return <LoadingSkeleton lines={2} />;

  const totalWords = grounding?.total_words ?? vocab?.words?.length ?? 0;
  const producible = grounding?.producible ?? vocab?.words?.filter(w => w.learning_phase === 'producible').length ?? 0;
  const grounded = grounding?.grounded ?? 0;
  const groundingRate = grounding?.grounding_rate ?? (producible > 0 ? grounded / producible : 0);
  const avgConf = grounding?.avg_confidence ?? 0;

  return (
    <div className="flex flex-col gap-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Total Words" value={totalWords} accent="haze" />
        <MetricCard label="Producible" value={producible} accent="growth" />
        <MetricCard label="Grounded" value={grounded} accent="pulse" />
        <MetricCard
          label="Grounding Rate"
          value={`${(groundingRate * 100).toFixed(0)}%`}
          accent="haze"
        />
      </div>

      {/* Top grounded words */}
      {grounding?.top_grounded && grounding.top_grounded.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            Top Grounded Words
            <span className="ml-2 text-xs font-normal text-titan-metal/40">
              avg confidence: {(avgConf * 100).toFixed(0)}%
            </span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {grounding.top_grounded.slice(0, 12).map((w) => (
              <div
                key={w.word}
                className="flex items-center justify-between bg-titan-bg/50 rounded-lg px-3 py-2"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-sm font-mono text-titan-haze truncate">&ldquo;{w.word}&rdquo;</span>
                  <span className="text-[10px] text-titan-metal/40 shrink-0">{w.word_type}</span>
                </div>
                <div className="flex items-center gap-2 shrink-0 ml-2">
                  {/* Sensory context badges */}
                  {w.sensory_contexts?.slice(0, 2).map((ctx) => (
                    <span
                      key={ctx}
                      className="text-[9px] px-1.5 py-0.5 rounded-full bg-titan-pulse/10 text-titan-pulse/70 border border-titan-pulse/20"
                    >
                      {ctx}
                    </span>
                  ))}
                  {/* Confidence bar */}
                  <div className="w-12 h-1.5 bg-titan-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-titan-growth"
                      style={{ width: `${(w.confidence ?? 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-titan-metal/50 font-mono w-8 text-right">
                    {(w.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

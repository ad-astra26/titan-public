'use client';

import { useCompositions } from '@/hooks/useTitanAPI';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

export default function CompositionsList({ titanId }: { titanId?: TitanId }) {
  const { data, isLoading } = useCompositions(titanId);

  if (isLoading) return <LoadingSkeleton lines={4} />;

  const compositions = data?.recent ?? [];
  const total = data?.total_compositions ?? 0;
  const latest = data?.latest;

  return (
    <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-titan-haze mb-3">
        Recent Compositions
        <span className="ml-2 text-xs font-normal text-titan-metal/40">
          {total} total
        </span>
      </h3>

      {latest && (
        <div className="bg-titan-haze/5 border border-titan-haze/20 rounded-lg px-3 py-2 mb-3">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] text-titan-haze/60 uppercase tracking-wider">Latest</span>
            <span className="text-[10px] font-mono text-titan-metal/40">L{latest.level}</span>
            <span className="text-[10px] font-mono text-titan-metal/40">
              {(latest.confidence * 100).toFixed(0)}% conf
            </span>
          </div>
          <p className="text-sm text-titan-metal italic">&ldquo;{latest.sentence}&rdquo;</p>
        </div>
      )}

      <div className="flex flex-col gap-1.5 max-h-[300px] overflow-y-auto scrollbar-thin">
        {compositions.length === 0 && !latest && (
          <p className="text-xs text-titan-metal/30 text-center py-4">No compositions yet</p>
        )}
        {compositions.map((c, i) => (
          <div
            key={`${c.sentence}-${i}`}
            className="flex items-start gap-2 bg-titan-bg/40 rounded-lg px-3 py-2"
          >
            <p className="text-xs text-titan-metal flex-1 italic">&ldquo;{c.sentence}&rdquo;</p>
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-titan-bg border border-titan-metal/10 text-titan-metal/50">
                L{c.level}
              </span>
              <span className="text-[10px] font-mono text-titan-metal/40">
                {(c.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

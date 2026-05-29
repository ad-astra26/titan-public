'use client';

import { useState } from 'react';
import { useResearch } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

export default function ResearchTab() {
  const titanId = useTitanId();
  const { data: research, isLoading } = useResearch(titanId);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (isLoading) return <LoadingSkeleton lines={6} />;

  const topics = research?.recent_topics ?? research?.topics ?? [];
  const sourceDistribution = research?.source_distribution ?? {};

  return (
    <div className="flex flex-col gap-4">
      {/* Source distribution badges */}
      {Object.keys(sourceDistribution).length > 0 && (
        <div className="flex items-center gap-3">
          <span className="text-xs text-titan-metal/50">Sources used:</span>
          {Object.entries(sourceDistribution).map(([src, count]) => (
            <span
              key={src}
              className={`text-xs px-2 py-1 rounded-lg ${
                src === 'Web'
                  ? 'bg-titan-growth/20 text-titan-growth'
                  : src === 'X'
                    ? 'bg-blue-500/20 text-blue-400'
                    : 'bg-titan-haze/20 text-titan-haze'
              }`}
            >
              {src}: {count as number}
            </span>
          ))}
        </div>
      )}

      {/* Topics list */}
      {topics.length === 0 ? (
        <p className="text-xs text-titan-metal/40 text-center py-8">
          No research topics yet. Titan researches autonomously when knowledge gaps are detected.
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {topics.map((topic, i) => {
            const t = topic as unknown as Record<string, unknown>;
            const query = (t.topic || t.query || '') as string;
            const summary = (t.summary || t.distilled || '') as string;
            const sources = (t.sources || []) as string[];
            const urls = (t.urls || []) as string[];
            const timestamp = (t.timestamp || '') as string;

            return (
              <button
                key={i}
                onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4 text-left hover:border-titan-haze/20 transition-colors"
              >
                <p className="text-sm text-titan-metal/80 font-medium leading-snug">
                  {query}
                </p>
                <div className="flex items-center gap-2 mt-2 flex-wrap">
                  {sources.map((src) => (
                    <span
                      key={src}
                      className={`text-[10px] px-1.5 py-0.5 rounded ${
                        src === 'Web'
                          ? 'bg-titan-growth/20 text-titan-growth'
                          : src === 'X'
                            ? 'bg-blue-500/20 text-blue-400'
                            : 'bg-titan-haze/20 text-titan-haze'
                      }`}
                    >
                      {src}
                    </span>
                  ))}
                  {timestamp && (
                    <span className="text-[10px] text-titan-metal/30 ml-auto">
                      {typeof timestamp === 'string' && timestamp.includes('T')
                        ? timestamp.replace('T', ' ').slice(0, 19) + ' UTC'
                        : timestamp}
                    </span>
                  )}
                </div>
                {expandedIdx === i && (
                  <div className="mt-3 border-t border-titan-metal/10 pt-3 space-y-2">
                    {summary && (
                      <p className="text-xs text-titan-metal/60 leading-relaxed">{summary}</p>
                    )}
                    {urls.length > 0 && (
                      <div className="text-[10px] text-titan-metal/40 space-y-0.5">
                        {urls.map((url, ui) => (
                          <div key={ui} className="truncate">{url}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

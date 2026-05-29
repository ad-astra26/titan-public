'use client';

import { useState } from 'react';
import { ResearchTopic } from '@/lib/types';
import { formatTimestamp } from '@/lib/formatters';

interface ResearchTopicsProps {
  topics: ResearchTopic[];
}

export default function ResearchTopics({ topics }: ResearchTopicsProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (topics.length === 0) {
    return (
      <p className="text-xs text-titan-metal/40 text-center py-8">
        No research topics yet
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {topics.map((topic, i) => (
        <button
          key={i}
          onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
          className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4 text-left hover:border-titan-haze/20 transition-colors"
        >
          <p className="text-sm text-titan-metal/80 font-medium">
            {topic.query}
          </p>
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            {topic.sources.map((src) => (
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
            <span className="text-[10px] text-titan-metal/30 ml-auto">
              {formatTimestamp(topic.timestamp)}
            </span>
          </div>
          {expandedIdx === i && topic.distilled && (
            <p className="text-xs text-titan-metal/60 mt-3 leading-relaxed border-t border-titan-metal/10 pt-3">
              {topic.distilled}
            </p>
          )}
        </button>
      ))}
    </div>
  );
}

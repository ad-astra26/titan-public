'use client';

import { useQuery } from '@tanstack/react-query';
import { titanFetch, tierQueryOptions, API_BASE } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';
import type { TitanId } from '@/lib/api';

interface FeedEntry {
  ts: number;
  category: string;
  narrative: string;
  subtitle: string;
  media_url?: string;
  media_type?: string;
  details?: Record<string, unknown>;
}

const CREATIVE_CATEGORIES = new Set(['speech', 'creation', 'hormone', 'dream']);

const CATEGORY_STYLES: Record<string, { color: string; border: string; bg: string; icon: string }> = {
  speech:   { color: 'text-amber-300',  border: 'border-amber-400/15',  bg: 'bg-amber-950/20',  icon: 'M2 3a1 1 0 011-1h10a1 1 0 011 1v7a1 1 0 01-1 1H5l-3 3V3z' },
  creation: { color: 'text-rose-400',   border: 'border-rose-400/15',   bg: 'bg-rose-950/15',   icon: 'M8 1l1.5 4.5L14 7l-4.5 1.5L8 13l-1.5-4.5L2 7l4.5-1.5z' },
  hormone:  { color: 'text-orange-400', border: 'border-orange-400/15', bg: 'bg-orange-950/15', icon: '' },
  dream:    { color: 'text-indigo-300', border: 'border-indigo-400/15', bg: 'bg-indigo-950/20', icon: '' },
};

function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 0) return 'now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export default function NarrativeTimeline({ titanId: propTitanId }: { titanId?: TitanId }) {
  const hookTitanId = useTitanId();
  const titanId = propTitanId ?? hookTitanId;

  const { data, isLoading } = useQuery({
    ...tierQueryOptions('slow', ['narrated-feed-creative'], titanId),
    queryFn: () => titanFetch<{ items: FeedEntry[]; count: number }>('/v6/expression/narrated-feed?limit=60', { titan: titanId }),
  });

  const items = (data?.items ?? []).filter(e => CREATIVE_CATEGORIES.has(e.category));

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map(i => <div key={i} className="bg-titan-card/30 rounded-xl h-16 animate-pulse" />)}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="bg-titan-card/30 rounded-xl p-6 text-center">
        <p className="text-xs text-titan-metal/40">No creative events yet — Titan is building inner pressure...</p>
      </div>
    );
  }

  return (
    <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-titan-haze mb-3">
        Creative Narrative
        <span className="ml-2 text-xs font-normal text-titan-metal/40">
          {items.length} events (speech, creation, hormonal, dreams)
        </span>
      </h3>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-[11px] top-2 bottom-2 w-px bg-titan-metal/10" />

        <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
          {items.slice(0, 30).map((entry, i) => {
            const style = CATEGORY_STYLES[entry.category] ?? CATEGORY_STYLES.creation;
            const isSpeech = entry.category === 'speech';
            const hasMedia = entry.media_url && entry.media_type === 'art';

            return (
              <div key={`${entry.ts}-${i}`} className="flex items-start gap-3 relative">
                {/* Timeline dot */}
                <div className={`w-[23px] h-[23px] rounded-full ${style.bg} border ${style.border} flex items-center justify-center shrink-0 z-10`}>
                  <div className={`w-2 h-2 rounded-full ${style.color.replace('text-', 'bg-')} opacity-60`} />
                </div>

                {/* Content */}
                <div className={`flex-1 min-w-0 rounded-lg px-3 py-2 ${style.bg} border ${style.border} transition-all hover:brightness-110`}>
                  {isSpeech ? (
                    <p className="text-xs text-titan-haze font-medium italic leading-relaxed">
                      &ldquo;{entry.narrative}&rdquo;
                    </p>
                  ) : (
                    <p className="text-xs text-titan-metal/80 leading-relaxed">{entry.narrative}</p>
                  )}
                  <div className="flex items-center justify-between mt-1">
                    <span className={`text-[10px] ${style.color} opacity-60`}>{entry.subtitle}</span>
                    <span className="text-[10px] text-titan-metal/30">{timeAgo(entry.ts)}</span>
                  </div>
                </div>

                {/* Art thumbnail */}
                {hasMedia && (
                  <div className="w-10 h-10 rounded-lg overflow-hidden shrink-0 bg-titan-bg border border-titan-metal/10">
                    <img
                      src={`${API_BASE}${entry.media_url}`}
                      alt="art"
                      className="w-full h-full object-cover"
                      loading="lazy"
                      onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

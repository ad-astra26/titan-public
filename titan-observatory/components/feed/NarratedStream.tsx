'use client';

import { useQuery } from '@tanstack/react-query';
import { titanFetch, v4Fetch, API_BASE } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';

interface FeedEntry {
  ts: number;
  category: string;
  narrative: string;
  subtitle: string;
  media_url?: string;
  media_type?: string;
  details?: Record<string, unknown>;
}

// SVG icons for each category
function IconSpeech() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M2 3a1 1 0 011-1h10a1 1 0 011 1v7a1 1 0 01-1 1H5l-3 3V3z" fill="currentColor" opacity="0.8"/>
    </svg>
  );
}
function IconCreation() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M8 1l1.5 4.5L14 7l-4.5 1.5L8 13l-1.5-4.5L2 7l4.5-1.5z" fill="currentColor" opacity="0.8"/>
    </svg>
  );
}
function IconAgency() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <polygon points="3,2 13,8 3,14" fill="currentColor" opacity="0.7"/>
    </svg>
  );
}
function IconConsciousness() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.7"/>
      <circle cx="8" cy="8" r="2" fill="currentColor" opacity="0.6"/>
    </svg>
  );
}
function IconDream() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M12 4c0 4-4 8-8 8a7 7 0 008-8z" fill="currentColor" opacity="0.7"/>
    </svg>
  );
}
function IconNeurology() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="5" fill="currentColor" opacity="0.15"/>
      <path d="M4 8h2l1-3 2 6 1-3h2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.8"/>
    </svg>
  );
}
function IconResonance() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.9"/>
      <circle cx="8" cy="8" r="5.5" stroke="currentColor" strokeWidth="0.8" fill="none" opacity="0.5"/>
      <circle cx="8" cy="8" r="7.5" stroke="currentColor" strokeWidth="0.5" fill="none" opacity="0.3"/>
    </svg>
  );
}
function IconPulse() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.2" fill="none" opacity="0.7"/>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="0.8" fill="none" opacity="0.4"/>
    </svg>
  );
}
function IconHormone() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M8 2c0 3-3 4-3 7a3.5 3.5 0 007 0c0-3-3-4-3-7z" fill="currentColor" opacity="0.6"/>
      <path d="M8 6c0 1.5-1.5 2-1.5 3.5a1.8 1.8 0 003.5 0c0-1.5-1.5-2-1.5-3.5z" fill="currentColor" opacity="0.9"/>
    </svg>
  );
}

const CATEGORY_CONFIG: Record<string, {
  Icon: () => JSX.Element;
  color: string;
  borderColor: string;
  bgColor: string;
}> = {
  speech:        { Icon: IconSpeech,        color: 'text-amber-300',   borderColor: 'border-amber-400/15',   bgColor: 'bg-amber-950/20' },
  creation:      { Icon: IconCreation,      color: 'text-rose-400',    borderColor: 'border-rose-400/15',    bgColor: 'bg-rose-950/15' },
  agency:        { Icon: IconAgency,        color: 'text-emerald-400', borderColor: 'border-emerald-400/15', bgColor: 'bg-emerald-950/15' },
  consciousness: { Icon: IconConsciousness, color: 'text-violet-400',  borderColor: 'border-violet-400/15',  bgColor: 'bg-violet-950/15' },
  dream:         { Icon: IconDream,         color: 'text-indigo-300',  borderColor: 'border-indigo-400/15',  bgColor: 'bg-indigo-950/20' },
  neurology:     { Icon: IconNeurology,     color: 'text-cyan-400',    borderColor: 'border-cyan-400/15',    bgColor: 'bg-cyan-950/15' },
  resonance:     { Icon: IconResonance,     color: 'text-yellow-300',  borderColor: 'border-yellow-400/20',  bgColor: 'bg-yellow-950/20' },
  pulse:         { Icon: IconPulse,         color: 'text-blue-400',    borderColor: 'border-blue-400/15',    bgColor: 'bg-blue-950/15' },
  hormone:       { Icon: IconHormone,       color: 'text-orange-400',  borderColor: 'border-orange-400/15',  bgColor: 'bg-orange-950/15' },
};

function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 0) return 'now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function FeedItem({ entry }: { entry: FeedEntry }) {
  const cfg = CATEGORY_CONFIG[entry.category] || CATEGORY_CONFIG.consciousness;
  const { Icon } = cfg;
  const isSpeech = entry.category === 'speech';
  const hasMedia = entry.media_url && entry.media_type === 'art';

  return (
    <div className={`rounded-xl px-4 py-3 border ${cfg.borderColor} ${cfg.bgColor} transition-all hover:brightness-110`}>
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={`mt-0.5 shrink-0 ${cfg.color}`}>
          <Icon />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {isSpeech ? (
            <p className="text-sm text-titan-haze font-medium italic leading-relaxed">
              &ldquo;{entry.narrative}&rdquo;
            </p>
          ) : (
            <p className="text-xs text-titan-metal/80 leading-relaxed">
              {entry.narrative}
            </p>
          )}

          <div className="flex items-center justify-between mt-1.5">
            <span className={`text-[10px] ${cfg.color} opacity-60`}>{entry.subtitle}</span>
            <span className="text-[10px] text-titan-metal/30">{timeAgo(entry.ts)}</span>
          </div>
        </div>

        {/* Optional thumbnail */}
        {hasMedia && (
          <div className="w-12 h-12 rounded-lg overflow-hidden shrink-0 bg-titan-bg">
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
    </div>
  );
}

export default function NarratedStream() {
  const titanId = useTitanId();
  const { data, isLoading } = useQuery({
    queryKey: ['narrated-feed', titanId],
    queryFn: () => v4Fetch<{ items: FeedEntry[]; count: number }>('narrated-feed', { titan: titanId, extraQuery: 'limit=40' }),
    refetchInterval: 15000,
    retry: 2,
  });

  const items = data?.items ?? [];

  // Count by category for quick summary
  const counts = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.category] = (acc[item.category] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-2">
      {/* Category summary pills */}
      {Object.keys(counts).length > 0 && (
        <div className="flex gap-1.5 flex-wrap mb-1">
          {Object.entries(counts).map(([cat, count]) => {
            const cfg = CATEGORY_CONFIG[cat] || CATEGORY_CONFIG.consciousness;
            return (
              <span key={cat} className={`text-[9px] px-2 py-0.5 rounded-full ${cfg.color} ${cfg.bgColor} border ${cfg.borderColor}`}>
                {cat} {count}
              </span>
            );
          })}
        </div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="bg-titan-card/30 rounded-xl h-20 animate-pulse" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="bg-titan-card/30 rounded-xl p-8 text-center">
          <p className="text-sm text-titan-metal/40">Titan is quietly observing...</p>
        </div>
      ) : (
        items.map((item, idx) => (
          <FeedItem key={`${item.ts}-${item.category}-${idx}`} entry={item} />
        ))
      )}
    </div>
  );
}

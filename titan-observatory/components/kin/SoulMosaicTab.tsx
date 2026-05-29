'use client';

import { useEffect, useState, useMemo } from 'react';
import { titanFetch } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

// Same-origin relative URL (empty base) per lib/api.ts:_resolveApiBase contract.
// Using NEXT_PUBLIC_TITAN_API_URL directly in a 'use client' component triggers
// CORS against https://iamtitan.tech from localhost dev sessions.
const API_BASE = '';

interface MosaicItem {
  type: 'speak' | 'art' | 'music' | 'social' | 'agency' | 'consciousness' | 'dream' | 'pulse' | 'program';
  ts: number;
  title: string;
  detail?: string;
  mediaUrl?: string;
  size: 'sm' | 'md' | 'lg';
}

const ICON_MAP: Record<string, string> = {
  speak: '\uD83D\uDDE3\uFE0F',
  art: '\uD83C\uDFA8',
  music: '\uD83C\uDFB5',
  social: '\uD83D\uDCE1',
  agency: '\uD83E\uDDE0',
  consciousness: '\u2728',
  dream: '\uD83C\uDF19',
  pulse: '\uD83D\uDFE2',
  program: '\u26A1',
};

const COLOR_MAP: Record<string, string> = {
  speak: 'border-titan-haze/30 bg-titan-haze/5',
  art: 'border-violet-500/30 bg-violet-500/5',
  music: 'border-pink-400/30 bg-pink-400/5',
  social: 'border-cyan-400/30 bg-cyan-400/5',
  agency: 'border-amber-400/30 bg-amber-400/5',
  consciousness: 'border-yellow-400/30 bg-yellow-400/5',
  dream: 'border-indigo-400/30 bg-indigo-400/5',
  pulse: 'border-emerald-400/30 bg-emerald-400/5',
  program: 'border-emerald-400/20 bg-emerald-400/5',
};

const TYPE_LABELS: Record<string, string> = {
  speak: 'Speech',
  art: 'Art',
  music: 'Music',
  social: 'Social',
  agency: 'Reasoning',
  consciousness: 'Consciousness',
  dream: 'Dream',
  pulse: 'Pulse',
  program: 'Program',
};

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  const month = d.toLocaleString('en', { month: 'short' });
  const day = d.getDate();
  const h = d.getHours().toString().padStart(2, '0');
  const m = d.getMinutes().toString().padStart(2, '0');
  return `${month} ${day} ${h}:${m}`;
}

function ArtCard({ item }: { item: MosaicItem }) {
  const [imgError, setImgError] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const imgSrc = item.mediaUrl ? `${API_BASE}${item.mediaUrl}` : null;

  return (
    <div className="border border-violet-500/30 bg-violet-500/5 rounded-xl overflow-hidden break-inside-avoid mb-3 hover:brightness-110 transition-all">
      {imgSrc && !imgError ? (
        <div className="relative cursor-pointer" onClick={() => setExpanded(!expanded)}>
          <img
            src={imgSrc}
            alt={item.title}
            loading="lazy"
            className={`w-full object-cover transition-all ${expanded ? 'max-h-[500px]' : 'max-h-[200px]'}`}
            onError={() => setImgError(true)}
          />
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-3">
            <p className="text-xs text-white/90 font-medium">{item.title}</p>
          </div>
        </div>
      ) : (
        <div className="p-4">
          <span className="text-2xl">{ICON_MAP.art}</span>
          <p className="text-xs text-titan-haze/80 mt-1">{item.title}</p>
        </div>
      )}
      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-[9px] text-titan-metal/30 uppercase tracking-wider">art</span>
        <span className="text-[9px] text-titan-metal/30 font-mono">{formatTime(item.ts)}</span>
      </div>
    </div>
  );
}

function SpeechCard({ item }: { item: MosaicItem }) {
  return (
    <div className="border border-titan-haze/30 bg-titan-haze/5 rounded-xl p-4 break-inside-avoid mb-3 hover:brightness-110 transition-all">
      <div className="flex items-start gap-2">
        <span className="text-lg shrink-0 mt-0.5">{ICON_MAP.speak}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-titan-haze/90 font-medium italic leading-snug">
            &ldquo;{item.title}&rdquo;
          </p>
          {item.detail && (
            <p className="text-[11px] text-titan-metal/50 mt-2 leading-relaxed">{item.detail}</p>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between mt-2">
        <span className="text-[9px] text-titan-metal/30 uppercase tracking-wider">speech</span>
        <span className="text-[9px] text-titan-metal/30 font-mono">{formatTime(item.ts)}</span>
      </div>
    </div>
  );
}

function ReasoningCard({ item }: { item: MosaicItem }) {
  return (
    <div className="border border-amber-400/30 bg-amber-400/5 rounded-xl p-4 break-inside-avoid mb-3 hover:brightness-110 transition-all min-h-[100px]">
      <div className="flex items-start gap-2">
        <span className="text-lg shrink-0">{ICON_MAP.agency}</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-amber-300/80 font-medium leading-snug">
            {item.title}
          </p>
          {item.detail && (
            <p className="text-[11px] text-titan-metal/50 mt-1.5 leading-relaxed font-mono">{item.detail}</p>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between mt-2">
        <span className="text-[9px] text-titan-metal/30 uppercase tracking-wider">reasoning</span>
        <span className="text-[9px] text-titan-metal/30 font-mono">{formatTime(item.ts)}</span>
      </div>
    </div>
  );
}

function DefaultCard({ item }: { item: MosaicItem }) {
  const borderClass = COLOR_MAP[item.type] ?? 'border-titan-metal/10';
  const icon = ICON_MAP[item.type] ?? '\u25CB';
  const sizeClass = item.size === 'lg' ? 'min-h-[140px]' : item.size === 'md' ? 'min-h-[100px]' : 'min-h-[70px]';

  return (
    <div className={`border ${borderClass} rounded-xl p-4 break-inside-avoid mb-3 hover:brightness-110 transition-all ${sizeClass}`}>
      <div className="flex items-start gap-2">
        <span className={`shrink-0 ${item.size === 'lg' ? 'text-2xl' : 'text-lg'}`}>{icon}</span>
        <div className="flex-1 min-w-0">
          <p className={`text-titan-haze/80 font-medium leading-snug ${item.size === 'lg' ? 'text-sm' : 'text-xs'}`}>
            {item.title}
          </p>
          {item.detail && (
            <p className="text-[11px] text-titan-metal/50 mt-1.5 leading-relaxed">{item.detail}</p>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between mt-2">
        <span className="text-[9px] text-titan-metal/30 uppercase tracking-wider">{TYPE_LABELS[item.type] ?? item.type}</span>
        <span className="text-[9px] text-titan-metal/30 font-mono">{formatTime(item.ts)}</span>
      </div>
    </div>
  );
}

function MosaicCard({ item }: { item: MosaicItem }) {
  if (item.type === 'art' && item.mediaUrl) return <ArtCard item={item} />;
  if (item.type === 'speak') return <SpeechCard item={item} />;
  if (item.type === 'agency') return <ReasoningCard item={item} />;
  return <DefaultCard item={item} />;
}

export default function SoulMosaicTab() {
  const titanId = useTitanId();
  const [items, setItems] = useState<MosaicItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMosaic() {
      try {
        // Use narrated-feed for rich human-readable data
        const feedData = await titanFetch<{ items: Array<Record<string, unknown>> }>('/v6/expression/narrated-feed?limit=80', { titan: titanId });
        const feedItems = feedData?.items ?? [];

        // Also get archive for art images with media URLs
        const archiveData = await titanFetch<{ items: Array<Record<string, unknown>> }>('/status/archive?type=art&limit=30', { titan: titanId });
        const archiveItems = (archiveData?.items ?? []) as Array<Record<string, unknown>>;

        // Also get event-based items (dreams, pulses)
        const eventsData = await titanFetch<{ items: Array<Record<string, unknown>> }>('/v6/expression/activity-feed?limit=100', { titan: titanId });
        const eventItems = eventsData?.items ?? [];

        const mosaic: MosaicItem[] = [];

        // Process narrated feed (speech, reasoning, creation, consciousness, programs)
        for (const e of feedItems) {
          const ts = (e.ts ?? 0) as number;
          const cat = (e.category ?? '') as string;
          const narrative = (e.narrative ?? '') as string;
          const subtitle = (e.subtitle ?? '') as string;
          const mediaUrl = (e.media_url ?? '') as string;
          const details = (e.details ?? {}) as Record<string, unknown>;

          if (cat === 'speech' && narrative) {
            mosaic.push({
              type: 'speak', ts,
              title: narrative,
              detail: subtitle,
              size: narrative.length > 40 ? 'lg' : 'md',
            });
          } else if (cat === 'agency' && narrative) {
            mosaic.push({
              type: 'agency', ts,
              title: subtitle || 'Reasoning chain',
              detail: narrative,
              size: 'md',
            });
          } else if (cat === 'creation') {
            const mediaType = (e.media_type ?? 'art') as string;
            mosaic.push({
              type: mediaType === 'audio' ? 'music' : 'art',
              ts,
              title: narrative,
              detail: subtitle,
              mediaUrl: mediaUrl || undefined,
              size: 'lg',
            });
          } else if (cat === 'consciousness') {
            mosaic.push({
              type: 'consciousness', ts,
              title: narrative,
              detail: subtitle,
              size: narrative.length > 80 ? 'lg' : 'md',
            });
          } else if (cat === 'program_fire') {
            mosaic.push({
              type: 'program', ts,
              title: narrative,
              detail: subtitle,
              size: 'sm',
            });
          }
        }

        // Add archive art with actual images
        for (const a of archiveItems) {
          const aTs = (a.ts ?? 0) as number;
          const aTitle = ((a.title ?? a.content ?? '') as string);
          const content = (a.content ?? '') as string;
          const mediaPath = content.startsWith('/media/') ? content : undefined;

          if (mediaPath) {
            // Only add if not already in narrated feed (dedup by timestamp)
            if (!mosaic.some(m => m.type === 'art' && Math.abs(m.ts - aTs) < 10)) {
              mosaic.push({
                type: 'art', ts: aTs,
                title: aTitle || 'Artwork',
                mediaUrl: mediaPath,
                size: 'lg',
              });
            }
          }
        }

        // Add event-based items (dreams, pulses)
        for (const e of eventItems) {
          const ts = (e.ts ?? 0) as number;
          const type = (e.type ?? '') as string;
          const details = (e.details ?? {}) as Record<string, unknown>;

          if (type === 'dream_state') {
            const isDreaming = !!details.is_dreaming;
            mosaic.push({
              type: 'dream', ts,
              title: isDreaming ? 'Entering Dream' : 'Waking Up',
              detail: isDreaming ? 'Consolidating experiences through dream distillation' : 'Refreshed and recovered from sleep',
              size: 'md',
            });
          } else if (type === 'great_pulse') {
            mosaic.push({
              type: 'pulse', ts,
              title: 'Deep Coherence Moment',
              detail: `Trinity parts fully aligned — great pulse #${details.great_pulse_count ?? '?'}`,
              size: 'lg',
            });
          }
        }

        // Sort by timestamp descending, deduplicate
        mosaic.sort((a, b) => b.ts - a.ts);
        const deduped: MosaicItem[] = [];
        for (const item of mosaic) {
          if (!deduped.some(d => d.type === item.type && Math.abs(d.ts - item.ts) < 5)) {
            deduped.push(item);
          }
        }
        setItems(deduped.slice(0, 80));
      } catch {
        // empty
      } finally {
        setLoading(false);
      }
    }
    fetchMosaic();
  }, [titanId]);

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const item of items) {
      counts[item.type] = (counts[item.type] ?? 0) + 1;
    }
    return counts;
  }, [items]);

  const filteredItems = useMemo(() => {
    if (!filter) return items;
    return items.filter(item => item.type === filter);
  }, [items, filter]);

  if (loading) return <LoadingSkeleton lines={8} />;

  if (items.length === 0) {
    return (
      <p className="text-xs text-titan-metal/40 text-center py-12">
        No soul expressions yet. Titan creates art, music, and speaks when hormonal programs fire.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Type filter pills */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setFilter(null)}
          className={`text-[10px] px-2 py-1 rounded-full transition-all ${
            !filter ? 'bg-titan-haze/20 text-titan-haze' : 'text-titan-metal/40 hover:text-titan-metal/60'
          }`}
        >
          All ({items.length})
        </button>
        {Object.entries(typeCounts).sort((a, b) => b[1] - a[1]).map(([type, count]) => (
          <button
            key={type}
            onClick={() => setFilter(filter === type ? null : type)}
            className={`text-[10px] px-2 py-1 rounded-full flex items-center gap-1 transition-all ${
              filter === type ? 'bg-titan-haze/20 text-titan-haze' : 'text-titan-metal/40 hover:text-titan-metal/60'
            }`}
          >
            <span>{ICON_MAP[type] ?? '\u25CB'}</span>
            <span>{TYPE_LABELS[type] ?? type}: {count}</span>
          </button>
        ))}
      </div>

      {/* Masonry grid */}
      <div className="columns-1 md:columns-2 lg:columns-3 gap-3">
        {filteredItems.map((item, i) => (
          <MosaicCard key={`${item.type}-${item.ts}-${i}`} item={item} />
        ))}
      </div>
    </div>
  );
}

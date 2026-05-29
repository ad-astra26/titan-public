'use client';

import { useQuery } from '@tanstack/react-query';
import { titanFetch, v4Fetch } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';

interface FeedItem {
  ts: number;
  category: string;
  type: string;
  title: string;
  details: Record<string, unknown>;
}

const CATEGORY_CONFIG: Record<string, { icon: string; color: string; bg: string }> = {
  creation: { icon: '~', color: 'text-amber-400', bg: 'bg-amber-400/10' },
  neurology: { icon: '*', color: 'text-cyan-400', bg: 'bg-cyan-400/10' },
  consciousness: { icon: 'o', color: 'text-violet-400', bg: 'bg-violet-400/10' },
  agency: { icon: '>', color: 'text-emerald-400', bg: 'bg-emerald-400/10' },
  expression: { icon: '+', color: 'text-rose-400', bg: 'bg-rose-400/10' },
  rhythm: { icon: '.', color: 'text-blue-400', bg: 'bg-blue-400/10' },
  dreaming: { icon: 'z', color: 'text-indigo-300', bg: 'bg-indigo-300/10' },
};

function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 0) return 'now';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

function FeedEntry({ item }: { item: FeedItem }) {
  const cfg = CATEGORY_CONFIG[item.category] || CATEGORY_CONFIG.agency;

  const subtitle = (() => {
    const d = item.details || {};
    switch (item.category) {
      case 'creation':
        if (item.type === 'speak' || item.type === 'speak_composition') {
          return item.title || 'Composed a sentence';
        }
        if (item.type === 'art_generate') return 'Generated visual art';
        if (item.type === 'audio_generate') return 'Created sonic expression';
        return item.title || item.type;

      case 'neurology':
        if (item.type === 'program_fire') {
          const intensity = typeof d.intensity === 'number' ? d.intensity : 0;
          return `${d.program || '?'} (${d.layer || '?'}) intensity ${(intensity as number).toFixed(2)}`;
        }
        if (item.type === 'hormone_fired') return item.title;
        return item.title;

      case 'consciousness':
        if (item.type === 'epoch') {
          const curv = typeof d.curvature === 'number' ? (d.curvature as number).toFixed(3) : '?';
          const distill = typeof d.distillation === 'string' && d.distillation
            ? (d.distillation as string).slice(0, 100) : '';
          return `curvature ${curv}${d.anchored ? ' [anchored]' : ''}${distill ? ` — "${distill}"` : ''}`;
        }
        return item.title;

      case 'agency': {
        const score = typeof d.score === 'number' ? ` score ${(d.score as number).toFixed(1)}` : '';
        return `${d.helper || item.title}${score}`;
      }

      case 'expression':
        return item.title;

      case 'rhythm':
        if (item.type === 'great_pulse') return 'GREAT PULSE — all pairs resonant';
        if (item.type === 'big_pulse') return item.title;
        return item.title;

      case 'dreaming':
        return item.title;

      default:
        return item.title;
    }
  })();

  return (
    <div className="flex items-start gap-2 py-1.5 border-b border-titan-metal/5 last:border-0">
      <span className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5 ${cfg.color} ${cfg.bg}`}>
        {cfg.icon}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] text-titan-metal leading-snug truncate">
          {subtitle}
        </p>
      </div>
      <span className="text-[9px] text-titan-metal/30 shrink-0 mt-0.5">
        {timeAgo(item.ts)}
      </span>
    </div>
  );
}

export default function ActivityFeed() {
  const titanId = useTitanId();
  const { data, isLoading } = useQuery({
    queryKey: ['activity-feed', titanId],
    queryFn: () => v4Fetch<{ items: FeedItem[]; count: number }>('activity-feed', { titan: titanId, extraQuery: 'limit=50' }),
    refetchInterval: 10000,
    retry: 2,
  });

  const items = data?.items ?? [];

  // Category counts for header
  const counts = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.category] = (acc[item.category] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-titan-metal">
          Activity Feed
        </h3>
        <div className="flex items-center gap-1.5">
          {Object.entries(counts).slice(0, 5).map(([cat, count]) => {
            const cfg = CATEGORY_CONFIG[cat] || CATEGORY_CONFIG.agency;
            return (
              <span key={cat} className={`text-[9px] px-1.5 py-0.5 rounded ${cfg.color} ${cfg.bg}`}>
                {cat === 'consciousness' ? 'epoch' : cat} {count}
              </span>
            );
          })}
        </div>
      </div>

      {/* Feed */}
      {isLoading ? (
        <p className="text-xs text-titan-metal/50">Loading activity...</p>
      ) : items.length === 0 ? (
        <p className="text-xs text-titan-metal/50 italic">
          No activity yet — Titan is waking up...
        </p>
      ) : (
        <div className="max-h-[400px] overflow-y-auto pr-1 space-y-0">
          {items.map((item, idx) => (
            <FeedEntry key={`${item.ts}-${item.type}-${idx}`} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}

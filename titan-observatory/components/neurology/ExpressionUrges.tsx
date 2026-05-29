'use client';

import { useExpressionComposites } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const COMPOSITES = ['SPEAK', 'ART', 'MUSIC', 'SOCIAL'];
const ICONS: Record<string, string> = { SPEAK: '💬', ART: '🎨', MUSIC: '🎵', SOCIAL: '📢' };

export default function ExpressionUrges() {
  const titanId = useTitanId();
  const { data, isLoading } = useExpressionComposites(titanId);
  const raw = (data ?? {}) as Record<string, unknown>;
  const composites = (raw.composites ?? raw) as Record<string, Record<string, unknown>>;

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 text-center text-titan-metal/40">Loading expressions...</div>;
  }

  return (
    <div className="bg-titan-card rounded-xl p-6">
      <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-4">Expression Urges</h3>
      <div className="flex gap-4 justify-around">
        {COMPOSITES.map(name => {
          const c = (composites[name] ?? {}) as Record<string, unknown>;
          // 2026-05-15: API returns `urge` not `last_urge` — fix frontend
          // field mismatch surfaced post-§4.Q expression-composites verification
          // (T3 backend SPEAK urge=1.16 was reading as 0 in UI).
          const urge = typeof c?.urge === 'number'
            ? c.urge
            : (typeof c?.last_urge === 'number' ? c.last_urge : 0);
          const threshold = typeof c?.threshold === 'number' ? c.threshold : 1;
          const fireCount = typeof c?.fire_count === 'number' ? c.fire_count : 0;
          const fillPct = Math.min(urge / Math.max(threshold, 0.01), 1) * 100;
          const firing = urge >= threshold;

          return (
            <div key={name} className="flex flex-col items-center gap-1">
              <span className="text-lg">{ICONS[name]}</span>
              <div className="relative w-8 h-24 bg-titan-bg rounded-sm overflow-hidden">
                <div
                  className="absolute bottom-0 left-0 right-0 rounded-sm transition-all duration-500"
                  style={{
                    height: `${fillPct}%`,
                    backgroundColor: firing ? 'var(--titan-haze)' : 'var(--titan-metal)',
                    boxShadow: firing ? '0 0 8px rgba(229,199,158,0.5)' : 'none',
                    opacity: 0.8,
                  }}
                />
                {/* Threshold line */}
                <div className="absolute left-0 right-0 border-t border-dashed border-titan-metal/40"
                  style={{ bottom: `${Math.min(100, (threshold > 0 ? 100 : 0))}%` }}
                />
              </div>
              <span className="font-mono text-[10px] text-titan-metal/60">{urge.toFixed(2)}</span>
              <span className="font-mono text-[10px] text-titan-metal/40">fc:{fireCount}</span>
              <span className="text-xs text-titan-metal/60">{name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

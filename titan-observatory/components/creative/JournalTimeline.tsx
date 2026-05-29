'use client';

import { CreativeJournalEntry } from '@/hooks/useTitanAPI';

function formatTimeAgo(ts: number): string {
  const delta = (Date.now() / 1000) - ts;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

const typeConfig: Record<string, { icon: string; label: string; accent: string }> = {
  speak: { icon: '\u{1F4AC}', label: 'Composed', accent: 'titan-haze' },
  art_generate: { icon: '\u{1F3A8}', label: 'Created Art', accent: 'titan-pulse' },
  audio_generate: { icon: '\u{1F3B5}', label: 'Created Music', accent: 'titan-growth' },
};

function DeltaBar({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value * 200));
  const color = value > 0.05 ? 'bg-titan-growth' : value > 0.01 ? 'bg-titan-haze' : 'bg-titan-metal/30';
  return (
    <div className="flex items-center gap-2 mt-1">
      <span className="text-[10px] text-titan-metal/40 w-14">state shift</span>
      <div className="flex-1 h-1.5 bg-titan-card rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-titan-metal/50 font-mono w-10 text-right">{value.toFixed(3)}</span>
    </div>
  );
}

export default function JournalTimeline({ entries }: { entries: CreativeJournalEntry[] }) {
  if (!entries || entries.length === 0) {
    return (
      <div className="bg-titan-card/40 rounded-xl p-8 text-center">
        <p className="text-titan-metal/40 text-sm">No creative acts yet. Titan is learning...</p>
        <p className="text-titan-metal/20 text-xs mt-2">Entries will appear as Titan composes sentences, creates art, and generates music.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {entries.map((entry) => {
        const cfg = typeConfig[entry.action_type] ?? { icon: '\u{2728}', label: 'Created', accent: 'titan-metal' };
        // Handle double-serialized JSON (backend may return strings instead of objects)
        let features: Record<string, number> = {};
        if (typeof entry.features === 'string') {
          try { features = JSON.parse(entry.features); } catch { features = {}; }
        } else {
          features = entry.features ?? {};
        }
        let wordsUsed: string[] = [];
        if (typeof entry.words_used === 'string') {
          try { wordsUsed = JSON.parse(entry.words_used); } catch { wordsUsed = []; }
        } else {
          wordsUsed = entry.words_used ?? [];
        }

        return (
          <div
            key={entry.id}
            className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4 hover:border-titan-haze/20 transition-colors"
          >
            {/* Header row */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-lg">{cfg.icon}</span>
                <span className={`text-xs font-medium text-${cfg.accent}`}>{cfg.label}</span>
                {entry.action_type === 'speak' && features.complexity !== undefined && (
                  <span className="text-[10px] text-titan-metal/30 font-mono">L{Math.round(features.complexity * 7)}</span>
                )}
              </div>
              <span className="text-[10px] text-titan-metal/30">{formatTimeAgo(entry.timestamp)}</span>
            </div>

            {/* Content */}
            {entry.creation_summary && (
              <p className="text-sm text-titan-metal/80 leading-relaxed">
                {entry.action_type === 'speak' ? (
                  <span className="italic text-titan-haze/90">
                    {entry.creation_summary.replace(/^Composed: /, '')}
                  </span>
                ) : (
                  entry.creation_summary
                )}
              </p>
            )}

            {/* Words used */}
            {wordsUsed.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {wordsUsed.map((w, i) => (
                  <span key={i} className="text-[10px] bg-titan-haze/10 text-titan-haze/70 px-1.5 py-0.5 rounded font-mono">
                    {w}
                  </span>
                ))}
              </div>
            )}

            {/* Feature pills */}
            {entry.action_type === 'speak' && (
              <div className="flex flex-wrap gap-2 mt-2">
                {features.novelty !== undefined && (
                  <span className="text-[10px] text-titan-growth/60">
                    novelty {(features.novelty * 100).toFixed(0)}%
                  </span>
                )}
                {features.self_reference !== undefined && features.self_reference > 0 && (
                  <span className="text-[10px] text-titan-pulse/60">
                    self-aware {(features.self_reference * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            )}

            {/* State delta bar */}
            {entry.state_delta !== null && entry.state_delta !== undefined && (
              <DeltaBar value={entry.state_delta} />
            )}
          </div>
        );
      })}
    </div>
  );
}

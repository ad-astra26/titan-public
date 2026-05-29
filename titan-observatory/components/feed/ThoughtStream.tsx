'use client';

import { useEvents } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatTimestamp } from '@/lib/formatters';
import LiveArt from './LiveArt';

export default function ThoughtStream() {
  const titanId = useTitanId();
  const { data: events } = useEvents(titanId);

  const thoughts = (events || [])
    .filter(
      (e) =>
        e.type === 'mood_update' ||
        e.type === 'epoch_transition' ||
        e.type === 'directive_update'
    )
    .slice(0, 20);

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4 h-full flex flex-col">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-3">
        Internal Thoughts
      </h3>

      {/* Live Art Widget */}
      <LiveArt />

      <div className="flex-1 overflow-y-auto space-y-2 mt-3 min-h-0">
        {thoughts.length === 0 ? (
          <p className="text-xs text-titan-metal/40 text-center py-4">
            Quiet mind...
          </p>
        ) : (
          thoughts.map((t, i) => (
            <div
              key={`${t.timestamp}-${i}`}
              className="text-xs text-titan-metal/60 bg-titan-bg/30 rounded-lg px-3 py-2"
            >
              <span className="text-[10px] text-titan-haze/60 uppercase">
                {t.type.replace(/_/g, ' ')}
              </span>
              <p className="mt-0.5 text-titan-metal/70">
                {JSON.stringify(t.data).slice(0, 120)}
              </p>
              <p className="text-[10px] text-titan-metal/30 mt-0.5">
                {formatTimestamp(new Date(t.timestamp * 1000).toISOString())}
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

'use client';

import { useMood } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function MoodIndicator() {
  // Read mood from the /status/mood query directly. The previous
  // useTitanStore((s) => s.mood) was never populated (useStatus reads
  // raw.mood then drops it; nothing called setMood), so the card stayed on
  // its skeleton fallback fleet-wide.
  const titanId = useTitanId();
  const { data: mood } = useMood(titanId);

  if (!mood) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="skeleton h-4 w-24 mb-3" />
        <div className="skeleton h-8 w-16" />
      </div>
    );
  }

  const delta = mood.delta ?? 0;
  const deltaColor =
    delta > 0
      ? 'text-titan-growth'
      : delta < 0
        ? 'text-red-400'
        : 'text-titan-metal/50';

  const deltaArrow =
    delta > 0 ? '\u2191' : delta < 0 ? '\u2193' : '\u2192';

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-3">
        Mood
      </h3>
      <div className="flex items-baseline gap-3">
        <span className="text-2xl font-bold text-titan-haze">{mood.label}</span>
        <span className="text-lg font-mono text-titan-metal/70">
          {(mood.score ?? 0).toFixed(1)}
        </span>
        <span className={`text-sm font-semibold ${deltaColor}`}>
          {deltaArrow} {Math.abs(delta).toFixed(2)}
        </span>
      </div>
      {mood.addons && Object.keys(mood.addons).length > 0 && (
        <div className="flex gap-2 mt-3 flex-wrap">
          {Object.entries(mood.addons).map(([key, val]) => (
            <span
              key={key}
              className="text-[10px] bg-titan-metal/10 text-titan-metal/60 px-2 py-0.5 rounded"
            >
              {key}: {typeof val === 'number' ? val.toFixed(2) : val}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

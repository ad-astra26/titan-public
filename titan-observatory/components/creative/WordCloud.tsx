'use client';

import { VocabWord } from '@/hooks/useTitanAPI';

const typeColors: Record<string, string> = {
  adjective: 'text-titan-haze',
  verb: 'text-titan-growth',
  noun: 'text-titan-pulse',
  unknown: 'text-titan-metal/60',
};

export default function WordCloud({ words }: { words: VocabWord[] }) {
  if (!words || words.length === 0) return null;

  // Sort by usage frequency for visual priority — most used words appear largest
  const sorted = [...words]
    .filter(w => w.confidence > 0.1)
    .sort((a, b) => (b.times_produced || 0) - (a.times_produced || 0));

  const maxProduced = Math.max(...sorted.map(w => w.times_produced || 1), 1);

  return (
    <div className="bg-titan-card/40 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-medium text-titan-metal/50 uppercase tracking-wider mb-3">
        Vocabulary ({sorted.length} words)
      </h3>
      <div className="flex flex-wrap gap-x-2 gap-y-1 items-baseline">
        {sorted.map((w) => {
          const freq = (w.times_produced || 0) / maxProduced;
          const scale = 0.65 + freq * 1.0;
          const opacity = 0.35 + freq * 0.65;
          const color = typeColors[w.word_type] ?? typeColors.unknown;

          return (
            <span
              key={w.word}
              className={`${color} font-medium transition-opacity hover:opacity-100 cursor-default`}
              style={{
                fontSize: `${scale}rem`,
                opacity,
              }}
              title={`${w.word} (${w.word_type}, conf: ${(w.confidence * 100).toFixed(0)}%, produced: ${w.times_produced}x)`}
            >
              {w.word}
            </span>
          );
        })}
      </div>
      <div className="flex gap-4 mt-3 text-[10px] text-titan-metal/30">
        <span><span className="text-titan-haze">adjective</span></span>
        <span><span className="text-titan-growth">verb</span></span>
        <span><span className="text-titan-pulse">noun</span></span>
      </div>
    </div>
  );
}

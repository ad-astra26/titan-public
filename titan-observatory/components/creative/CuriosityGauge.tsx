'use client';

import { CreativeJournalEntry } from '@/hooks/useTitanAPI';

interface CuriosityGaugeProps {
  entries: CreativeJournalEntry[];
  totalWords: number;
}

export default function CuriosityGauge({ entries, totalWords }: CuriosityGaugeProps) {
  // Derive stats from journal entries
  const speakEntries = entries.filter(e => e.action_type === 'speak');
  const artEntries = entries.filter(e => e.action_type === 'art_generate');
  const musicEntries = entries.filter(e => e.action_type === 'audio_generate');

  // Handle double-serialized features
  const parseFeatures = (e: CreativeJournalEntry): Record<string, number> => {
    if (typeof e.features === 'string') {
      try { return JSON.parse(e.features); } catch { return {}; }
    }
    return e.features ?? {};
  };

  const avgNovelty = speakEntries.length > 0
    ? speakEntries.reduce((sum, e) => sum + (parseFeatures(e).novelty ?? 0), 0) / speakEntries.length
    : 0;
  const avgDelta = speakEntries.length > 0
    ? speakEntries.reduce((sum, e) => sum + (e.state_delta ?? 0), 0) / speakEntries.length
    : 0;

  // Unique words used across all compositions (handle double-serialized JSON)
  const allWords = new Set<string>();
  speakEntries.forEach(e => {
    let wu: string[] = [];
    if (typeof e.words_used === 'string') {
      try { wu = JSON.parse(e.words_used); } catch { wu = []; }
    } else {
      wu = e.words_used ?? [];
    }
    wu.forEach(w => allWords.add(w.toLowerCase()));
  });

  return (
    <div className="bg-titan-card/40 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5 space-y-4">
      <h3 className="text-xs font-medium text-titan-metal/50 uppercase tracking-wider">
        Creative Stats
      </h3>

      {/* Composition breakdown */}
      <div className="grid grid-cols-3 gap-3">
        <StatBlock
          label="Sentences"
          value={speakEntries.length}
          icon={'\u{1F4AC}'}
          accent="titan-haze"
        />
        <StatBlock
          label="Art"
          value={artEntries.length}
          icon={'\u{1F3A8}'}
          accent="titan-pulse"
        />
        <StatBlock
          label="Music"
          value={musicEntries.length}
          icon={'\u{1F3B5}'}
          accent="titan-growth"
        />
      </div>

      {/* Gauges */}
      <GaugeBar label="Avg Novelty" value={avgNovelty} color="titan-haze" />
      <GaugeBar label="Avg State Shift" value={Math.min(1, avgDelta * 5)} color="titan-growth" />
      <GaugeBar label="Word Coverage" value={totalWords > 0 ? allWords.size / totalWords : 0} color="titan-pulse" />

      {/* Word diversity */}
      <div className="flex justify-between text-[10px] text-titan-metal/40 pt-1 border-t border-titan-metal/5">
        <span>{allWords.size} unique words used</span>
        <span>{totalWords} total vocabulary</span>
      </div>
    </div>
  );
}

function StatBlock({ label, value, icon, accent }: { label: string; value: number; icon: string; accent: string }) {
  return (
    <div className="text-center">
      <div className="text-lg">{icon}</div>
      <div className={`text-xl font-semibold text-${accent}`}>{value}</div>
      <div className="text-[10px] text-titan-metal/40">{label}</div>
    </div>
  );
}

function GaugeBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div>
      <div className="flex justify-between text-[10px] text-titan-metal/40 mb-1">
        <span>{label}</span>
        <span className="font-mono">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 bg-titan-bg rounded-full overflow-hidden">
        <div
          className={`h-full bg-${color} rounded-full transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

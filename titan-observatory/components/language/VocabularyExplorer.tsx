'use client';

import { useState, useMemo } from 'react';
import { useVocabulary, type VocabWord } from '@/hooks/useTitanAPI';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

const PHASE_COLORS: Record<string, string> = {
  first_word: 'text-titan-metal/40 border-titan-metal/20',
  contextual: 'text-orange-400 border-orange-400/30',
  producible: 'text-emerald-400 border-emerald-400/30',
};

export default function VocabularyExplorer({ titanId }: { titanId?: TitanId }) {
  const { data, isLoading } = useVocabulary(titanId);
  const [search, setSearch] = useState('');
  const [phaseFilter, setPhaseFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [sortBy, setSortBy] = useState<'confidence' | 'encounters' | 'produced'>('confidence');

  const words = data?.words ?? [];

  const wordTypes = useMemo(() => {
    const types = new Set<string>();
    words.forEach((w) => types.add(w.word_type));
    return Array.from(types).sort();
  }, [words]);

  const wordPhases = useMemo(() => {
    const phases = new Set<string>();
    words.forEach((w) => { if (w.learning_phase) phases.add(w.learning_phase); });
    return Array.from(phases).sort();
  }, [words]);

  const filtered = useMemo(() => {
    let result = words;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter((w) => w.word.toLowerCase().includes(q));
    }
    if (phaseFilter) {
      result = result.filter((w) => w.learning_phase === phaseFilter);
    }
    if (typeFilter) {
      result = result.filter((w) => w.word_type === typeFilter);
    }
    result.sort((a, b) => {
      if (sortBy === 'confidence') return b.confidence - a.confidence;
      if (sortBy === 'encounters') return b.times_encountered - a.times_encountered;
      return b.times_produced - a.times_produced;
    });
    return result;
  }, [words, search, phaseFilter, typeFilter, sortBy]);

  if (isLoading) return <LoadingSkeleton lines={8} />;

  return (
    <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-titan-haze mb-3">
        Vocabulary Explorer
        <span className="ml-2 text-xs font-normal text-titan-metal/40">
          {filtered.length} of {words.length} words
        </span>
      </h3>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search words..."
          className="px-3 py-1.5 text-xs rounded-lg bg-titan-bg border border-titan-metal/20 text-titan-metal placeholder-titan-metal/30 focus:border-titan-haze/40 focus:outline-none w-40"
        />
        <select
          value={phaseFilter}
          onChange={(e) => setPhaseFilter(e.target.value)}
          className="px-2 py-1.5 text-xs rounded-lg bg-titan-bg border border-titan-metal/20 text-titan-metal focus:border-titan-haze/40 focus:outline-none"
        >
          <option value="">All phases</option>
          {wordPhases.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="px-2 py-1.5 text-xs rounded-lg bg-titan-bg border border-titan-metal/20 text-titan-metal focus:border-titan-haze/40 focus:outline-none"
        >
          <option value="">All types</option>
          {wordTypes.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
          className="px-2 py-1.5 text-xs rounded-lg bg-titan-bg border border-titan-metal/20 text-titan-metal focus:border-titan-haze/40 focus:outline-none"
        >
          <option value="confidence">Sort: Confidence</option>
          <option value="encounters">Sort: Encounters</option>
          <option value="produced">Sort: Produced</option>
        </select>
      </div>

      {/* Word list */}
      <div className="flex flex-col gap-1 max-h-[500px] overflow-y-auto scrollbar-thin">
        {filtered.slice(0, 100).map((w) => (
          <WordRow key={w.word} word={w} />
        ))}
        {filtered.length === 0 && (
          <p className="text-xs text-titan-metal/30 text-center py-6">No words match your filters</p>
        )}
        {filtered.length > 100 && (
          <p className="text-xs text-titan-metal/30 text-center py-2">
            Showing 100 of {filtered.length} — refine filters to see more
          </p>
        )}
      </div>
    </div>
  );
}

function WordRow({ word: w }: { word: VocabWord }) {
  const phaseClass = PHASE_COLORS[w.learning_phase] || PHASE_COLORS.unlearned;
  const confPct = Math.round(w.confidence * 100);

  return (
    <div className="flex items-center gap-3 bg-titan-bg/40 rounded-lg px-3 py-2 hover:bg-titan-bg/70 transition-colors group">
      {/* Word */}
      <span className="text-sm font-mono text-titan-haze min-w-[80px] truncate">&ldquo;{w.word}&rdquo;</span>

      {/* Type badge */}
      <span className="text-[10px] text-titan-metal/40 w-16 truncate">{w.word_type}</span>

      {/* Phase badge */}
      <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${phaseClass}`}>
        {w.learning_phase}
      </span>

      {/* Confidence bar */}
      <div className="flex-1 flex items-center gap-2 min-w-0">
        <div className="flex-1 h-1.5 bg-titan-bg rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${confPct}%`,
              backgroundColor: confPct > 70 ? '#44CC66' : confPct > 40 ? '#E5C79E' : '#8E9AAF',
            }}
          />
        </div>
        <span className="text-[10px] font-mono text-titan-metal/50 w-8 text-right shrink-0">
          {confPct}%
        </span>
      </div>

      {/* Encounter / Produced counts */}
      <div className="hidden md:flex items-center gap-3 text-[10px] text-titan-metal/40 shrink-0">
        <span>{w.times_encountered} enc</span>
        <span>{w.times_produced} prod</span>
      </div>
    </div>
  );
}

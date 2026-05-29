'use client';

import dynamic from 'next/dynamic';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import JournalTimeline from '@/components/creative/JournalTimeline';
import WordCloud from '@/components/creative/WordCloud';
import CuriosityGauge from '@/components/creative/CuriosityGauge';
import CreativeGallery from '@/components/creative/CreativeGallery';
import { useCreativeJournal, useVocabulary, useLanguageGrounding, useCompositions } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const NarrativeTimeline = dynamic(() => import('@/components/creative/NarrativeTimeline'), { ssr: false });

export default function CreativeTab() {
  const titanId = useTitanId();
  const { data: journal, isLoading: journalLoading } = useCreativeJournal(200, titanId);
  const { data: vocab, isLoading: vocabLoading } = useVocabulary(titanId);
  const { data: grounding } = useLanguageGrounding(titanId);
  const { data: compositions } = useCompositions(titanId);

  const entries = journal?.entries ?? [];
  const words = vocab?.words ?? [];
  const totalCreations = journal?.count ?? entries.length;
  const speakEntries = entries.filter(e => e.action_type === 'speak' || e.action_type === 'speak_composition');
  const artEntries = entries.filter(e => e.action_type === 'art_generate');
  const musicEntries = entries.filter(e => e.action_type === 'audio_generate');

  const parseFeatures = (e: { features: Record<string, number> | null }): Record<string, number> => {
    if (typeof e.features === 'string') {
      try { return JSON.parse(e.features); } catch { return {}; }
    }
    return e.features ?? {};
  };
  const speakWithFeatures = entries.filter(e => e.action_type === 'speak' && parseFeatures(e).novelty !== undefined);
  const avgNovelty = speakWithFeatures.length > 0
    ? speakWithFeatures.reduce((s, e) => s + (parseFeatures(e).novelty ?? 0), 0) / speakWithFeatures.length
    : 0;
  const producible = words.filter(w => w.learning_phase === 'producible');

  if (journalLoading && vocabLoading) {
    return <LoadingSkeleton lines={8} />;
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Creations"
          value={totalCreations}
          sublabel={`${speakEntries.length} speak \u00b7 ${artEntries.length} art \u00b7 ${musicEntries.length} music`}
          accent="haze"
        />
        <MetricCard
          label="Vocabulary"
          value={producible.length}
          sublabel={`${words.length} total words`}
          accent="growth"
        />
        <MetricCard
          label="Grounded"
          value={grounding?.grounded ?? '--'}
          sublabel={grounding ? `${(grounding.grounding_rate * 100).toFixed(0)}% grounding rate` : 'loading...'}
          accent="pulse"
        />
        <MetricCard
          label="Compositions"
          value={compositions?.total_compositions ?? '--'}
          sublabel={compositions?.latest ? `latest: L${compositions.latest.level}` : 'loading...'}
          accent="metal"
        />
      </div>

      {/* Language grounding summary */}
      {grounding && grounding.top_grounded && grounding.top_grounded.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-medium text-titan-haze mb-2">
            Language Grounding
            <span className="ml-2 text-xs font-normal text-titan-metal/40">
              avg conf: {(grounding.avg_confidence * 100).toFixed(0)}% | grounding: {(grounding.avg_grounding_confidence * 100).toFixed(0)}%
            </span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {grounding.top_grounded.slice(0, 8).map((w) => (
              <div
                key={w.word}
                className="flex items-center gap-1.5 bg-titan-bg/50 rounded-lg px-2.5 py-1.5"
              >
                <span className="text-xs font-mono text-titan-haze">{w.word}</span>
                <span className="text-[10px] text-titan-metal/30">{w.word_type}</span>
                {w.sensory_contexts?.slice(0, 1).map((ctx) => (
                  <span
                    key={ctx}
                    className="text-[9px] px-1 py-0.5 rounded bg-titan-pulse/10 text-titan-pulse/60 border border-titan-pulse/15"
                  >
                    {ctx}
                  </span>
                ))}
                <div className="w-8 h-1 bg-titan-bg rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-titan-growth"
                    style={{ width: `${(w.cross_modal_conf ?? 0) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent compositions */}
      {compositions?.recent && compositions.recent.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-medium text-titan-haze mb-2">
            Recent Compositions
            <span className="ml-2 text-xs font-normal text-titan-metal/40">
              {compositions.total_compositions} total
            </span>
          </h3>
          <div className="flex flex-col gap-1.5">
            {compositions.recent.slice(0, 5).map((c, i) => (
              <div key={`${c.sentence}-${i}`} className="flex items-center gap-2 bg-titan-bg/40 rounded-lg px-3 py-2">
                <span className="text-xs text-titan-metal italic flex-1">&ldquo;{c.sentence}&rdquo;</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-titan-bg border border-titan-metal/10 text-titan-metal/50 shrink-0">
                  L{c.level}
                </span>
                <span className="text-[10px] font-mono text-titan-metal/40 shrink-0">
                  {(c.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <CreativeGallery />

      {/* LLM-narrated creative timeline */}
      <NarrativeTimeline titanId={titanId} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <h3 className="text-sm font-medium text-titan-metal/60">Timeline</h3>
          <JournalTimeline entries={entries} />
        </div>
        <div className="space-y-4">
          <CuriosityGauge entries={entries} totalWords={words.length} />
          <WordCloud words={words} />
        </div>
      </div>
    </div>
  );
}

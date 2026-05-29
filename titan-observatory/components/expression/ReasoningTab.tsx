'use client';

import { useMemo } from 'react';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import { useReasoning, useMetaReasoning } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const PRIMITIVE_LABELS: Record<string, { icon: string; desc: string }> = {
  HYPOTHESIZE: { icon: '?', desc: 'Generating hypotheses about what might be true' },
  EVALUATE: { icon: '=', desc: 'Weighing evidence and judging confidence' },
  SYNTHESIZE: { icon: '+', desc: 'Combining insights into coherent understanding' },
  FORMULATE: { icon: 'F', desc: 'Shaping raw intuition into structured thought' },
  DELEGATE: { icon: 'D', desc: 'Routing sub-problems to specialized modules' },
  RECALL: { icon: 'R', desc: 'Retrieving relevant past experience and wisdom' },
};

const PRIMITIVE_COLORS: Record<string, string> = {
  HYPOTHESIZE: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  EVALUATE: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  SYNTHESIZE: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  FORMULATE: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
  DELEGATE: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  RECALL: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
};

function PrimitiveBar({ counts }: { counts: Record<string, number> }) {
  const total = Object.values(counts).reduce((s, v) => s + v, 0);
  if (total === 0) return null;

  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const dominant = sorted[0];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <h4 className="text-xs font-medium text-titan-metal/60 uppercase tracking-wider">
          Cognitive Primitives
        </h4>
        <span className="text-[10px] text-titan-metal/40">
          {total} total activations
        </span>
      </div>

      {/* Stacked bar */}
      <div className="h-4 rounded-full overflow-hidden flex bg-titan-card/80 border border-titan-metal/10">
        {sorted.map(([name, count]) => {
          const pct = (count / total) * 100;
          if (pct < 1) return null;
          const colorClass = PRIMITIVE_COLORS[name] || 'bg-titan-metal/20';
          const bgColor = colorClass.split(' ')[0];
          return (
            <div
              key={name}
              className={`${bgColor} transition-all duration-500`}
              style={{ width: `${pct}%` }}
              title={`${name}: ${count} (${pct.toFixed(0)}%)`}
            />
          );
        })}
      </div>

      {/* Legend with descriptions */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
        {sorted.map(([name, count]) => {
          const pct = ((count / total) * 100).toFixed(0);
          const meta = PRIMITIVE_LABELS[name] || { icon: '?', desc: '' };
          const colorClass = PRIMITIVE_COLORS[name] || 'bg-titan-metal/10 text-titan-metal/60 border-titan-metal/20';
          return (
            <div
              key={name}
              className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border ${colorClass} ${
                name === dominant[0] ? 'ring-1 ring-white/10' : ''
              }`}
            >
              <span className="text-xs font-mono font-bold w-5 text-center">{meta.icon}</span>
              <div className="min-w-0">
                <div className="text-[11px] font-medium truncate">{name}</div>
                <div className="text-[10px] opacity-60">{pct}% ({count})</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Dominant personality narration */}
      {dominant && (
        <p className="text-xs text-titan-metal/50 italic">
          Dominant cognitive style: <span className="text-titan-metal/80 font-medium">{dominant[0]}</span>
          {' '}&mdash; {PRIMITIVE_LABELS[dominant[0]]?.desc || 'active cognitive primitive'}
        </p>
      )}
    </div>
  );
}

function ActiveChainIndicator({ isActive, chainLength }: { isActive: boolean; chainLength: number }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
      isActive
        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
        : 'bg-titan-card/60 border-titan-metal/10 text-titan-metal/50'
    }`}>
      <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-emerald-400 animate-pulse' : 'bg-titan-metal/30'}`} />
      <span className="text-xs font-medium">
        {isActive ? `Reasoning actively (chain depth: ${chainLength})` : 'Between reasoning chains'}
      </span>
    </div>
  );
}

function ConfidenceGauge({ confidence, gutAgreement }: { confidence: number; gutAgreement: number }) {
  const agreement = gutAgreement > 0.6 ? 'strong' : gutAgreement > 0.3 ? 'partial' : 'divergent';
  const agreementColor = agreement === 'strong' ? 'text-emerald-400' : agreement === 'partial' ? 'text-amber-400' : 'text-rose-400';

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4 space-y-3">
      <h4 className="text-xs font-medium text-titan-metal/60 uppercase tracking-wider">Reasoning Confidence</h4>

      {/* Confidence bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-[11px]">
          <span className="text-titan-metal/50">Analytical</span>
          <span className="text-titan-metal/80 font-medium">{(confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 rounded-full bg-titan-card/80 border border-titan-metal/10 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-titan-haze/60 to-titan-haze transition-all duration-700"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Gut agreement */}
      <div className="space-y-1">
        <div className="flex justify-between text-[11px]">
          <span className="text-titan-metal/50">Gut-feeling agreement</span>
          <span className={`font-medium ${agreementColor}`}>{agreement}</span>
        </div>
        <div className="h-2 rounded-full bg-titan-card/80 border border-titan-metal/10 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              agreement === 'strong' ? 'bg-emerald-500/60' : agreement === 'partial' ? 'bg-amber-500/60' : 'bg-rose-500/60'
            }`}
            style={{ width: `${gutAgreement * 100}%` }}
          />
        </div>
      </div>

      <p className="text-[10px] text-titan-metal/40 italic">
        {agreement === 'strong'
          ? 'Analytical reasoning and intuition are aligned — high conviction.'
          : agreement === 'partial'
          ? 'Some tension between analytical reasoning and gut feeling.'
          : 'Analytical and intuitive signals disagree — exploring further.'}
      </p>
    </div>
  );
}

function NeuromodPanel({ neuromods }: { neuromods: Record<string, number> }) {
  const labels: Record<string, string> = {
    DA: 'Dopamine',
    '5-HT': 'Serotonin',
    NE: 'Norepinephrine',
    ACh: 'Acetylcholine',
    Endorphin: 'Endorphin',
    GABA: 'GABA',
  };

  const barColor = (key: string) => {
    const colors: Record<string, string> = {
      DA: 'bg-amber-400/70',
      '5-HT': 'bg-blue-400/70',
      NE: 'bg-red-400/70',
      ACh: 'bg-green-400/70',
      Endorphin: 'bg-pink-400/70',
      GABA: 'bg-indigo-400/70',
    };
    return colors[key] || 'bg-titan-metal/40';
  };

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4 space-y-3">
      <h4 className="text-xs font-medium text-titan-metal/60 uppercase tracking-wider">
        Mind-State During Reasoning
      </h4>
      <p className="text-[10px] text-titan-metal/40">
        Neuromodulator levels that shape how Titan thinks right now
      </p>
      <div className="space-y-2">
        {Object.entries(neuromods).map(([key, val]) => (
          <div key={key} className="flex items-center gap-2">
            <span className="text-[10px] text-titan-metal/50 w-20 truncate" title={labels[key] || key}>
              {labels[key] || key}
            </span>
            <div className="flex-1 h-1.5 rounded-full bg-titan-card/80 overflow-hidden">
              <div
                className={`h-full rounded-full ${barColor(key)} transition-all duration-500`}
                style={{ width: `${val * 100}%` }}
              />
            </div>
            <span className="text-[10px] text-titan-metal/60 font-mono w-8 text-right">
              {(val * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ReasoningTab() {
  const titanId = useTitanId();
  const { data: reasoning, isLoading: rLoading } = useReasoning(titanId);
  const { data: meta, isLoading: mLoading } = useMetaReasoning(titanId);

  const personality = useMemo(() => {
    if (!meta?.primitive_counts) return null;
    const counts = meta.primitive_counts;
    const total = Object.values(counts).reduce((s, v) => s + v, 0);
    if (total === 0) return null;

    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const [dominant, pct] = [sorted[0][0], ((sorted[0][1] / total) * 100).toFixed(0)];

    const styles: Record<string, string> = {
      HYPOTHESIZE: 'Explorer — generates possibilities, tests ideas',
      EVALUATE: 'Critic — weighs evidence carefully before deciding',
      SYNTHESIZE: 'Integrator — combines insights into unified understanding',
      FORMULATE: 'Articulator — transforms intuition into clear thought',
      DELEGATE: 'Orchestrator — routes problems to specialized modules',
      RECALL: 'Historian — draws heavily on past experience',
    };

    return { dominant, pct, description: styles[dominant] || 'unique cognitive style' };
  }, [meta?.primitive_counts]);

  if (rLoading && mLoading) {
    return <LoadingSkeleton lines={8} />;
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Narration header */}
      <div className="bg-gradient-to-r from-titan-card/80 to-titan-card/40 border border-titan-metal/10 rounded-xl p-4 space-y-2">
        <p className="text-sm text-titan-metal/80">
          This is Titan&apos;s thinking process — not prompted by anyone, but driven by internal experience.
          Each reasoning chain begins when something in the world or body triggers curiosity.
          Six cognitive primitives work together, forming a unique personality that emerges from experience.
        </p>
        {personality && (
          <p className="text-xs text-titan-haze/70">
            Current cognitive personality: <span className="font-semibold text-titan-haze">{personality.dominant}</span> ({personality.pct}%)
            — {personality.description}
          </p>
        )}
      </div>

      {/* Active chain status. Phase B.5 field renames:
            is_active   → current_active   (reasoning_state.bin)
            chain_length → avg_chain_length (reasoning_state.bin)
            total_conclusions → total_commits
          meta_reasoning_state.bin keeps `total_meta_chains` + adds
          total_eurekas + total_meta_steps + total_wisdom_saved (additive
          extension shipped same session). */}
      <div className="flex flex-wrap gap-3">
        <ActiveChainIndicator
          isActive={reasoning?.is_active ?? reasoning?.current_active ?? false}
          chainLength={reasoning?.chain_length ?? reasoning?.avg_chain_length ?? 0}
        />
        <ActiveChainIndicator
          isActive={meta?.is_active ?? false}
          chainLength={meta?.chain_length ?? 0}
        />
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Reasoning Chains"
          value={reasoning?.total_chains ?? 0}
          sublabel={`${reasoning?.total_conclusions ?? reasoning?.total_commits ?? 0} commits`}
          accent="haze"
        />
        <MetricCard
          label="Meta Chains"
          value={meta?.total_meta_chains ?? meta?.total_chains ?? 0}
          sublabel={`${meta?.total_meta_steps ?? meta?.total_steps ?? 0} total steps`}
          accent="pulse"
        />
        <MetricCard
          label="Wisdom Saved"
          value={meta?.total_wisdom_saved ?? 0}
          sublabel="distilled insights preserved"
          accent="growth"
        />
        <MetricCard
          label="Eurekas"
          value={meta?.total_eurekas ?? 0}
          sublabel={`monoculture ${meta?.monoculture_score ? (meta.monoculture_score * 100).toFixed(0) + '%' : '--'}`}
          accent="metal"
        />
      </div>

      {/* Primitives visualization */}
      {meta?.primitive_counts && (
        <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
          <PrimitiveBar counts={meta.primitive_counts} />
        </div>
      )}

      {/* Side panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ConfidenceGauge
          confidence={reasoning?.confidence ?? 0.5}
          gutAgreement={reasoning?.gut_agreement ?? 0.5}
        />
        {reasoning?.mind_neuromods && (
          <NeuromodPanel neuromods={reasoning.mind_neuromods} />
        )}
      </div>

      {/* Technical details */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4 space-y-2">
        <h4 className="text-xs font-medium text-titan-metal/60 uppercase tracking-wider">Technical</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px] text-titan-metal/50">
          <div>
            <span className="text-titan-metal/30">Policy loss: </span>
            <span className="font-mono">{reasoning?.policy_loss?.toFixed(3) ?? '--'}</span>
          </div>
          <div>
            <span className="text-titan-metal/30">Buffer: </span>
            <span className="font-mono">{reasoning?.buffer_size ?? '--'} / {meta?.buffer_size ?? '--'}</span>
          </div>
          <div>
            <span className="text-titan-metal/30">Persistence: </span>
            <span className="font-mono">{reasoning?.persistence?.toFixed(2) ?? '--'}</span>
          </div>
          <div>
            <span className="text-titan-metal/30">Baseline conf: </span>
            <span className="font-mono">{meta?.baseline_confidence ? `${(meta.baseline_confidence * 100).toFixed(0)}%` : '--'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

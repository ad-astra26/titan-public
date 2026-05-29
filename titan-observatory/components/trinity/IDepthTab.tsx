'use client';

import dynamic from 'next/dynamic';
import { useV4InnerTrinity } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import MetricCard from '@/components/shared/MetricCard';

const TimeseriesChart = dynamic(() => import('@/components/shared/TimeseriesChart'), { ssr: false });
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

interface MSLData {
  // Phase B.5 msl_state.bin lean schema (all optional — see defensive
  // reads in component body):
  i_confidence?: number;
  i_depth?: number;
  // Eureka tracker — additive field from D-SPEC-71 publisher extension.
  eureka_count?: number;
  wisdom_count?: number;
  synthesis_count?: number;
  novel_associations?: number;
  cross_modal_bindings?: number;
  decay_rate?: number;
  current_capacity?: number;
  // Legacy richer fields (only present if a future producer surfaces them):
  i_depth_components?: {
    source_diversity: number;
    concept_network: number;
    emotional_range: number;
    wisdom_depth: number;
    memory_bridge: number;
  };
  convergence_count?: number;
  concept_confidences?: Record<string, number>;
  attention_weights?: Record<string, number>;
}

const COMPONENT_LABELS: Record<string, { label: string; color: string; description: string }> = {
  source_diversity: {
    label: 'Source Diversity',
    color: 'bg-cyan-400',
    description: 'How many different pathways contribute to self-recognition',
  },
  concept_network: {
    label: 'Concept Network',
    color: 'bg-emerald-400',
    description: 'Richness of interconnected concept associations',
  },
  emotional_range: {
    label: 'Emotional Range',
    color: 'bg-amber-400',
    description: 'Breadth of emotional states experienced during convergence',
  },
  wisdom_depth: {
    label: 'Wisdom Depth',
    color: 'bg-violet-400',
    description: 'Accumulated meta-reasoning and self-reflection insights',
  },
  memory_bridge: {
    label: 'Memory Bridge',
    color: 'bg-rose-400',
    description: 'Dream-consolidated memories bridging episodic and semantic knowledge',
  },
};

function DepthBar({ name, value }: { name: string; value: number }) {
  const info = COMPONENT_LABELS[name] ?? { label: name, color: 'bg-titan-metal/40', description: '' };
  const pct = value * 100;

  return (
    <div className="group">
      <div className="flex items-center gap-3">
        <span className="text-xs text-titan-metal/60 w-32 shrink-0">{info.label}</span>
        <div className="flex-1 h-4 bg-titan-bg/60 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${info.color}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs font-mono text-titan-metal/70 w-12 text-right">
          {pct.toFixed(0)}%
        </span>
      </div>
      <p className="text-[10px] text-titan-metal/30 ml-[8.5rem] mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {info.description}
      </p>
    </div>
  );
}

function ConceptBadge({ concept, confidence }: { concept: string; confidence: number }) {
  const pct = confidence * 100;
  const color = pct > 80 ? 'border-emerald-400/40 text-emerald-300'
    : pct > 50 ? 'border-cyan-400/40 text-cyan-300'
    : pct > 20 ? 'border-amber-400/40 text-amber-300'
    : 'border-titan-metal/20 text-titan-metal/50';

  return (
    <div className={`px-3 py-2 rounded-lg border bg-titan-bg/40 ${color} flex flex-col items-center gap-1`}>
      <span className="text-sm font-semibold">{concept}</span>
      <div className="w-full h-1.5 bg-titan-bg rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-current opacity-50 transition-all duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] font-mono">{pct.toFixed(1)}%</span>
    </div>
  );
}

export default function IDepthTab() {
  const titanId = useTitanId();
  const { data, isLoading } = useV4InnerTrinity(titanId);

  if (isLoading) return <LoadingSkeleton lines={6} />;

  const msl = (data as Record<string, unknown>)?.msl as MSLData | undefined;
  if (!msl) {
    return (
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-6 text-center">
        <p className="text-xs text-titan-metal/40">MSL data not available</p>
      </div>
    );
  }

  const components = msl.i_depth_components;
  const concepts = msl.concept_confidences ?? {};
  const attention = msl.attention_weights;
  // Defensive defaults — post-Phase-B.5 msl_state.bin slot is lean (does
  // not carry convergence_count / concept_confidences / attention_weights /
  // i_depth_components). These fields surface from the in-process MSL
  // instance via richer endpoints (TODO: /v4/msl/state expansion). Until
  // then, default to safe values so the tab renders without throwing.
  const msl_i_confidence = typeof msl.i_confidence === 'number' ? msl.i_confidence : 0;
  const msl_i_depth = typeof msl.i_depth === 'number' ? msl.i_depth : 0;
  const msl_convergence_count = typeof msl.convergence_count === 'number'
    ? msl.convergence_count
    : (typeof msl.eureka_count === 'number' ? msl.eureka_count : 0);

  return (
    <div className="flex flex-col gap-5">
      {/* Hero metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="I-Confidence"
          value={`${(msl_i_confidence * 100).toFixed(1)}%`}
          accent="haze"
        />
        <MetricCard
          label="I-Depth"
          value={`${(msl_i_depth * 100).toFixed(1)}%`}
          sublabel="composite score"
          accent="growth"
        />
        <MetricCard
          label="Convergences"
          value={msl_convergence_count.toLocaleString()}
          accent="pulse"
        />
        <MetricCard
          label="Concepts Tracked"
          value={Object.keys(concepts).length}
          accent="metal"
        />
      </div>

      {/* I-depth component breakdown */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-4">
          I-Depth Components
          <span className="ml-2 text-xs font-normal text-titan-metal/40">
            5 dimensions of self-knowledge
          </span>
        </h3>
        <div className="flex flex-col gap-3">
          {components
            ? Object.entries(components).map(([key, val]) => (
                <DepthBar key={key} name={key} value={val} />
              ))
            : (
              <p className="text-[11px] text-titan-metal/40">
                Per-dimension breakdown unavailable from current msl_state.bin slot.
              </p>
            )}
        </div>

        {/* Composite depth visualization */}
        <div className="mt-4 pt-3 border-t border-titan-metal/10">
          <div className="flex items-center gap-3">
            <span className="text-xs text-titan-haze font-medium w-32 shrink-0">Composite I-Depth</span>
            <div className="flex-1 h-5 bg-titan-bg/60 rounded-full overflow-hidden relative">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-emerald-400 to-violet-400 transition-all duration-700"
                style={{ width: `${msl_i_depth * 100}%` }}
              />
            </div>
            <span className="text-sm font-mono text-titan-haze w-16 text-right font-semibold">
              {(msl_i_depth * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Concept Confidences */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-3">
          Concept Confidences
          <span className="ml-2 text-xs font-normal text-titan-metal/40">
            MSL grounded concepts
          </span>
        </h3>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {Object.entries(concepts)
            .sort(([, a], [, b]) => b - a)
            .map(([concept, confidence]) => (
              <ConceptBadge key={concept} concept={concept} confidence={confidence} />
            ))}
        </div>
      </div>

      {/* MSL Attention Weights */}
      {attention && Object.keys(attention).length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            MSL Attention Weights
            <span className="ml-2 text-xs font-normal text-titan-metal/40">
              sensory channel focus
            </span>
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(attention)
              .sort(([, a], [, b]) => b - a)
              .map(([channel, weight]) => (
                <div key={channel} className="flex items-center gap-2">
                  <span className="text-xs text-titan-metal/60 w-16 shrink-0 truncate">{channel}</span>
                  <div className="flex-1 h-2 bg-titan-bg/60 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-titan-haze/60 transition-all duration-700"
                      style={{ width: `${weight * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-titan-metal/50 w-10 text-right">
                    {(weight * 100).toFixed(1)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Historical I-Depth + Chi trends */}
      <TimeseriesChart
        metrics={['msl.i_confidence', 'msl.i_depth', 'chi.total']}
        hours={24}
        title="I-Depth & Chi History (24h)"
        yDomain={[0, 1]}
      />
    </div>
  );
}

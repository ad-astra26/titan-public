'use client';

import { useTrinityLive } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const BODY_DIMS = ['Interoception', 'Proprioception', 'Somatosensation', 'Entropy', 'Thermal'];

const MIND_DIMS = [
  'Vision', 'Hearing', 'Taste', 'Smell', 'Touch',
  'Feeling.1', 'Feeling.2', 'Feeling.3',
  'Thinking.1', 'Thinking.2', 'Thinking.3',
  'Willing.1', 'Willing.2', 'Willing.3', 'Willing.4',
];

const SPIRIT_DIMS_SHORT = [
  'WHO', 'WHY', 'WHAT', 'Body\u03A3', 'Mind\u03A3',
  's5', 's6', 's7', 's8', 's9',
  's10', 's11', 's12', 's13', 's14',
];

export default function TrinityRadar() {
  const titanId = useTitanId();
  const { data: trinityRaw } = useTrinityLive(titanId);

  const raw = (trinityRaw ?? {}) as Record<string, unknown>;
  const trinity = (raw?.trinity ?? raw) as Record<string, { values?: number[]; dims?: string[] }>;
  const outerBody = (raw?.outer_body ?? []) as number[];
  const outerMind = (raw?.outer_mind ?? []) as number[];
  const outerSpirit = (raw?.outer_spirit ?? []) as number[];

  const innerBody = trinity?.body?.values ?? [0.5, 0.5, 0.5, 0.5, 0.5];
  const innerMind = trinity?.mind?.values ?? Array(15).fill(0.5);
  const innerSpirit = trinity?.spirit?.values ?? Array(15).fill(0.5);

  // Body radar data (5D)
  const bodyData = BODY_DIMS.map((dim, i) => ({
    dim,
    inner: Math.round((innerBody[i] ?? 0.5) * 100) / 100,
    outer: Math.round(((outerBody[i] ?? innerBody[i]) ?? 0.5) * 100) / 100,
  }));

  // Mind radar data (15D)
  const mindData = MIND_DIMS.map((dim, i) => ({
    dim,
    inner: Math.round((innerMind[i] ?? 0.5) * 100) / 100,
    outer: Math.round(((outerMind[i] ?? innerMind[i]) ?? 0.5) * 100) / 100,
  }));

  // Spirit radar data (first 15 of 45D for readability)
  const spiritData = SPIRIT_DIMS_SHORT.map((dim, i) => ({
    dim,
    inner: Math.round((innerSpirit[i] ?? 0.5) * 100) / 100,
    outer: Math.round(((outerSpirit[i] ?? innerSpirit[i]) ?? 0.5) * 100) / 100,
  }));

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-1">
        Divine Trinity — Live Tensors
      </h3>
      <p className="text-[10px] text-titan-metal/30 mb-4">
        Inner Trinity (solid) overlaid with Outer Trinity (dashed). When inner and outer align, resonance emerges.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Body Radar (5D) */}
        <div>
          <p className="text-[10px] text-titan-metal/50 text-center mb-1 uppercase tracking-wider">Body (5D Somatic)</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={bodyData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#8E9AAF15" />
              <PolarAngleAxis dataKey="dim" tick={{ fill: '#8E9AAF', fontSize: 8 }} />
              <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} />
              <Radar name="Outer" dataKey="outer" stroke="#FF6B6B" fill="#FF6B6B" fillOpacity={0.08} strokeWidth={1} strokeDasharray="4 3" />
              <Radar name="Inner" dataKey="inner" stroke="#FF6B6B" fill="#FF6B6B" fillOpacity={0.2} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Mind Radar (15D) */}
        <div>
          <p className="text-[10px] text-titan-metal/50 text-center mb-1 uppercase tracking-wider">Mind (15D Cognitive)</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={mindData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#8E9AAF15" />
              <PolarAngleAxis dataKey="dim" tick={{ fill: '#8E9AAF', fontSize: 7 }} />
              <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} />
              <Radar name="Outer" dataKey="outer" stroke="#9945FF" fill="#9945FF" fillOpacity={0.08} strokeWidth={1} strokeDasharray="4 3" />
              <Radar name="Inner" dataKey="inner" stroke="#9945FF" fill="#9945FF" fillOpacity={0.2} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Spirit Radar (first 15 of 45D) */}
        <div>
          <p className="text-[10px] text-titan-metal/50 text-center mb-1 uppercase tracking-wider">Spirit (45D — top 15)</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={spiritData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#8E9AAF15" />
              <PolarAngleAxis dataKey="dim" tick={{ fill: '#8E9AAF', fontSize: 7 }} />
              <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} />
              <Radar name="Outer" dataKey="outer" stroke="#77CCCC" fill="#77CCCC" fillOpacity={0.08} strokeWidth={1} strokeDasharray="4 3" />
              <Radar name="Inner" dataKey="inner" stroke="#77CCCC" fill="#77CCCC" fillOpacity={0.2} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-2 text-[10px] text-titan-metal/40">
        <span className="flex items-center gap-1.5">
          <span className="w-4 h-0.5 bg-titan-haze inline-block rounded" /> Inner Trinity
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-4 h-0.5 bg-titan-haze/30 inline-block rounded border-b border-dashed border-titan-haze" /> Outer Trinity
        </span>
      </div>
    </div>
  );
}

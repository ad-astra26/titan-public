'use client';

import { useState } from 'react';
import { useTrinityHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { mockTrinityHistory } from '@/lib/mockData';
import { shouldUseMock } from '@/lib/mockData';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const TIME_RANGES = [
  { label: '1h', hours: 1, format: 'time' },
  { label: '6h', hours: 6, format: 'time' },
  { label: '24h', hours: 24, format: 'time' },
  { label: '7d', hours: 168, format: 'date' },
] as const;

type TimeRange = (typeof TIME_RANGES)[number];

const BODY_COLORS = ['#FF6B6B', '#FF8E8E', '#FFB0B0', '#FF5252', '#FF7979'];
const MIND_COLORS = ['#9945FF', '#B070FF', '#C99AFF', '#7B2FCC', '#A855FF'];
const BODY_DIMS = ['Interoception', 'Proprioception', 'Somatosensation', 'Entropy', 'Thermal'];
const MIND_DIMS = ['Vision', 'Hearing', 'Taste', 'Smell', 'Touch'];

const tooltipStyle = {
  backgroundColor: '#1A1D23',
  border: '1px solid #8E9AAF30',
  borderRadius: '8px',
  fontSize: 11,
};

export default function TrinityTrends() {
  const titanId = useTitanId();
  const [range, setRange] = useState<TimeRange>(TIME_RANGES[1]); // default 6h
  const { data: rawHistory, isLoading } = useTrinityHistory(range.hours, titanId);

  const history = shouldUseMock(rawHistory) ? mockTrinityHistory : rawHistory!;

  if (isLoading) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Trinity Trends
        </h3>
        <LoadingSkeleton lines={5} />
      </div>
    );
  }

  // Filter + format
  const cutoff = Date.now() / 1000 - range.hours * 3600;
  const filtered = history.filter((s) => s.ts >= cutoff);

  const bodyData = filtered.map((s) => ({
    time: range.format === 'time'
      ? new Date(s.ts * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
      : new Date(s.ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    ...Object.fromEntries(BODY_DIMS.map((d, i) => [d, s.body_tensor?.[i] ?? 0.5])),
  }));

  const mindData = filtered.map((s) => ({
    time: range.format === 'time'
      ? new Date(s.ts * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
      : new Date(s.ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    ...Object.fromEntries(MIND_DIMS.map((d, i) => [d, s.mind_tensor?.[i] ?? 0.5])),
  }));

  const lossData = filtered.map((s) => ({
    time: range.format === 'time'
      ? new Date(s.ts * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
      : new Date(s.ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    loss: s.middle_path_loss ?? 0,
  }));

  const rangeSelector = (
    <div className="flex gap-1">
      {TIME_RANGES.map((r) => (
        <button
          key={r.label}
          onClick={() => setRange(r)}
          className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
            range.label === r.label
              ? 'bg-titan-pulse/20 text-titan-pulse'
              : 'text-titan-metal/40 hover:text-titan-metal/60 hover:bg-titan-metal/5'
          }`}
        >
          {r.label}
        </button>
      ))}
    </div>
  );

  if (filtered.length === 0) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Trinity Trends
          </h3>
          {rangeSelector}
        </div>
        <p className="text-xs text-titan-metal/40">No Trinity data for this period</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Body 5DT Trends */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Body Tensor Trends ({range.label})
          </h3>
          {rangeSelector}
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={bodyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
            <XAxis dataKey="time" tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
            {BODY_DIMS.map((dim, i) => (
              <Line key={dim} type="monotone" dataKey={dim} stroke={BODY_COLORS[i]} strokeWidth={1.5} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Mind 5DT Trends */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Mind Tensor Trends ({range.label})
          </h3>
          {rangeSelector}
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={mindData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
            <XAxis dataKey="time" tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
            {MIND_DIMS.map((dim, i) => (
              <Line key={dim} type="monotone" dataKey={dim} stroke={MIND_COLORS[i]} strokeWidth={1.5} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Middle Path Loss */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Middle Path Equilibrium ({range.label})
          </h3>
          {rangeSelector}
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={lossData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
            <XAxis dataKey="time" tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Area
              type="monotone"
              dataKey="loss"
              name="Middle Path Loss"
              stroke="#E5C79E"
              fill="#E5C79E"
              fillOpacity={0.15}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
        <p className="text-[10px] text-titan-metal/30 mt-1 text-center">
          0 = perfect equilibrium (Divine Center) | 1 = maximum imbalance
        </p>
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { useHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
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
} from 'recharts';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const TIME_RANGES = [
  { label: '1h', days: 0.042, format: 'time' },
  { label: '24h', days: 1, format: 'time' },
  { label: '7d', days: 7, format: 'date' },
  { label: '30d', days: 30, format: 'date' },
] as const;

type TimeRange = (typeof TIME_RANGES)[number];

function TimeRangeSelector({
  selected,
  onChange,
}: {
  selected: TimeRange;
  onChange: (r: TimeRange) => void;
}) {
  return (
    <div className="flex gap-1">
      {TIME_RANGES.map((r) => (
        <button
          key={r.label}
          onClick={() => onChange(r)}
          className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
            selected.label === r.label
              ? 'bg-titan-pulse/20 text-titan-pulse'
              : 'text-titan-metal/40 hover:text-titan-metal/60 hover:bg-titan-metal/5'
          }`}
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}

export default function HistoryCharts() {
  const titanId = useTitanId();
  const [range, setRange] = useState<TimeRange>(TIME_RANGES[2]); // default 7d
  const { data: history, isLoading } = useHistory(Math.max(1, Math.ceil(range.days)), titanId);

  if (isLoading) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Historical Data
        </h3>
        <LoadingSkeleton lines={5} />
      </div>
    );
  }

  // Filter to the actual time range
  const cutoff = Date.now() - range.days * 24 * 60 * 60 * 1000;
  const filtered = (history || []).filter(
    (pt) => new Date(pt.timestamp).getTime() >= cutoff
  );

  const chartData = filtered.map((pt) => ({
    time:
      range.format === 'time'
        ? new Date(pt.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
          })
        : new Date(pt.timestamp).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
          }),
    mood: pt.mood_score ?? 0,
    memories: pt.memory_count ?? 0,
  }));

  if (chartData.length === 0) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Historical Data
          </h3>
          <TimeRangeSelector selected={range} onChange={setRange} />
        </div>
        <p className="text-xs text-titan-metal/40">No historical data for this period</p>
      </div>
    );
  }

  const tooltipStyle = {
    backgroundColor: '#1A1D23',
    border: '1px solid #8E9AAF30',
    borderRadius: '8px',
    fontSize: 11,
  };

  return (
    <div className="space-y-6">
      {/* Mood over time */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Mood Score ({range.label})
          </h3>
          <TimeRangeSelector selected={range} onChange={setRange} />
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
            <XAxis dataKey="time" tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <YAxis tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Line type="monotone" dataKey="mood" stroke="#E5C79E" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Memory count over time */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Memory Growth ({range.label})
          </h3>
          <TimeRangeSelector selected={range} onChange={setRange} />
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
            <XAxis dataKey="time" tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <YAxis tick={{ fill: '#8E9AAF', fontSize: 10 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Area type="monotone" dataKey="memories" stroke="#77CCCC" fill="#77CCCC" fillOpacity={0.15} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

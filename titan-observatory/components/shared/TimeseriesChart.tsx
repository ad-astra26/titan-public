'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { useTimeseries } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

const METRIC_COLORS: Record<string, string> = {
  'neuromod.DA': '#f59e0b',
  'neuromod.5HT': '#3b82f6',
  'neuromod.NE': '#ef4444',
  'neuromod.ACh': '#10b981',
  'neuromod.Endorphin': '#ec4899',
  'neuromod.GABA': '#8b5cf6',
  'dreaming.fatigue': '#6366f1',
  'dreaming.cycle_count': '#a78bfa',
  'dreaming.is_dreaming': '#818cf8',
  'chi.total': '#14b8a6',
  'chi.spirit': '#a78bfa',
  'chi.mind': '#06b6d4',
  'chi.body': '#22c55e',
  'msl.i_confidence': '#f97316',
  'msl.i_depth': '#eab308',
  'reasoning.total_chains': '#64748b',
  'meta.total_chains': '#94a3b8',
  'meta.avg_reward': '#f43f5e',
  'ns.total_train_steps': '#0ea5e9',
  'pi.heartbeat_ratio': '#d946ef',
  'epoch.id': '#78716c',
  'vocab.count': '#84cc16',
  'expression.speak_fires': '#fbbf24',
  'expression.art_fires': '#fb7185',
  'social.urge': '#c084fc',
};

const DEFAULT_COLORS = [
  '#06b6d4', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#3b82f6', '#84cc16',
];

function shortLabel(name: string): string {
  const parts = name.split('.');
  return parts.length > 1 ? parts[parts.length - 1] : name;
}

function formatTime(ts: number, hours: number): string {
  const d = new Date(ts * 1000);
  if (hours > 48) {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
      ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit' });
  }
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

interface TimeseriesChartProps {
  metrics: string[];
  hours?: number;
  titanId?: TitanId;
  title?: string;
  height?: number;
  yDomain?: [number, number];
}

export default function TimeseriesChart({
  metrics,
  hours = 24,
  titanId: propTitanId,
  title,
  height = 240,
  yDomain,
}: TimeseriesChartProps) {
  const hookTitanId = useTitanId();
  const titanId = propTitanId ?? hookTitanId;
  const { data, isLoading } = useTimeseries(metrics, hours, titanId);

  const chartData = useMemo(() => {
    if (!data?.metrics) return [];

    // Collect all unique timestamps
    const tsSet = new Set<number>();
    for (const points of Object.values(data.metrics)) {
      for (const p of points) tsSet.add(p.ts);
    }
    const timestamps = Array.from(tsSet).sort((a, b) => a - b);

    // Build rows: { ts, time, metric1, metric2, ... }
    return timestamps.map(ts => {
      const row: Record<string, number | string> = {
        ts,
        time: formatTime(ts, hours),
      };
      for (const [name, points] of Object.entries(data.metrics)) {
        const point = points.find(p => p.ts === ts);
        if (point) row[name] = point.value;
      }
      return row;
    });
  }, [data, hours]);

  if (isLoading) return <LoadingSkeleton lines={3} />;

  if (chartData.length === 0) {
    return (
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        {title && <h3 className="text-sm font-semibold text-titan-haze mb-2">{title}</h3>}
        <div className="flex items-center justify-center" style={{ height }}>
          <p className="text-xs text-titan-metal/40">
            No historical data yet — first snapshot in ~5 minutes
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
      {title && (
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-titan-haze">{title}</h3>
          <span className="text-[10px] text-titan-metal/40">
            {data?.resolution} resolution &middot; {chartData.length} points
          </span>
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: 'rgba(148,163,184,0.4)' }}
            interval="preserveStartEnd"
            minTickGap={60}
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'rgba(148,163,184,0.4)' }}
            width={45}
            domain={yDomain ?? ['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(15,23,42,0.95)',
              border: '1px solid rgba(148,163,184,0.15)',
              borderRadius: '8px',
              fontSize: '11px',
            }}
            labelStyle={{ color: 'rgba(148,163,184,0.6)', fontSize: '10px' }}
          />
          {metrics.length > 1 && (
            <Legend
              wrapperStyle={{ fontSize: '10px' }}
              formatter={(value: string) => shortLabel(value)}
            />
          )}
          {metrics.map((name, i) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              name={name}
              stroke={METRIC_COLORS[name] ?? DEFAULT_COLORS[i % DEFAULT_COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

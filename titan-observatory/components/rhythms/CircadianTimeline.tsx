'use client';

import { useMemo } from 'react';
import { useV4History, useDreaming } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export default function CircadianTimeline() {
  const titanId = useTitanId();
  const { data: historyData, isLoading } = useV4History(24, titanId);
  const { data: dreamData } = useDreaming(titanId);

  const fatigue = dreamData?.fatigue ?? 0;
  const isDreaming = dreamData?.is_dreaming === true;
  const epochsSinceDream = dreamData?.epochs_since_dream ?? 0;

  const chartData = useMemo(() => {
    const snapshots = historyData?.snapshots ?? [];
    if (snapshots.length === 0) return [];

    // Downsample to ~200 points for smooth rendering
    const step = Math.max(1, Math.floor(snapshots.length / 200));
    return snapshots
      .filter((_, i) => i % step === 0)
      .map((s) => {
        const date = new Date(s.ts * 1000);
        return {
          time: date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          ts: s.ts,
          loss: s.middle_path_loss ?? 0,
          velocity: Math.abs(s.spirit_velocity ?? 0),
          pulses: (s.great_pulse_count ?? 0) + (s.big_pulse_count ?? 0),
        };
      });
  }, [historyData]);

  if (isLoading) {
    return (
      <div className="bg-titan-card rounded-xl p-6 animate-pulse">
        <div className="h-4 bg-titan-metal/10 rounded w-40 mb-4" />
        <div className="h-40 bg-titan-metal/5 rounded" />
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className="bg-titan-card rounded-xl p-6 text-center">
        <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-2">
          Circadian Timeline
        </h3>
        <p className="text-xs text-titan-metal/30">No activity data in last 24 hours</p>
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
    <div className="bg-titan-card rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider">
            Circadian Timeline (24h)
          </h3>
          <p className="text-[10px] text-titan-metal/30 mt-0.5">
            Middle Path equilibrium + spirit velocity over time
          </p>
        </div>
        <div className="flex items-center gap-3">
          {isDreaming ? (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-titan-pulse/10 border border-titan-pulse/20">
              <span className="w-1.5 h-1.5 rounded-full bg-titan-pulse animate-pulse" />
              <span className="text-[10px] font-mono text-titan-pulse">DREAMING</span>
            </span>
          ) : (
            <span className="text-[10px] font-mono text-titan-metal/40">
              fatigue: {(fatigue * 100).toFixed(0)}% · {epochsSinceDream} epochs awake
            </span>
          )}
        </div>
      </div>

      {/* Middle Path Loss — lower = more balanced */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-mono text-titan-haze/60 uppercase">Middle Path Equilibrium</span>
          <span className="text-[10px] font-mono text-titan-metal/30">lower = more balanced</span>
        </div>
        <ResponsiveContainer width="100%" height={120}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#E5C79E" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#E5C79E" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF08" />
            <XAxis
              dataKey="time"
              tick={{ fill: '#8E9AAF', fontSize: 9 }}
              axisLine={{ stroke: '#8E9AAF15' }}
              interval={Math.max(1, Math.floor(chartData.length / 8))}
            />
            <YAxis
              tick={{ fill: '#8E9AAF', fontSize: 9 }}
              axisLine={{ stroke: '#8E9AAF15' }}
              width={35}
            />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Area
              type="monotone"
              dataKey="loss"
              stroke="#E5C79E"
              fill="url(#lossGrad)"
              strokeWidth={1.5}
              name="Middle Path Loss"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Spirit Velocity — activity/change rate */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-mono text-titan-growth/60 uppercase">Spirit Velocity</span>
          <span className="text-[10px] font-mono text-titan-metal/30">rate of inner change</span>
        </div>
        <ResponsiveContainer width="100%" height={80}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF08" />
            <XAxis
              dataKey="time"
              tick={{ fill: '#8E9AAF', fontSize: 9 }}
              axisLine={{ stroke: '#8E9AAF15' }}
              interval={Math.max(1, Math.floor(chartData.length / 8))}
            />
            <YAxis
              tick={{ fill: '#8E9AAF', fontSize: 9 }}
              axisLine={{ stroke: '#8E9AAF15' }}
              width={35}
            />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            <Line
              type="monotone"
              dataKey="velocity"
              stroke="#77CCCC"
              strokeWidth={1}
              name="Spirit Velocity"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-[9px] text-titan-metal/30">
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-titan-haze rounded" /> Middle Path Loss
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-titan-growth rounded" /> Spirit Velocity
        </span>
        <span className="ml-auto">{chartData.length} data points</span>
      </div>
    </div>
  );
}

'use client';

import { useGrowthHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { mockGrowthHistory, shouldUseMock } from '@/lib/mockData';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const METRICS = [
  { key: 'learning_velocity', label: 'Learning Velocity', color: '#9945FF', description: 'How fast Titan acquires new knowledge' },
  { key: 'social_density', label: 'Social Density', color: '#77CCCC', description: 'Depth and diversity of social interactions' },
  { key: 'metabolic_health', label: 'Metabolic Health', color: '#FF6B6B', description: 'Body tensor aggregate (SOL + resource health)' },
  { key: 'directive_alignment', label: 'Directive Alignment', color: '#E5C79E', description: 'Coherence with Maker prime directives' },
] as const;

const tooltipStyle = {
  backgroundColor: '#1A1D23',
  border: '1px solid #8E9AAF30',
  borderRadius: '8px',
  fontSize: 11,
};

export default function GrowthDashboard() {
  const titanId = useTitanId();
  const { data: rawHistory, isLoading } = useGrowthHistory(7, titanId);

  const history = shouldUseMock(rawHistory) ? mockGrowthHistory : rawHistory!;

  if (isLoading) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Growth Metrics
        </h3>
        <LoadingSkeleton lines={4} />
      </div>
    );
  }

  if (!history || history.length === 0) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Growth Metrics
        </h3>
        <p className="text-xs text-titan-metal/40">No growth data yet — metrics accumulate over time</p>
      </div>
    );
  }

  // Latest values for gauges
  const latest = history[history.length - 1];

  // Sparkline data (downsample to ~24 points for smooth mini charts)
  const step = Math.max(1, Math.floor(history.length / 24));
  const sparkData = history.filter((_, i) => i % step === 0 || i === history.length - 1).map((s) => ({
    time: new Date(s.ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    learning_velocity: s.learning_velocity,
    social_density: s.social_density,
    metabolic_health: s.metabolic_health,
    directive_alignment: s.directive_alignment,
  }));

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
        Growth Metrics (7d)
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {METRICS.map((metric) => {
          const currentVal = (latest as unknown as Record<string, number>)[metric.key] ?? 0;

          return (
            <div key={metric.key} className="space-y-2">
              {/* Gauge */}
              <div className="text-center">
                <div className="relative w-16 h-16 mx-auto">
                  <svg viewBox="0 0 36 36" className="w-16 h-16 -rotate-90">
                    {/* Background circle */}
                    <circle
                      cx="18" cy="18" r="15.9"
                      fill="none"
                      stroke="#8E9AAF15"
                      strokeWidth="3"
                    />
                    {/* Value arc */}
                    <circle
                      cx="18" cy="18" r="15.9"
                      fill="none"
                      stroke={metric.color}
                      strokeWidth="3"
                      strokeDasharray={`${currentVal * 100} ${100 - currentVal * 100}`}
                      strokeLinecap="round"
                      className="transition-all duration-1000"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-sm font-semibold" style={{ color: metric.color }}>
                      {(currentVal * 100).toFixed(0)}
                    </span>
                  </div>
                </div>
                <p className="text-[10px] text-titan-metal/60 mt-1">{metric.label}</p>
              </div>

              {/* Sparkline */}
              <ResponsiveContainer width="100%" height={40}>
                <AreaChart data={sparkData}>
                  <Area
                    type="monotone"
                    dataKey={metric.key}
                    stroke={metric.color}
                    fill={metric.color}
                    fillOpacity={0.1}
                    strokeWidth={1.5}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>

      {/* Combined trend line */}
      <div className="mt-4 pt-3 border-t border-titan-metal/10">
        <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider mb-2">Combined Growth Trajectory</p>
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={sparkData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
            <XAxis dataKey="time" tick={{ fill: '#8E9AAF', fontSize: 9 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#8E9AAF', fontSize: 9 }} axisLine={{ stroke: '#8E9AAF20' }} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#8E9AAF' }} />
            {METRICS.map((m) => (
              <Area
                key={m.key}
                type="monotone"
                dataKey={m.key}
                name={m.label}
                stroke={m.color}
                fill={m.color}
                fillOpacity={0.05}
                strokeWidth={1.5}
                dot={false}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

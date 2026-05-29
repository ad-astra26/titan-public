'use client';

interface MetricCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
  accent?: 'haze' | 'growth' | 'pulse' | 'metal';
  className?: string;
}

const accentMap = {
  haze: 'border-titan-haze/20 shadow-haze-glow',
  growth: 'border-titan-growth/20 shadow-growth-glow',
  pulse: 'border-titan-pulse/20 shadow-pulse-glow',
  metal: 'border-titan-metal/20',
};

export default function MetricCard({
  label,
  value,
  sublabel,
  accent = 'metal',
  className = '',
}: MetricCardProps) {
  return (
    <div
      className={`bg-titan-card/60 backdrop-blur-sm border rounded-xl p-4 ${accentMap[accent]} ${className}`}
    >
      <p className="text-[11px] text-titan-metal/60 uppercase tracking-wider font-medium">
        {label}
      </p>
      <p className="text-2xl font-semibold text-titan-metal mt-1">{value}</p>
      {sublabel && (
        <p className="text-xs text-titan-metal/40 mt-0.5">{sublabel}</p>
      )}
    </div>
  );
}

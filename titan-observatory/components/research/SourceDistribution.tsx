'use client';

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const COLORS: Record<string, string> = {
  Web: '#77CCCC',
  X: '#4488FF',
  Document: '#E5C79E',
};

interface SourceDistributionProps {
  distribution: Record<string, number>;
}

export default function SourceDistribution({
  distribution,
}: SourceDistributionProps) {
  const data = Object.entries(distribution).map(([name, value]) => ({
    name,
    value,
  }));

  if (data.length === 0) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Source Distribution
        </h3>
        <p className="text-xs text-titan-metal/40 text-center py-4">No data</p>
      </div>
    );
  }

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
        Source Distribution
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={3}
            dataKey="value"
          >
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={COLORS[entry.name] || '#8E9AAF'}
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: '#1A1D23',
              border: '1px solid #8E9AAF30',
              borderRadius: '8px',
              fontSize: 11,
            }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-4 mt-2">
        {data.map((entry) => (
          <div key={entry.name} className="flex items-center gap-1.5 text-[10px] text-titan-metal/60">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: COLORS[entry.name] || '#8E9AAF' }}
            />
            {entry.name}: {entry.value}
          </div>
        ))}
      </div>
    </div>
  );
}

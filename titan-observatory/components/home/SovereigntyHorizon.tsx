'use client';

import { useHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

function getNarrative(history: { sovereignty_pct: number; sol_balance: number }[]): string {
  if (history.length < 2) return 'Gathering data...';
  const first = history[0];
  const last = history[history.length - 1];
  const sovUp = last.sovereignty_pct > first.sovereignty_pct;
  const solUp = last.sol_balance > first.sol_balance;

  if (sovUp && solUp) return 'Golden Age: Full power';
  if (sovUp && !solUp) return 'Efficiency: Growing stronger with less';
  if (!sovUp && solUp) return 'Dependency: Resources up but autonomy declining';
  return 'Recalibrating: Adjusting to current conditions';
}

export default function SovereigntyHorizon() {
  const titanId = useTitanId();
  const { data: history, isLoading } = useHistory(7, titanId);

  if (isLoading) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Sovereignty Horizon (7d)
        </h3>
        <LoadingSkeleton lines={4} />
      </div>
    );
  }

  const validHistory = history || [];
  const chartData = validHistory.map((pt) => ({
    time: new Date(pt.timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    }),
    sovereignty: pt.sovereignty_pct,
    sol: pt.sol_balance,
    energy: pt.energy_state,
  }));

  const narrative = getNarrative(validHistory);

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
        Sovereignty Horizon (7d)
      </h3>
      {chartData.length > 0 ? (
        <>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#8E9AAF10" />
              <XAxis
                dataKey="time"
                tick={{ fill: '#8E9AAF', fontSize: 10 }}
                axisLine={{ stroke: '#8E9AAF20' }}
              />
              <YAxis
                yAxisId="left"
                tick={{ fill: '#E5C79E', fontSize: 10 }}
                axisLine={{ stroke: '#E5C79E20' }}
                domain={[0, 100]}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: '#77CCCC', fontSize: 10 }}
                axisLine={{ stroke: '#77CCCC20' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1A1D23',
                  border: '1px solid #8E9AAF30',
                  borderRadius: '8px',
                  fontSize: 11,
                }}
                labelStyle={{ color: '#8E9AAF' }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="sovereignty"
                stroke="#E5C79E"
                strokeWidth={2}
                dot={false}
                name="Sovereignty %"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="sol"
                stroke="#77CCCC"
                strokeWidth={2}
                dot={false}
                name="SOL Balance"
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-titan-haze/70 mt-3 italic">{narrative}</p>
        </>
      ) : (
        <p className="text-xs text-titan-metal/40">No historical data available</p>
      )}
    </div>
  );
}

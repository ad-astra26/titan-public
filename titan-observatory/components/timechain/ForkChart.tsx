'use client';

import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import type { TimeChainStatus, PoTStats } from '@/hooks/useTitanAPI';
import InfoTooltip from '@/components/shared/InfoTooltip';

interface Props {
  status: TimeChainStatus | undefined;
  potStats: PoTStats | undefined;
  onSelectFork: (forkId: number) => void;
}

const FORK_COLORS: Record<string, string> = {
  main: '#9945FF',
  declarative: '#8E9AAF',
  procedural: '#E5C79E',
  episodic: '#77CCCC',
  meta: '#FF6B6B',
  conversation: '#00FF88',
};

const FORK_IDS: Record<string, number> = {
  main: 0, declarative: 1, procedural: 2, episodic: 3, meta: 4, conversation: 5,
};

const SOURCE_DESCRIPTIONS: Record<string, string> = {
  expression: 'Emotional/creative expressions (art, music, social)',
  reasoning_chain: 'Logical reasoning steps and conclusions',
  meta_reasoning: 'Self-reflection — thinking about thinking',
  heartbeat: 'Periodic consciousness checkpoints',
  dream: 'Dream cycle events and consolidations',
  expression_speak: 'Language production (Titan speaking)',
  cgn_dream_consolidation: 'Neural network dream training',
  dream_distillation: 'Wisdom extracted from dream cycles',
  knowledge: 'Facts and concepts learned',
  output_verifier: 'OVG-signed verified outputs',
};

export default function ForkChart({ status, potStats, onSelectFork }: Props) {
  if (!status?.forks) return null;

  const primaryForks = Object.entries(status.forks)
    .filter(([, f]) => f.type === 'primary' && f.block_count > 0)
    .sort((a, b) => b[1].block_count - a[1].block_count);

  // Use log scale for visual balance
  const chartData = primaryForks.map(([id, fork]) => ({
    name: fork.name,
    blocks: fork.block_count,
    logBlocks: Math.log10(fork.block_count + 1),
    chi: fork.total_chi_spent,
    significance: fork.avg_significance,
    forkId: FORK_IDS[fork.name] ?? parseInt(id),
  }));

  const sources = potStats?.blocks_by_source
    ? Object.entries(potStats.blocks_by_source)
        .sort((a, b) => b[1].blocks - a[1].blocks)
        .slice(0, 10)
    : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Fork Distribution */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <h3 className="text-sm font-semibold text-titan-metal/70 uppercase tracking-wider">
            Fork Distribution
          </h3>
          <InfoTooltip text="Each fork stores a different type of cognitive content. Click a bar to view its recent blocks below." />
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={chartData} layout="vertical">
            <XAxis
              type="number"
              dataKey="logBlocks"
              tick={{ fill: '#8E9AAF', fontSize: 11 }}
              tickFormatter={() => ''}
              hide
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#E5C79E', fontSize: 12, fontFamily: 'JetBrains Mono' }}
              width={115}
            />
            <Tooltip
              contentStyle={{
                background: '#0B0E14',
                border: '1px solid #8E9AAF40',
                borderRadius: 8,
                color: '#E5C79E',
                fontSize: 13,
                boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
              }}
              itemStyle={{ color: '#E5C79E' }}
              cursor={{ fill: 'transparent' }}
              formatter={(_value: unknown, _name: unknown, item: { payload?: { blocks?: number; chi?: number; significance?: number } }) => {
                const p = item?.payload;
                return [
                  `${(p?.blocks ?? 0).toLocaleString()} blocks | chi: ${(p?.chi ?? 0).toFixed(2)} | sig: ${(p?.significance ?? 0).toFixed(2)}`,
                  'Details',
                ];
              }}
            />
            <Bar
              dataKey="logBlocks"
              radius={[0, 6, 6, 0]}
              cursor="pointer"
              onClick={(_data, index) => {
                if (index !== undefined && chartData[index]) onSelectFork(chartData[index].forkId);
              }}
              label={((props: { x?: number; y?: number; width?: number; index?: number }) => (
                <text
                  x={(props.x ?? 0) + (props.width ?? 0) + 6}
                  y={(props.y ?? 0) + 14}
                  fill="#8E9AAF"
                  fontSize={11}
                  fontFamily="JetBrains Mono"
                >
                  {chartData[props.index ?? 0]?.blocks.toLocaleString()}
                </text>
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
              )) as any}
            >
              {chartData.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={FORK_COLORS[entry.name] || '#8E9AAF'}
                  fillOpacity={0.8}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Thought Sources */}
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <h3 className="text-sm font-semibold text-titan-metal/70 uppercase tracking-wider">
            Thought Sources
          </h3>
          <InfoTooltip text="Proof of Thought — breakdown of what types of cognitive events are recorded on the TimeChain. Each thought must pass the PoT threshold to be admitted." />
        </div>
        <div className="space-y-2.5">
          {sources.map(([source, stats]) => {
            const pct = potStats ? (stats.blocks / potStats.total_blocks) * 100 : 0;
            const desc = SOURCE_DESCRIPTIONS[source] || source;
            // Log scale: ensures small sources are still visible
            // Map log(blocks) to 10%-100% bar width
            const maxLog = Math.log10(Math.max(...sources.map(([, s]) => s.blocks), 1));
            const logPct = maxLog > 0 ? (Math.log10(stats.blocks + 1) / maxLog) * 100 : 0;
            const barWidth = Math.max(8, logPct); // minimum 8% so all bars are visible
            return (
              <div key={source} className="relative group">
                <div className="flex items-center gap-3">
                  <span className="text-xs text-titan-haze/80 w-40 truncate font-mono cursor-help" title={desc}>
                    {source}
                  </span>
                  <div className="flex-1 h-2.5 bg-titan-metal/10 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${barWidth}%`,
                        background: `linear-gradient(90deg, #77CCCC, #9945FF)`,
                        opacity: 0.75,
                      }}
                    />
                  </div>
                  <span className="text-xs text-titan-metal/60 w-28 text-right font-mono">
                    {stats.blocks.toLocaleString()} <span className="text-titan-metal/35">({pct.toFixed(1)}%)</span>
                  </span>
                </div>
                {/* Floating tooltip */}
                <div className="absolute left-0 -bottom-8 z-50 hidden group-hover:block pointer-events-none">
                  <span className="bg-[#0B0E14] border border-titan-metal/30 rounded-md px-3 py-1.5 text-[11px] text-titan-haze/80 whitespace-nowrap shadow-xl shadow-black/50">
                    {desc} &middot; sig: {stats.avg_significance.toFixed(2)} &middot; {pct.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

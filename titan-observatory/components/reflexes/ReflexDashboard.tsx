'use client';

import { useReflexes, useReflexHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

/** State register bar — body/mind/spirit with colored fill */
function RegisterBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-titan-metal/60 w-12 shrink-0">{label}</span>
      <div className="flex-1 h-3 bg-titan-bg/60 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${(value * 100).toFixed(1)}%` }}
        />
      </div>
      <span className="text-xs font-mono text-titan-metal/70 w-12 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export default function ReflexDashboard({ titanId: propTitanId }: { titanId?: TitanId }) {
  const hookTitanId = useTitanId();
  const titanId = propTitanId ?? hookTitanId;
  const { data: reflexes, isLoading: rLoading } = useReflexes(titanId);
  const { data: history, isLoading: hLoading } = useReflexHistory(titanId);

  if (rLoading && hLoading) return <LoadingSkeleton lines={5} />;

  const stats = reflexes?.stats_24h ?? history?.stats;
  const reg = reflexes?.state_register;
  const executors = reflexes?.collector?.registered_executors ?? [];
  const successRate = stats && stats.total_fires > 0
    ? ((stats.total_successes / stats.total_fires) * 100).toFixed(0)
    : '---';

  return (
    <div className="flex flex-col gap-5">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="24h Fires"
          value={stats?.total_fires ?? 0}
          sublabel={stats?.total_fires === 0 ? 'no reflexes triggered' : undefined}
          accent="pulse"
        />
        <MetricCard
          label="Success Rate"
          value={stats?.total_fires ? `${successRate}%` : 'N/A'}
          sublabel={stats?.total_fires === 0 ? 'awaiting first fire' : undefined}
          accent="growth"
        />
        <MetricCard
          label="Avg Reward"
          value={stats?.total_fires ? stats.avg_reward.toFixed(3) : 'N/A'}
          sublabel={stats?.total_fires === 0 ? 'no reward signal yet' : undefined}
          accent="haze"
        />
        <MetricCard
          label="Executors"
          value={executors.length}
          accent="metal"
        />
      </div>

      {/* State Register */}
      {reg && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            Perceptual State Register
            <span className="ml-2 text-xs font-normal text-titan-metal/40">
              age: {reg.age_seconds.toFixed(1)}s
            </span>
          </h3>
          <div className="flex flex-col gap-2.5">
            <RegisterBar label="Body" value={reg.body_avg} color="bg-emerald-400" />
            <RegisterBar label="Mind" value={reg.mind_avg} color="bg-cyan-400" />
            <RegisterBar label="Spirit" value={reg.spirit_avg} color="bg-violet-400" />
          </div>
        </div>
      )}

      {/* Thresholds & Config */}
      {reflexes?.collector && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">Reflex Configuration</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div>
              <span className="text-titan-metal/50">Fire threshold</span>
              <p className="font-mono text-titan-metal mt-0.5">{reflexes.collector.fire_threshold}</p>
            </div>
            <div>
              <span className="text-titan-metal/50">Action threshold</span>
              <p className="font-mono text-titan-metal mt-0.5">{reflexes.collector.action_threshold}</p>
            </div>
            <div>
              <span className="text-titan-metal/50">Public threshold</span>
              <p className="font-mono text-titan-metal mt-0.5">{reflexes.collector.public_action_threshold}</p>
            </div>
            <div>
              <span className="text-titan-metal/50">Cooldown</span>
              <p className="font-mono text-titan-metal mt-0.5">{reflexes.collector.session_cooldown}s</p>
            </div>
          </div>
        </div>
      )}

      {/* Registered Executors */}
      {executors.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            Registered Executors
            <span className="ml-2 text-xs font-normal text-titan-metal/40">{executors.length} active</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {executors.map((ex) => (
              <span
                key={ex}
                className="px-2.5 py-1 text-xs font-mono rounded-lg bg-titan-bg/60 text-titan-metal/70 border border-titan-metal/10"
              >
                {ex}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Per-type stats */}
      {stats?.per_type && Object.keys(stats.per_type).length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">Firing by Type (24h)</h3>
          <div className="space-y-2">
            {Object.entries(stats.per_type).map(([type, data]) => {
              const d = data as { fires: number; successes: number; avg_reward: number };
              return (
                <div key={type} className="flex items-center justify-between text-xs">
                  <span className="font-mono text-titan-metal/70">{type}</span>
                  <div className="flex items-center gap-4">
                    <span className="text-titan-metal/50">{d.fires} fires</span>
                    <span className="text-titan-growth/70">{d.successes} ok</span>
                    <span className="text-titan-haze/70">r={d.avg_reward.toFixed(3)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Recent history entries */}
      {history?.entries && history.entries.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            Recent Reflex Firings
            <span className="ml-2 text-xs font-normal text-titan-metal/40">{history.count} total</span>
          </h3>
          <div className="space-y-1.5 max-h-64 overflow-y-auto">
            {history.entries.slice(0, 30).map((e, i) => (
              <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-titan-metal/5">
                <div className="flex items-center gap-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${e.success ? 'bg-emerald-400' : 'bg-red-400'}`} />
                  <span className="font-mono text-titan-metal/70">{e.type}</span>
                </div>
                <div className="flex items-center gap-3 text-titan-metal/50">
                  <span>{e.duration_ms}ms</span>
                  <span>r={e.reward.toFixed(3)}</span>
                  <span>{new Date(e.timestamp * 1000).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

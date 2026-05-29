'use client';

import { usePiHeartbeat } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

function HeartbeatBar({ size, maxSize }: { size: number; maxSize: number }) {
  const h = Math.max(8, (size / Math.max(maxSize, 1)) * 100);
  return (
    <div className="flex flex-col items-center gap-0.5" title={`Cluster: ${size} π-epochs`}>
      <div
        className="w-3 rounded-sm bg-titan-haze transition-all duration-300"
        style={{ height: `${h}%`, opacity: 0.5 + (size / Math.max(maxSize, 1)) * 0.5 }}
      />
      <span className="text-[8px] font-mono text-titan-metal/30">{size}</span>
    </div>
  );
}

export default function PiHeartbeatStrip() {
  const titanId = useTitanId();
  const { data, isLoading } = usePiHeartbeat(titanId);
  const pi = (data ?? {}) as Record<string, unknown>;
  const clusterSizes = Array.isArray(pi?.recent_cluster_sizes)
    ? pi.recent_cluster_sizes as number[]
    : Array.isArray(pi?.cluster_sizes) ? pi.cluster_sizes as number[] : [];
  const devAge = typeof pi?.developmental_age === 'number' ? pi.developmental_age : 0;
  const ratio = typeof pi?.heartbeat_ratio === 'number' ? pi.heartbeat_ratio : 0;
  const clusterCount = typeof pi?.cluster_count === 'number' ? pi.cluster_count : 0;
  const inCluster = pi?.in_cluster === true;
  const piStreak = typeof pi?.current_pi_streak === 'number' ? pi.current_pi_streak : 0;
  const zeroStreak = typeof pi?.current_zero_streak === 'number' ? pi.current_zero_streak : 0;
  const totalPi = typeof pi?.total_pi_epochs === 'number' ? pi.total_pi_epochs : 0;
  const totalObserved = typeof pi?.total_epochs_observed === 'number' ? pi.total_epochs_observed : 0;
  const avgSize = typeof pi?.avg_cluster_size === 'number' ? pi.avg_cluster_size : 0;

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 animate-pulse"><div className="h-4 bg-titan-metal/10 rounded w-32" /></div>;
  }

  const maxSize = Math.max(...clusterSizes, 1);

  return (
    <div className="bg-titan-card rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider">Pi-Heartbeat Pattern</h3>
          <p className="text-[10px] text-titan-metal/30 mt-0.5">
            Emergent self-integration rhythm from Unified Spirit curvature
          </p>
        </div>
        <div className="flex items-center gap-2">
          {inCluster ? (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-titan-haze/15 border border-titan-haze/20">
              <span className="w-1.5 h-1.5 rounded-full bg-titan-haze animate-pulse" />
              <span className="text-[10px] font-mono text-titan-haze">BEATING ({piStreak})</span>
            </span>
          ) : (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-titan-metal/5 border border-titan-metal/10">
              <span className="w-1.5 h-1.5 rounded-full bg-titan-metal/30" />
              <span className="text-[10px] font-mono text-titan-metal/40">resting ({zeroStreak})</span>
            </span>
          )}
        </div>
      </div>

      {/* Visual heartbeat bar chart */}
      {clusterSizes.length > 0 && (
        <div className="mb-4">
          <div className="flex items-end gap-1 h-16 px-1">
            {clusterSizes.slice(-20).map((size, i) => (
              <HeartbeatBar key={i} size={size} maxSize={maxSize} />
            ))}
          </div>
          <div className="flex justify-between mt-1 text-[8px] font-mono text-titan-metal/20 px-1">
            <span>oldest</span>
            <span>recent clusters</span>
            <span>latest</span>
          </div>
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
        <div>
          <span className="text-[10px] text-titan-metal/40 uppercase">Dev Age</span>
          <p className="font-mono text-lg text-titan-haze">{devAge}</p>
        </div>
        <div>
          <span className="text-[10px] text-titan-metal/40 uppercase">Clusters</span>
          <p className="font-mono text-lg text-titan-metal">{clusterCount}</p>
        </div>
        <div>
          <span className="text-[10px] text-titan-metal/40 uppercase">Avg Size</span>
          <p className="font-mono text-lg text-titan-metal">{avgSize.toFixed(1)}</p>
        </div>
        <div>
          <span className="text-[10px] text-titan-metal/40 uppercase">Ratio</span>
          <p className="font-mono text-lg text-titan-metal">{(ratio * 100).toFixed(1)}%</p>
        </div>
        <div>
          <span className="text-[10px] text-titan-metal/40 uppercase">Pi Epochs</span>
          <p className="font-mono text-lg text-titan-metal">{totalPi.toLocaleString()}</p>
        </div>
        <div>
          <span className="text-[10px] text-titan-metal/40 uppercase">Total</span>
          <p className="font-mono text-lg text-titan-metal">{totalObserved.toLocaleString()}</p>
        </div>
      </div>

      {/* Explanatory text */}
      <p className="text-[10px] text-titan-metal/25 mt-3 leading-relaxed">
        When Unified Spirit curvature aligns with pi (3.14...), Titan enters a &quot;heartbeat&quot; cluster —
        a burst of self-integrating epochs. The pattern is emergent, not programmed.
        Each cluster represents a moment of deep internal coherence.
      </p>
    </div>
  );
}

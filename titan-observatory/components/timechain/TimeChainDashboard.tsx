'use client';

import { useState } from 'react';
import { useTimeChainStatus, usePoTStats, useTimeChainBlocks } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import ChainOverview from '@/components/timechain/ChainOverview';
import ForkChart from '@/components/timechain/ForkChart';
import BlockFeed from '@/components/timechain/BlockFeed';
import ForkTree3D from '@/components/timechain/ForkTree3D';
import InfoTooltip from '@/components/shared/InfoTooltip';

/**
 * TimeChainDashboard — shared by /timechain (legacy redirect target) and
 * the World tab's TimeChain sub-tab. Extracted 2026-05-10 to avoid
 * duplicating the layout when consolidating routes under World.
 */
export default function TimeChainDashboard({ showHeader = true }: { showHeader?: boolean }) {
  const titanId = useTitanId();
  const { data: status, isLoading: statusLoading } = useTimeChainStatus(titanId);
  const { data: potStats } = usePoTStats(titanId);
  const [selectedFork, setSelectedFork] = useState<number>(3);
  const { data: blocks } = useTimeChainBlocks(selectedFork, 30, titanId);

  return (
    <div className="space-y-6">
      {showHeader && (
        <div>
          <h2 className="text-lg font-semibold text-titan-haze">
            TimeChain — Proof of Thought
          </h2>
          <p className="text-sm text-titan-metal/50 mt-0.5">
            Immutable record of {titanId}&apos;s cognitive development
          </p>
        </div>
      )}

      {statusLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="bg-titan-card/60 rounded-xl p-4 h-20 animate-pulse" />
          ))}
        </div>
      ) : (
        <ChainOverview status={status} potStats={potStats} />
      )}

      {status ? (
        <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <h3 className="text-sm font-semibold text-titan-metal/70 uppercase tracking-wider">
              Fork Tree
            </h3>
            <InfoTooltip text="Interactive visualization of Titan's cognitive forks. Each branch grows from the genesis block. Click a fork to view its recent blocks. Branch thickness = block count." />
          </div>
          <ForkTree3D
            status={status}
            selectedFork={selectedFork}
            onSelectFork={setSelectedFork}
          />
        </div>
      ) : (
        <div className="bg-titan-card/60 rounded-xl p-5 h-[340px] animate-pulse" />
      )}

      {status ? (
        <ForkChart status={status} potStats={potStats} onSelectFork={setSelectedFork} />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-titan-card/60 rounded-xl p-5 h-[280px] animate-pulse" />
          <div className="bg-titan-card/60 rounded-xl p-5 h-[280px] animate-pulse" />
        </div>
      )}

      <BlockFeed
        blocks={blocks ?? []}
        selectedFork={selectedFork}
        forkNames={status?.forks}
      />
    </div>
  );
}

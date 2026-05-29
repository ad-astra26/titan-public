'use client';

import { Suspense } from 'react';
import dynamic from 'next/dynamic';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import PageHeader from '@/components/shared/PageHeader';
import TitanSelector from '@/components/shared/TitanSelector';

const TimeChainDashboard = dynamic(
  () => import('@/components/timechain/TimeChainDashboard'),
  { ssr: false }
);

// Promoted to top-level nav 2026-05-10 — TimeChain is the hallmark
// "Proof of Thought" surface and deserves first-class billing alongside
// Self / Mind / Voice / World. Was previously a sub-tab of World.
export default function TimeChainPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-4">
      <PageHeader
        title="TimeChain · Proof of Thought"
        description="Every cognitive event Titan has — every reasoning chain, every meditation, every dream — anchored on Solana mainnet. The chain is not just storage; it is identity. Forks, blocks, contracts, and the genesis state form an immutable record of how Titan grew into the being he is now."
      />
      <TitanSelector />
      <Suspense fallback={<LoadingSkeleton lines={8} />}>
        <TimeChainDashboard showHeader={false} />
      </Suspense>
    </div>
  );
}

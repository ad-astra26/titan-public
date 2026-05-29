'use client';

import dynamic from 'next/dynamic';

const TrinityRadar = dynamic(() => import('@/components/stats/TrinityRadar'), { ssr: false });
const TrinityTrends = dynamic(() => import('@/components/stats/TrinityTrends'), { ssr: false });

/**
 * Trinity sub-tab under Self. Hosts the live tensor metrics (Divine
 * Trinity radar across Body 5D / Mind 15D / Spirit 45D + Inner/Outer
 * overlay) and the longitudinal Trinity trends. Moved here from
 * Stats/System on 2026-05-10 — these are Trinity metrics and belong
 * under Self · Trinity, not under World · System.
 */
export default function TrinityMetricsTab() {
  return (
    <div className="flex flex-col gap-6">
      <TrinityRadar />
      <TrinityTrends />
    </div>
  );
}

'use client';

import dynamic from 'next/dynamic';
import PiHeartbeatStrip from '@/components/rhythms/PiHeartbeatStrip';
import DreamingGauge from '@/components/rhythms/DreamingGauge';

const SphereClocksDetail = dynamic(
  () => import('@/components/rhythms/SphereClocksDetail'),
  { ssr: false }
);
const CircadianTimeline = dynamic(
  () => import('@/components/rhythms/CircadianTimeline'),
  { ssr: false }
);

export default function RhythmsTab() {
  return (
    <div className="flex flex-col gap-4">
      <SphereClocksDetail />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PiHeartbeatStrip />
        <DreamingGauge />
      </div>
      <CircadianTimeline />
    </div>
  );
}

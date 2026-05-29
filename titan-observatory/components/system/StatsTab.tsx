'use client';

import dynamic from 'next/dynamic';
import { useHealth, useStatus } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { useTitanStore } from '@/store/titanStore';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

const CapabilityMatrix = dynamic(() => import('@/components/stats/CapabilityMatrix'), { ssr: false });
const IntegrityBadge = dynamic(() => import('@/components/stats/IntegrityBadge'), { ssr: false });
const SubsystemGrid = dynamic(() => import('@/components/stats/SubsystemGrid'), { ssr: false });
const GuardianActivity = dynamic(() => import('@/components/stats/GuardianActivity'), { ssr: false });
const ArcCompetition = dynamic(() => import('@/components/stats/ArcCompetition'), { ssr: false });
const HistoryCharts = dynamic(() => import('@/components/stats/HistoryCharts'), { ssr: false });
const GrowthDashboard = dynamic(() => import('@/components/stats/GrowthDashboard'), { ssr: false });
// TrinityRadar + TrinityTrends moved to Self · Trinity sub-tab on 2026-05-10
// — they were Trinity metrics misfiled under System.

export default function StatsTab() {
  const titanId = useTitanId();
  const { data: health, isLoading } = useHealth(titanId);
  const { data: statusData } = useStatus(titanId);
  const status = useTitanStore((s) => s.status) || statusData;

  if (isLoading) return <LoadingSkeleton lines={6} />;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Version" value={health?.version ?? '--'} accent="metal" />
        <MetricCard label="Status" value={health?.status ?? '--'} accent="growth" />
        <MetricCard label="Persistent Memories" value={status?.memory_count ?? '--'} accent="haze" />
        <MetricCard label="Privacy Redactions" value={health?.privacy_filter?.redactions ?? 0} sublabel={health?.privacy_filter?.enabled ? 'Enabled' : 'Disabled'} accent="pulse" />
      </div>

      <GrowthDashboard />
      <ArcCompetition />
      <HistoryCharts />

      {health?.capabilities && <CapabilityMatrix capabilities={health.capabilities} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {health?.subsystems && <SubsystemGrid subsystems={health.subsystems} />}
        </div>
        <IntegrityBadge vault={status?.vault ?? null} />
      </div>

      <GuardianActivity />
    </div>
  );
}

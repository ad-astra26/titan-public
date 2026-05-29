'use client';

import StatusBadge from '@/components/shared/StatusBadge';

interface SubsystemGridProps {
  subsystems: Record<string, 'ACTIVE' | 'ABSENT' | 'DEGRADED'>;
}

const SUBSYSTEM_ORDER = [
  'memory',
  'metabolism',
  'soul',
  'guardian',
  'gatekeeper',
  'studio',
  'social',
  'observatory',
];

export default function SubsystemGrid({ subsystems }: SubsystemGridProps) {
  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
        Subsystem Health
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {SUBSYSTEM_ORDER.map((name) => {
          const status = subsystems[name] || 'ABSENT';
          return (
            <div
              key={name}
              className="flex flex-col items-center gap-2 bg-titan-bg/40 rounded-lg p-3"
            >
              <span className="text-xs text-titan-metal/70 capitalize">{name}</span>
              <StatusBadge status={status as 'ACTIVE' | 'ABSENT' | 'DEGRADED' | 'STUB'} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

'use client';

import StatusBadge from '@/components/shared/StatusBadge';
import { CapabilityEntry } from '@/lib/types';

interface CapabilityMatrixProps {
  capabilities: CapabilityEntry[];
}

export default function CapabilityMatrix({ capabilities }: CapabilityMatrixProps) {
  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
        Capability Matrix
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {capabilities.map((cap) => (
          <div
            key={cap.name}
            className="flex items-center justify-between bg-titan-bg/40 rounded-lg px-3 py-2"
          >
            <span className="text-xs text-titan-metal/70 truncate mr-2">
              {cap.name}
            </span>
            <StatusBadge status={cap.status} />
          </div>
        ))}
      </div>
    </div>
  );
}

'use client';

import { useGuardian } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatTimestamp } from '@/lib/formatters';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

export default function GuardianActivity() {
  const titanId = useTitanId();
  const { data: guardian, isLoading } = useGuardian(titanId);

  if (isLoading) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-4">
          Guardian Activity
        </h3>
        <LoadingSkeleton lines={3} />
      </div>
    );
  }

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
          Guardian Activity
        </h3>
        <span className="text-[10px] text-titan-metal/40">
          Total blocks: {guardian?.total_blocks ?? 0}
        </span>
      </div>

      {(!guardian?.recent_actions || guardian.recent_actions.length === 0) ? (
        <p className="text-xs text-titan-metal/40">No recent guardian actions</p>
      ) : (
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {guardian.recent_actions.map((action, i) => (
            <div
              key={i}
              className="flex items-center gap-2 text-xs bg-titan-bg/30 rounded-lg px-3 py-2"
            >
              <span
                className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                  action.tier === 'keyword'
                    ? 'bg-red-500/20 text-red-400'
                    : action.tier === 'semantic'
                      ? 'bg-yellow-500/20 text-yellow-400'
                      : 'bg-titan-pulse/20 text-titan-pulse'
                }`}
              >
                {action.tier}
              </span>
              <span className="text-titan-metal/60 truncate flex-1">
                {action.action} ({action.category})
              </span>
              <span className="text-titan-metal/40 shrink-0">
                {formatTimestamp(action.timestamp)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

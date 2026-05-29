'use client';

import { useState } from 'react';
import { useGuardian } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatTimestamp } from '@/lib/formatters';

export default function GuardianShield() {
  const titanId = useTitanId();
  const { data: guardian } = useGuardian(titanId);
  const [expanded, setExpanded] = useState(false);

  const recentBlocks = guardian?.recent_actions?.length ?? 0;
  const shieldColor =
    recentBlocks === 0
      ? 'text-titan-growth'
      : recentBlocks < 3
        ? 'text-yellow-400'
        : 'text-red-400';

  return (
    <div className="relative">
      <button
        onClick={() => setExpanded(!expanded)}
        className={`p-1.5 rounded-lg hover:bg-titan-card/50 transition-colors ${shieldColor}`}
        title="Guardian Shield"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          className="w-5 h-5"
        >
          <path
            fillRule="evenodd"
            d="M12.516 2.17a.75.75 0 00-1.032 0 11.209 11.209 0 01-7.877 3.08.75.75 0 00-.722.515A12.74 12.74 0 002.25 9.75c0 5.942 4.064 10.933 9.563 12.348a.749.749 0 00.374 0c5.499-1.415 9.563-6.406 9.563-12.348 0-1.39-.223-2.73-.635-3.985a.75.75 0 00-.722-.516 11.209 11.209 0 01-7.877-3.08z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {expanded && guardian?.recent_actions && (
        <div className="absolute right-0 top-full mt-2 w-72 bg-titan-card border border-titan-metal/20 rounded-lg shadow-xl z-50 p-3">
          <h4 className="text-xs font-semibold text-titan-haze uppercase tracking-wider mb-2">
            Guardian Activity
          </h4>
          {guardian.recent_actions.length === 0 ? (
            <p className="text-xs text-titan-metal/60">No recent actions</p>
          ) : (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {guardian.recent_actions.slice(0, 10).map((action, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 text-xs"
                >
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                      action.tier === 'keyword'
                        ? 'bg-red-500/20 text-red-400'
                        : action.tier === 'semantic'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-titan-pulse/20 text-titan-pulse'
                    }`}
                  >
                    {action.tier}
                  </span>
                  <span className="text-titan-metal/70 truncate flex-1">
                    {action.action} ({action.category})
                  </span>
                  <span className="text-titan-metal/40 shrink-0">
                    {formatTimestamp(action.timestamp)}
                  </span>
                </div>
              ))}
            </div>
          )}
          <div className="mt-2 pt-2 border-t border-titan-metal/10 text-[10px] text-titan-metal/40">
            Total blocks: {guardian.total_blocks}
          </div>
        </div>
      )}
    </div>
  );
}

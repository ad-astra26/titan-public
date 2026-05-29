'use client';

import { useDreaming } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function DreamingOverlay({ children }: { children: React.ReactNode }) {
  const titanId = useTitanId();
  const { data } = useDreaming(titanId);
  const isDreaming = data?.is_dreaming === true;
  const cycleCount = data?.cycle_count ?? 0;
  const recoveryPct = data?.recovery_pct ?? 0;

  return (
    <div className="relative">
      {children}
      {isDreaming && (
        <div className="absolute inset-0 bg-blue-900/15 rounded-xl pointer-events-none z-10 flex items-start justify-center pt-4">
          <div className="bg-blue-900/80 backdrop-blur-sm px-4 py-2 rounded-full flex items-center gap-2">
            <span className="text-blue-300 text-sm">DREAMING</span>
            <span className="font-mono text-xs text-blue-400">cycle {cycleCount} &middot; {recoveryPct.toFixed(0)}% recovered</span>
          </div>
        </div>
      )}
    </div>
  );
}

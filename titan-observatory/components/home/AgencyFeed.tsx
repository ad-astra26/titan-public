'use client';

import { useAgency } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

interface Assessment {
  action_id: number;
  impulse_id: number;
  score: number;
  reflection: string;
  enrichment: Record<string, Record<string, number>>;
  mood_delta: number;
  threshold_direction: string;
  ts: number;
}

interface AgencyStats {
  action_count: number;
  budget_remaining: number;
  budget_per_hour: number;
  llm_calls_this_hour: number;
  helper_statuses: Record<string, string>;
}

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 0.7
      ? 'text-teal-400 bg-teal-400/10'
      : score >= 0.4
      ? 'text-yellow-400 bg-yellow-400/10'
      : 'text-red-400 bg-red-400/10';

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono ${color}`}>
      {score.toFixed(2)}
    </span>
  );
}

function DirectionIcon({ direction }: { direction: string }) {
  if (direction === 'lower') return <span className="text-teal-400 text-xs">↓ more autonomy</span>;
  if (direction === 'raise') return <span className="text-red-400 text-xs">↑ more caution</span>;
  return <span className="text-titan-metal/40 text-xs">= hold</span>;
}

function EnrichmentPills({ enrichment }: { enrichment: Record<string, Record<string, number>> }) {
  if (!enrichment || Object.keys(enrichment).length === 0) return null;

  return (
    <div className="flex gap-1 flex-wrap mt-1">
      {Object.entries(enrichment).map(([layer, dims]) =>
        Object.entries(dims).map(([dim, delta]) => (
          <span
            key={`${layer}-${dim}`}
            className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
              delta > 0
                ? 'text-teal-300 bg-teal-400/10'
                : 'text-red-300 bg-red-400/10'
            }`}
          >
            {layer}[{dim}] {delta > 0 ? '+' : ''}{delta.toFixed(3)}
          </span>
        ))
      )}
    </div>
  );
}

function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export default function AgencyFeed() {
  const titanId = useTitanId();
  const { data, isLoading } = useAgency(titanId);

  if (isLoading || !data) {
    return (
      <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-metal mb-2">Agency</h3>
        <p className="text-xs text-titan-metal/50">Loading...</p>
      </div>
    );
  }

  const agencyData = data as { enabled: boolean; agency?: AgencyStats; assessment?: { total: number; avg_score: number; recent: Assessment[] } };

  if (!agencyData.enabled) {
    return (
      <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-metal mb-2">Agency</h3>
        <p className="text-xs text-titan-metal/50">Disabled</p>
      </div>
    );
  }

  const agency = agencyData.agency;
  const assessment = agencyData.assessment;
  const recent = assessment?.recent ?? [];

  return (
    <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-titan-metal">
          Autonomous Actions
        </h3>
        <div className="flex items-center gap-2 text-[10px] text-titan-metal/50">
          <span>
            {agency?.action_count ?? 0} actions
          </span>
          <span className="text-titan-metal/30">|</span>
          <span>
            avg {(assessment?.avg_score ?? 0).toFixed(2)}
          </span>
          <span className="text-titan-metal/30">|</span>
          <span>
            budget {agency?.budget_remaining ?? 0}/{agency?.budget_per_hour ?? 0}
          </span>
        </div>
      </div>

      {/* Assessment feed */}
      {recent.length === 0 ? (
        <p className="text-xs text-titan-metal/50 italic">
          No autonomous actions yet — impulse engine is observing...
        </p>
      ) : (
        <div className="space-y-2">
          {[...recent].reverse().map((a) => (
            <div
              key={a.action_id}
              className="bg-titan-bg/50 border border-titan-metal/5 rounded px-3 py-2"
            >
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs text-titan-metal leading-relaxed flex-1">
                  {a.reflection.length > 500
                    ? a.reflection.slice(0, 500) + '...'
                    : a.reflection}
                </p>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <ScoreBadge score={a.score} />
                  <DirectionIcon direction={a.threshold_direction} />
                </div>
              </div>
              <div className="flex items-center justify-between mt-1.5">
                <EnrichmentPills enrichment={a.enrichment} />
                <span className="text-[10px] text-titan-metal/30">
                  {timeAgo(a.ts)}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

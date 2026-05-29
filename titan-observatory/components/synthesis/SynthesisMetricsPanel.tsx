'use client';

// Synthesis Engine metrics panel (Phase 10 / D-SPEC-PHASE10).
// Surfaces the headline sovereignty ratio + groundedness heatmap + skill
// library + retrieval p99 / chi + chain-growth, read from GET
// /v6/synthesis/metrics (snapshot-backed, observation-only INV-Syn-25).

import { useSynthesisMetrics } from '@/hooks/useTitanAPI';
import type { TitanId } from '@/lib/api';

function pct(n: number | undefined): string {
  if (n === undefined || n === null) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

function TrendArrow({ trend }: { trend: number | null | undefined }) {
  if (trend === null || trend === undefined) return <span className="text-zinc-500">—</span>;
  if (trend > 0.0001) return <span className="text-emerald-400">▲ {(trend * 100).toFixed(1)}pp</span>;
  if (trend < -0.0001) return <span className="text-rose-400">▼ {(Math.abs(trend) * 100).toFixed(1)}pp</span>;
  return <span className="text-zinc-400">▬ flat</span>;
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-400">{title}</h3>
      {children}
    </div>
  );
}

export default function SynthesisMetricsPanel({ titanId }: { titanId?: TitanId }) {
  const { data, isLoading, isError } = useSynthesisMetrics(titanId);

  if (isLoading) return <div className="p-6 text-zinc-500" data-testid="synthesis-loading">Loading synthesis metrics…</div>;
  if (isError) return <div className="p-6 text-rose-400" data-testid="synthesis-error">Failed to load synthesis metrics.</div>;

  const snapshot = data?.snapshot ?? 'missing';
  const m = data?.metrics;
  const sov = m?.sovereignty?.windows;
  const skills = m?.skills;
  const retrieval = m?.retrieval;
  const chi = m?.chi;
  const chain = m?.chain_growth;
  const heatmap = m?.groundedness?.heatmap ?? [];

  return (
    <div className="flex flex-col gap-4" data-testid="synthesis-metrics-panel">
      {snapshot !== 'ok' && (
        <div className="rounded border border-amber-700/50 bg-amber-900/20 px-3 py-2 text-xs text-amber-300">
          metrics snapshot: <b>{snapshot}</b> — values may be empty until the synthesis_worker recompute fires.
        </div>
      )}

      {/* Headline sovereignty ratio */}
      <Card title="Sovereignty Ratio (recall vs LLM re-derivation)">
        {sov ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {(['24h', '7d', 'all'] as const).map((w) => {
              const row = sov[w];
              return (
                <div key={w} className="rounded bg-zinc-800/50 p-3" data-testid={`sov-${w}`}>
                  <div className="text-xs text-zinc-500">{w}</div>
                  <div className="text-2xl font-bold text-cyan-300">{pct(row?.ratio)}</div>
                  <div className="text-xs"><TrendArrow trend={row?.trend} /></div>
                  <div className="mt-1 text-[11px] text-zinc-500">
                    {row?.recall_satisfied ?? 0}/{row?.knowledge_moments ?? 0} moments
                    {' · '}{row?.cited_recalls ?? 0} cited · {row?.skill_delegations ?? 0} skill
                  </div>
                </div>
              );
            })}
          </div>
        ) : <div className="text-zinc-500">no sovereignty data</div>}
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Skill library */}
        <Card title="Skill Library">
          {skills?.available ? (
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <dt className="text-zinc-500">Size</dt><dd className="text-right text-zinc-200">{skills.size ?? 0}</dd>
              <dt className="text-zinc-500">Verified</dt><dd className="text-right text-zinc-200">{skills.verified_count ?? 0}</dd>
              <dt className="text-zinc-500">Mean utility</dt><dd className="text-right text-zinc-200">{(skills.mean_utility ?? 0).toFixed(3)}</dd>
              <dt className="text-zinc-500">Success ratio</dt><dd className="text-right text-zinc-200">{skills.success_ratio == null ? '—' : pct(skills.success_ratio)}</dd>
            </dl>
          ) : <div className="text-zinc-500">no compiled skills yet</div>}
        </Card>

        {/* Retrieval p99 + chi */}
        <Card title="Retrieval latency + χ budget">
          <dl className="grid grid-cols-2 gap-2 text-sm">
            <dt className="text-zinc-500">p50 / p95 / p99</dt>
            <dd className="text-right text-zinc-200">
              {retrieval?.overall
                ? `${retrieval.overall.p50} / ${retrieval.overall.p95} / ${retrieval.overall.p99} ms`
                : '—'}
            </dd>
            <dt className="text-zinc-500">samples</dt>
            <dd className="text-right text-zinc-200">{retrieval?.samples ?? 0}{retrieval?.warming ? ' (warming)' : ''}</dd>
            <dt className="text-zinc-500">χ spent / cap</dt>
            <dd className="text-right text-zinc-200">{chi?.spent != null ? `${chi.spent} / ${chi.cap ?? '—'}` : '—'}</dd>
          </dl>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Groundedness heatmap */}
        <Card title="Groundedness (top concepts)">
          {heatmap.length ? (
            <div className="flex flex-col gap-1">
              {heatmap.slice(0, 12).map((c) => (
                <div key={c.concept_id} className="flex items-center gap-2 text-xs">
                  <span className="w-28 truncate text-zinc-400">{c.name || c.concept_id}</span>
                  <div className="h-2 flex-1 rounded bg-zinc-800">
                    <div className="h-2 rounded bg-cyan-500" style={{ width: `${Math.round((c.groundedness || 0) * 100)}%` }} />
                  </div>
                  <span className="w-10 text-right text-zinc-300">{(c.groundedness ?? 0).toFixed(2)}</span>
                </div>
              ))}
            </div>
          ) : <div className="text-zinc-500">no concept groundedness yet</div>}
        </Card>

        {/* Chain growth */}
        <Card title="Chain growth (B.7 bounded)">
          {chain?.available ? (
            <div className="text-sm text-zinc-200">
              {((chain.total_bytes ?? 0) / 1_048_576).toFixed(2)} MB on disk
            </div>
          ) : <div className="text-zinc-500">no chain-growth sample yet</div>}
        </Card>
      </div>
    </div>
  );
}

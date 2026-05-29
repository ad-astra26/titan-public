'use client';

import { useArcStatus } from '@/hooks/useTitanAPI';
import type { ArcGameResult } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

function GameCard({ gameId, game }: { gameId: string; game: ArcGameResult }) {
  return (
    <div className="bg-titan-bg rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-mono text-titan-haze font-semibold uppercase">{gameId}</span>
        <span className="text-xs font-mono text-titan-metal/50">{game.num_episodes} episodes</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <span className="text-[10px] font-mono text-titan-metal/40 uppercase">Best Reward</span>
          <div className="text-lg font-mono text-titan-growth">{game.best_reward.toFixed(1)}</div>
        </div>
        <div>
          <span className="text-[10px] font-mono text-titan-metal/40 uppercase">Avg Reward</span>
          <div className="text-lg font-mono text-titan-haze">{game.avg_reward.toFixed(2)}</div>
        </div>
        <div>
          <span className="text-[10px] font-mono text-titan-metal/40 uppercase">Avg Steps</span>
          <div className="text-sm font-mono text-titan-metal">{game.avg_steps.toFixed(0)}</div>
        </div>
        <div>
          <span className="text-[10px] font-mono text-titan-metal/40 uppercase">Best Levels</span>
          <div className="text-sm font-mono text-titan-metal">{game.best_levels}</div>
        </div>
      </div>

      <div className="mt-2 text-[10px] font-mono text-titan-metal/30">
        duration: {game.duration_s.toFixed(0)}s
      </div>
    </div>
  );
}

function LevelScoreBar({ scores }: { scores: number[] }) {
  return (
    <div className="flex gap-0.5 items-end h-6">
      {scores.map((score, i) => (
        <div
          key={i}
          className={`flex-1 rounded-sm transition-all duration-500 ${
            score >= 1.0 ? 'bg-titan-growth' :
            score > 0 ? 'bg-titan-haze/60' :
            'bg-titan-metal/15'
          }`}
          style={{ height: `${Math.max(15, score * 100)}%` }}
          title={`Level ${i + 1}: ${(score * 100).toFixed(0)}%`}
        />
      ))}
    </div>
  );
}

export default function ArcCompetition() {
  const titanId = useTitanId();
  const { data: arc } = useArcStatus(titanId);

  if (!arc || !arc.active) {
    return (
      <div className="bg-titan-card rounded-xl p-6">
        <h3 className="text-sm font-mono uppercase tracking-wider text-titan-metal/60 mb-2">
          ARC-AGI-3 Competition
        </h3>
        <p className="text-xs text-titan-metal/40">No active competition data</p>
      </div>
    );
  }

  const results = arc.results;
  const games = results?.games ?? {};
  const scorecard = results?.scorecard;
  const nsPrograms = results?.ns_programs ?? [];

  // Flatten all runs from scorecard for level visualization
  const allRuns = scorecard?.environments?.flatMap(env => env.runs) ?? [];
  const bestRun = allRuns.reduce((best, run) =>
    (run.levels_completed > (best?.levels_completed ?? 0)) ? run : best,
    allRuns[0] ?? null
  );

  return (
    <div className="bg-titan-card rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-mono uppercase tracking-wider text-titan-metal/60">
            ARC-AGI-3 Competition
          </h3>
          <span className="text-[10px] font-mono text-titan-metal/30">
            {results?.mode ?? 'train'} mode · {results?.episodes_per_game ?? 0} episodes/game · max {results?.max_steps ?? 0} steps
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-titan-growth/10 border border-titan-growth/20">
            <span className="w-1.5 h-1.5 rounded-full bg-titan-growth animate-pulse" />
            <span className="text-[10px] font-mono text-titan-growth">LIVE</span>
          </span>
          {scorecard && (
            <span className="text-xs font-mono text-titan-haze">
              score: {scorecard.score.toFixed(1)}
            </span>
          )}
        </div>
      </div>

      {/* NS Programs used */}
      {nsPrograms.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {nsPrograms.map(prog => (
            <span
              key={prog}
              className="px-2 py-0.5 text-[10px] font-mono rounded-full bg-titan-pulse/10 text-titan-pulse/70 border border-titan-pulse/15"
            >
              {prog}
            </span>
          ))}
        </div>
      )}

      {/* Game results */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
        {Object.entries(games).map(([id, game]) => (
          <GameCard key={id} gameId={id} game={game} />
        ))}
      </div>

      {/* Best run level scores */}
      {bestRun && bestRun.level_scores && bestRun.level_scores.length > 0 && (
        <div className="bg-titan-bg rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono text-titan-metal/50 uppercase">
              Best Run — Level Progress
            </span>
            <span className="text-[10px] font-mono text-titan-metal/40">
              {bestRun.levels_completed} completed · {bestRun.actions} actions · {bestRun.resets} resets
            </span>
          </div>
          <LevelScoreBar scores={bestRun.level_scores} />
          <div className="flex justify-between mt-1 text-[8px] font-mono text-titan-metal/30">
            <span>Level 1</span>
            <span>Level {bestRun.level_scores.length}</span>
          </div>
        </div>
      )}

      {/* Scorer stats */}
      {Object.keys(arc.scorers).length > 0 && (
        <div className="flex gap-4 mt-3 text-[10px] font-mono text-titan-metal/40">
          {Object.entries(arc.scorers).map(([game, scorer]) => (
            <span key={game}>
              {game}: {scorer.total_updates.toLocaleString()} updates, loss={scorer.last_loss.toFixed(4)}
            </span>
          ))}
        </div>
      )}

      {results?.timestamp && (
        <div className="mt-2 text-[9px] font-mono text-titan-metal/20 text-right">
          last updated: {new Date(results.timestamp).toLocaleString()}
        </div>
      )}
    </div>
  );
}

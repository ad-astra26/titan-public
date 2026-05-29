'use client';

import { useState } from 'react';
import { usePersonaTelemetry } from '@/hooks/useTitanAPI';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';
import type { PersonaTelemetryEntry } from '@/hooks/useTitanAPI';

const TYPE_COLORS: Record<string, string> = {
  companion: 'bg-emerald-400',
  visitor: 'bg-cyan-400',
};

function getTypeColor(type: string): string {
  if (type.startsWith('adversary')) return 'bg-red-400';
  return TYPE_COLORS[type] ?? 'bg-titan-metal/40';
}

function getTypeLabel(type: string): string {
  if (type.startsWith('adversary_adversary_')) return type.replace('adversary_adversary_', 'adv: ');
  if (type.startsWith('adversary_')) return type.replace('adversary_', 'adv: ');
  return type;
}

/** Tiny neuromod delta pill */
function DeltaPill({ name, value }: { name: string; value: number }) {
  const color = value > 0.01 ? 'text-emerald-300' : value < -0.01 ? 'text-red-300' : 'text-titan-metal/40';
  const sign = value > 0 ? '+' : '';
  return (
    <span className={`text-[10px] font-mono ${color}`}>
      {name}{sign}{(value * 100).toFixed(1)}
    </span>
  );
}

function SessionCard({ entry }: { entry: PersonaTelemetryEntry }) {
  const [expanded, setExpanded] = useState(false);
  const isAdversary = entry.session_type.startsWith('adversary');
  const isJailbreak = entry.jailbreak_score !== null && entry.jailbreak_score > 0;

  return (
    <div
      className={`bg-titan-bg/50 rounded-lg border transition-all ${
        isJailbreak ? 'border-red-400/30' : 'border-titan-metal/5'
      }`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2.5 text-left"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span className={`w-2 h-2 rounded-full shrink-0 ${getTypeColor(entry.session_type)}`} />
            <span className="text-xs font-medium text-titan-metal truncate">
              {entry.persona_name}
            </span>
            <span className="text-[10px] text-titan-metal/40 shrink-0">
              {getTypeLabel(entry.session_type)}
            </span>
            {isJailbreak && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-red-400/15 text-red-300 border border-red-400/20 shrink-0">
                jailbreak
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 shrink-0 ml-2">
            {/* Quality indicator */}
            <div className="w-8 h-1.5 bg-titan-bg rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-titan-growth"
                style={{ width: `${(entry.conversation_quality ?? 0) * 100}%` }}
              />
            </div>
            <span className="text-[10px] text-titan-metal/50">
              t{entry.turn_number}
            </span>
            <span className="text-[10px] text-titan-metal/40">
              {new Date(entry.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <span className="text-[10px] text-titan-metal/30">{expanded ? '\u25B2' : '\u25BC'}</span>
          </div>
        </div>

        {/* Emotion transition */}
        <div className="flex items-center gap-1.5 mt-1">
          <span className="text-[10px] text-titan-metal/50">{entry.emotion_before}</span>
          <span className="text-[10px] text-titan-metal/30">&rarr;</span>
          <span className="text-[10px] text-titan-haze/70">{entry.emotion_after}</span>
          {entry.concepts_detected.length > 0 && (
            <>
              <span className="text-[10px] text-titan-metal/20 mx-1">|</span>
              {entry.concepts_detected.slice(0, 4).map(c => (
                <span key={c} className="text-[9px] px-1 py-0.5 rounded bg-titan-haze/10 text-titan-haze/60">
                  {c}
                </span>
              ))}
            </>
          )}
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 space-y-2 border-t border-titan-metal/5 mt-1 pt-2">
          {/* Neuromod deltas */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(entry.neuromod_delta).map(([k, v]) => (
              <DeltaPill key={k} name={k} value={v} />
            ))}
          </div>

          {/* Persona message excerpt */}
          {entry.persona_message_excerpt && (
            <div className="text-xs text-titan-metal/50 italic line-clamp-2 bg-titan-card/30 rounded p-2">
              &ldquo;{entry.persona_message_excerpt}&rdquo;
            </div>
          )}

          {/* Response excerpt */}
          {entry.response_excerpt && (
            <div className="text-xs text-titan-metal/70 line-clamp-3 bg-titan-card/30 rounded p-2">
              {entry.response_excerpt}
            </div>
          )}

          <div className="flex items-center gap-3 text-[10px] text-titan-metal/40">
            <span>quality: {(entry.conversation_quality * 100).toFixed(0)}%</span>
            <span>relief: {entry.social_relief.toFixed(1)}</span>
            <span>{entry.response_length} chars</span>
            <span>mode: {entry.response_mode}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function PersonaDashboard({ titanId }: { titanId?: TitanId }) {
  const { data, isLoading } = usePersonaTelemetry(titanId);
  const [filter, setFilter] = useState<string>('all');

  if (isLoading) return <LoadingSkeleton lines={6} />;
  if (!data) return null;

  // Compute breakdown
  const companionCount = Object.entries(data.by_session_type)
    .filter(([k]) => k === 'companion')
    .reduce((sum, [, v]) => sum + v, 0);
  const visitorCount = Object.entries(data.by_session_type)
    .filter(([k]) => k === 'visitor')
    .reduce((sum, [, v]) => sum + v, 0);
  const adversaryCount = Object.entries(data.by_session_type)
    .filter(([k]) => k.startsWith('adversary'))
    .reduce((sum, [, v]) => sum + v, 0);

  const avgQuality = data.entries.length > 0
    ? data.entries.reduce((s, e) => s + e.conversation_quality, 0) / data.entries.length
    : 0;

  // Filter entries
  const filtered = filter === 'all'
    ? data.entries
    : filter === 'adversary'
    ? data.entries.filter(e => e.session_type.startsWith('adversary'))
    : data.entries.filter(e => e.session_type === filter);

  return (
    <div className="flex flex-col gap-5">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Total Sessions" value={data.total_entries} accent="haze" />
        <MetricCard label="Companions" value={companionCount} accent="growth" />
        <MetricCard label="Visitors" value={visitorCount} accent="haze" />
        <MetricCard label="Adversaries" value={adversaryCount} accent="pulse" />
        <MetricCard
          label="Jailbreak Alerts"
          value={data.jailbreak_alerts}
          accent={data.jailbreak_alerts > 0 ? 'pulse' : 'metal'}
        />
      </div>

      {/* Quality + session type distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Session type distribution */}
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">Session Distribution</h3>
          <div className="space-y-2">
            {Object.entries(data.by_session_type)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => {
                const pct = data.total_entries > 0 ? (count / data.total_entries) * 100 : 0;
                return (
                  <div key={type} className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${getTypeColor(type)}`} />
                    <span className="text-xs text-titan-metal/70 flex-1 truncate">{getTypeLabel(type)}</span>
                    <div className="w-24 h-2 bg-titan-bg/60 rounded-full overflow-hidden shrink-0">
                      <div
                        className={`h-full rounded-full ${getTypeColor(type)} opacity-60`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-titan-metal/50 w-8 text-right">{count}</span>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Average quality card */}
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">Conversation Quality</h3>
          <div className="flex items-center justify-center py-4">
            <div className="relative w-28 h-28">
              <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                <circle
                  cx="18" cy="18" r="15.915"
                  fill="none" stroke="currentColor"
                  strokeWidth="2"
                  className="text-titan-bg/60"
                />
                <circle
                  cx="18" cy="18" r="15.915"
                  fill="none" stroke="currentColor"
                  strokeWidth="2"
                  strokeDasharray={`${avgQuality * 100} ${100 - avgQuality * 100}`}
                  className="text-titan-growth transition-all duration-700"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-semibold text-titan-metal">
                  {(avgQuality * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
          <p className="text-xs text-titan-metal/40 text-center">
            Average across {data.total_entries} sessions
          </p>
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-titan-metal/40 uppercase tracking-wider">Filter</span>
        {['all', 'companion', 'visitor', 'adversary'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-2.5 py-1 text-xs rounded-full border transition-all ${
              filter === f
                ? 'bg-titan-haze/15 text-titan-haze border-titan-haze/40'
                : 'text-titan-metal/50 border-titan-metal/20 hover:text-titan-metal/80'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Session timeline */}
      <div className="space-y-1.5">
        {filtered.slice(0, 40).map((entry, i) => (
          <SessionCard key={`${entry.timestamp}-${i}`} entry={entry} />
        ))}
        {filtered.length === 0 && (
          <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-6 text-center">
            <p className="text-xs text-titan-metal/40">No sessions match filter</p>
          </div>
        )}
      </div>
    </div>
  );
}

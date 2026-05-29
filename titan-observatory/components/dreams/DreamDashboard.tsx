'use client';

import { useDreamInbox, useDreaming } from '@/hooks/useTitanAPI';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import type { TitanId } from '@/lib/api';

function FatigueBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(value * 100, 100);
  const color = pct > 70 ? 'bg-red-400' : pct > 40 ? 'bg-amber-400' : 'bg-emerald-400';
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-titan-metal/60 w-20 shrink-0">{label}</span>
      <div className="flex-1 h-3 bg-titan-bg/60 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-titan-metal/70 w-12 text-right">
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

export default function DreamDashboard({ titanId }: { titanId?: TitanId }) {
  const { data: dreaming, isLoading: dLoading } = useDreaming(titanId);
  const { data: inbox, isLoading: iLoading } = useDreamInbox(titanId);

  if (dLoading && iLoading) return <LoadingSkeleton lines={5} />;

  const dreamState = inbox?.dream_state;
  const isDreaming = dreaming?.is_dreaming ?? dreamState?.is_dreaming ?? false;

  return (
    <div className="flex flex-col gap-5">
      {/* Dream state banner */}
      <div className={`rounded-xl p-4 border ${
        isDreaming
          ? 'bg-violet-500/10 border-violet-400/30'
          : 'bg-titan-card/40 border-titan-metal/10'
      }`}>
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${isDreaming ? 'bg-violet-400 animate-pulse' : 'bg-emerald-400'}`} />
          <span className="text-sm font-semibold text-titan-haze">
            {isDreaming ? 'Dreaming' : 'Awake'}
          </span>
          {dreamState?.wake_transition && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30">
              waking up
            </span>
          )}
          {dreamState?.just_woke && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-400/20 text-cyan-300 border border-cyan-400/30">
              just woke
            </span>
          )}
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Sleep Cycles"
          value={dreaming?.cycle_count ?? 0}
          accent="haze"
        />
        <MetricCard
          label="Dream Epochs"
          value={dreaming?.dream_epochs ?? 0}
          sublabel={dreaming?.epochs_since_dream !== undefined ? `${dreaming.epochs_since_dream} since last` : undefined}
          accent="pulse"
        />
        <MetricCard
          label="Dev. Age"
          value={dreaming?.developmental_age ?? 0}
          accent="growth"
        />
        <MetricCard
          label="Dream Inbox"
          value={inbox?.inbox_count ?? 0}
          accent={inbox?.inbox_count ? 'pulse' : 'metal'}
        />
      </div>

      {/* Fatigue & Recovery */}
      {dreaming && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">Fatigue & Recovery</h3>
          <div className="flex flex-col gap-2.5">
            <FatigueBar label="Fatigue" value={dreaming.fatigue} />
            <FatigueBar label="Recovery" value={dreaming.recovery_pct} />
            {dreaming.onset_fatigue > 0 && (
              <FatigueBar label="Onset" value={dreaming.onset_fatigue} />
            )}
          </div>
          {dreaming.remaining_epochs > 0 && (
            <p className="text-xs text-titan-metal/50 mt-3">
              {dreaming.remaining_epochs} epochs until wake
            </p>
          )}
        </div>
      )}

      {/* Dream Inbox Messages */}
      {inbox?.messages && inbox.messages.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            Dream Inbox
            <span className="ml-2 text-xs font-normal text-titan-metal/40">
              {inbox.inbox_count} messages
            </span>
          </h3>
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {inbox.messages.map((msg, i) => (
              <div key={i} className="bg-titan-bg/50 rounded-lg px-3 py-2 border border-titan-metal/5">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono text-violet-300">{msg.channel}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-titan-metal/40">
                      pri {msg.priority}
                    </span>
                    <span className="text-[10px] text-titan-metal/40">
                      {new Date(msg.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
                <p className="text-xs text-titan-metal/70 line-clamp-2">{msg.preview}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {(!inbox?.messages || inbox.messages.length === 0) && !dLoading && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-6 text-center">
          <p className="text-xs text-titan-metal/40">
            {isDreaming
              ? 'Dream inbox is empty. Consolidation in progress...'
              : 'No dream messages queued. Dreams accumulate during waking hours.'
            }
          </p>
        </div>
      )}
    </div>
  );
}

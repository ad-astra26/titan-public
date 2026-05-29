'use client';

import { useDreaming, useDreamInbox, useNeuromodulators } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

function FatigueBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(value * 100, 100);
  const color = pct > 70 ? 'bg-red-400' : pct > 40 ? 'bg-amber-400' : 'bg-emerald-400';
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-titan-metal/60 w-20 shrink-0">{label}</span>
      <div className="flex-1 h-3 bg-titan-bg/60 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-700 ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-titan-metal/70 w-12 text-right">{pct.toFixed(1)}%</span>
    </div>
  );
}

export default function DreamingTab() {
  const titanId = useTitanId();
  const { data: dreamData, isLoading: dreamLoading } = useDreaming(titanId);
  const { data: inbox, isLoading: inboxLoading } = useDreamInbox(titanId);
  const { data: neuroData, isLoading: neuroLoading } = useNeuromodulators(titanId);

  if (dreamLoading && neuroLoading && inboxLoading) return <LoadingSkeleton lines={6} />;

  const isDreaming = dreamData?.is_dreaming ?? false;
  const fatigue = dreamData?.fatigue ?? 0;
  const cycleCount = dreamData?.cycle_count ?? 0;
  const dreamEpochs = dreamData?.dream_epochs ?? 0;
  const recoveryPct = dreamData?.recovery_pct ?? 0;
  const remainingEpochs = dreamData?.remaining_epochs ?? 0;
  const epochsSinceDream = dreamData?.epochs_since_dream ?? 0;
  const devAge = dreamData?.developmental_age ?? 0;
  const onsetFatigue = dreamData?.onset_fatigue ?? 0;
  const dreamState = inbox?.dream_state;

  // Extract neuromodulator levels for sleep/wake drive visualization
  const nm = (neuroData as Record<string, unknown>)?.modulators as Record<string, Record<string, number>> | undefined;
  const gaba = nm?.GABA?.level ?? 0;
  const ne = nm?.NE?.level ?? 0;
  const da = nm?.DA?.level ?? 0;
  const sleepPressure = gaba * 0.6 + fatigue * 0.4;
  const wakePressure = (ne + da) / 2;

  return (
    <div className="flex flex-col gap-4">
      {/* Dream state banner */}
      <div className={`rounded-xl p-4 border flex items-center gap-3 ${
        isDreaming
          ? 'bg-violet-500/10 border-violet-400/30'
          : 'bg-titan-card/40 border-titan-metal/10'
      }`}>
        <div className={`w-3 h-3 rounded-full ${isDreaming ? 'bg-violet-400 animate-pulse' : 'bg-emerald-400'}`} />
        <div>
          <span className={`text-sm font-semibold ${isDreaming ? 'text-indigo-300' : 'text-titan-haze'}`}>
            {isDreaming ? 'Dreaming' : 'Awake'}
          </span>
          <p className="text-xs text-titan-metal/50 mt-0.5">
            {isDreaming
              ? `Recovery: ${(recoveryPct * 100).toFixed(0)}% — ${remainingEpochs} epochs remaining`
              : `Fatigue: ${(fatigue * 100).toFixed(1)}% — ${cycleCount} dream cycles completed`
            }
          </p>
        </div>
        {dreamState?.wake_transition && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 ml-auto">
            waking up
          </span>
        )}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Fatigue" value={`${(fatigue * 100).toFixed(1)}%`} accent="pulse" />
        <MetricCard label="Dream Cycles" value={cycleCount} sublabel={`${dreamEpochs} epochs`} accent="haze" />
        <MetricCard label="Epochs Awake" value={epochsSinceDream} accent="metal" />
        <MetricCard label="Dev Age" value={devAge} sublabel={onsetFatigue > 0 ? `onset: ${(onsetFatigue * 100).toFixed(0)}%` : undefined} accent="growth" />
        <MetricCard label="Dream Inbox" value={inbox?.inbox_count ?? 0} accent={inbox?.inbox_count ? 'pulse' : 'metal'} />
      </div>

      {/* Fatigue & Recovery bars */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-3">Fatigue & Recovery</h3>
        <div className="flex flex-col gap-2.5">
          <FatigueBar label="Fatigue" value={fatigue} />
          <FatigueBar label="Recovery" value={recoveryPct} />
          {onsetFatigue > 0 && <FatigueBar label="Onset" value={onsetFatigue} />}
        </div>
      </div>

      {/* Sleep/Wake balance from neuromodulators */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-3">Sleep/Wake Balance</h3>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-xs text-titan-metal mb-1">
              <span>Sleep Pressure (GABA={gaba.toFixed(2)} + fatigue={fatigue.toFixed(2)})</span>
              <span>{sleepPressure.toFixed(3)}</span>
            </div>
            <div className="h-2 bg-titan-bg rounded-full overflow-hidden">
              <div className="h-full bg-indigo-400 rounded-full transition-all" style={{ width: `${Math.min(100, sleepPressure * 100)}%` }} />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs text-titan-metal mb-1">
              <span>Wake Pressure (NE={ne.toFixed(2)} + DA={da.toFixed(2)})</span>
              <span>{wakePressure.toFixed(3)}</span>
            </div>
            <div className="h-2 bg-titan-bg rounded-full overflow-hidden">
              <div className="h-full bg-amber-400 rounded-full transition-all" style={{ width: `${Math.min(100, wakePressure * 100)}%` }} />
            </div>
          </div>
        </div>
        <p className="text-[10px] text-titan-metal/40 mt-3">
          When GABA + accumulated fatigue exceeds NE + DA arousal, Titan enters a dreaming phase. Recovery happens through dream distillation.
        </p>
      </div>

      {/* Dream Inbox Messages */}
      {inbox?.messages && inbox.messages.length > 0 && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-titan-haze mb-3">
            Dream Inbox
            <span className="ml-2 text-xs font-normal text-titan-metal/40">{inbox.inbox_count} messages</span>
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {inbox.messages.map((msg, i) => (
              <div key={i} className="bg-titan-bg/50 rounded-lg px-3 py-2 border border-titan-metal/5">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono text-violet-300">{msg.channel}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-titan-metal/40">pri {msg.priority}</span>
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

      {/* Empty inbox state */}
      {(!inbox?.messages || inbox.messages.length === 0) && !inboxLoading && (
        <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4 text-center">
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

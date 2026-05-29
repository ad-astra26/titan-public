'use client';

import { useKinSignature, useKinSociety, KinEncounter, KinProfile } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import MetricCard from '@/components/shared/MetricCard';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';

function formatTime(ts: number): string {
  if (!ts) return '---';
  const d = new Date(ts * 1000);
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`;
}

function formatAgo(ts: number): string {
  if (!ts) return '---';
  const secs = Math.floor(Date.now() / 1000 - ts);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function resonanceColor(r: number): string {
  if (r >= 0.80) return 'text-yellow-300';
  if (r >= 0.65) return 'text-emerald-400';
  if (r >= 0.50) return 'text-cyan-400';
  return 'text-titan-metal';
}

function ResonanceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const color = value >= 0.80 ? 'bg-yellow-400' : value >= 0.65 ? 'bg-emerald-400' : 'bg-cyan-500';
  return (
    <div className="h-1.5 w-full bg-titan-card rounded-full overflow-hidden">
      <div className={`h-full ${color} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function KinProfileCard({ profile }: { profile: KinProfile }) {
  return (
    <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-titan-haze">{profile.name || 'Unknown Kin'}</h3>
          <p className="text-xs text-titan-metal mt-0.5">
            {profile.relationship_label || 'familiar_presence'} &middot; {profile.encounter_count} encounters
          </p>
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold ${resonanceColor(profile.avg_resonance)}`}>
            {(profile.avg_resonance * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-titan-metal uppercase tracking-wider">avg resonance</p>
        </div>
      </div>
      <ResonanceBar value={profile.avg_resonance} />
      <div className="grid grid-cols-3 gap-3 text-center text-xs">
        <div>
          <div className="text-titan-haze font-medium">{formatAgo(profile.first_encounter_ts)}</div>
          <div className="text-titan-metal">First Contact</div>
        </div>
        <div>
          <div className="text-titan-haze font-medium">{formatAgo(profile.last_encounter_ts)}</div>
          <div className="text-titan-metal">Latest</div>
        </div>
        <div>
          <div className="text-yellow-300 font-medium">{profile.great_kin_pulses}</div>
          <div className="text-titan-metal">Great Pulses</div>
        </div>
      </div>
    </div>
  );
}

function EncounterRow({ enc }: { enc: KinEncounter }) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-titan-card/30 transition-colors">
      <div className="text-xs text-titan-metal w-16 shrink-0 font-mono">{formatTime(enc.timestamp)}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-medium ${resonanceColor(enc.resonance)}`}>
            {(enc.resonance * 100).toFixed(1)}%
          </span>
          {enc.great_kin_pulse > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 bg-yellow-400/20 text-yellow-300 rounded-full font-bold uppercase">GKP</span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-1 text-xs">
        <span className="text-titan-metal">me:</span>
        <span className="text-titan-haze">{enc.my_emotion}</span>
      </div>
      <div className="flex items-center gap-1 text-xs">
        <span className="text-titan-metal">kin:</span>
        <span className="text-cyan-400">{enc.kin_emotion}</span>
      </div>
    </div>
  );
}

function ResonanceTimeline({ encounters }: { encounters: KinEncounter[] }) {
  if (encounters.length < 2) return null;
  const sorted = [...encounters].sort((a, b) => a.timestamp - b.timestamp);
  const min = Math.min(...sorted.map(e => e.resonance));
  const max = Math.max(...sorted.map(e => e.resonance));
  const range = Math.max(max - min, 0.01);
  const w = 600, h = 120, pad = 20;
  const points = sorted.map((e, i) => {
    const x = pad + (i / (sorted.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((e.resonance - min) / range) * (h - 2 * pad);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-titan-haze mb-3">Resonance Over Time</h3>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto" preserveAspectRatio="none">
        <defs>
          <linearGradient id="resGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgb(52,211,153)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="rgb(52,211,153)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={`${pad},${h - pad} ${points} ${w - pad},${h - pad}`} fill="url(#resGrad)" />
        <polyline points={points} fill="none" stroke="rgb(52,211,153)" strokeWidth="2" strokeLinejoin="round" />
        <text x={pad} y={12} fill="#8b8fa3" fontSize="10" fontFamily="monospace">{(max * 100).toFixed(1)}%</text>
        <text x={pad} y={h - 4} fill="#8b8fa3" fontSize="10" fontFamily="monospace">{(min * 100).toFixed(1)}%</text>
      </svg>
    </div>
  );
}

export default function KinSocietyTab() {
  const titanId = useTitanId();
  const { data: sig, isLoading: sigLoading } = useKinSignature(titanId);
  const { data: society, isLoading: socLoading } = useKinSociety(titanId);

  const encounters = society?.recent_encounters ?? [];
  const profiles = society?.profiles ?? [];
  const totalEnc = society?.total_encounters ?? 0;
  const totalGKP = society?.total_great_kin_pulses ?? 0;

  if (sigLoading && socLoading) return <LoadingSkeleton lines={8} />;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Total Encounters" value={totalEnc} />
        <MetricCard label="Great Kin Pulses" value={totalGKP} />
        <MetricCard label="My Emotion" value={sig?.emotion ?? '---'} />
        <MetricCard label="Epoch" value={sig?.epoch_id?.toLocaleString() ?? '---'} />
        <MetricCard label="Maturity" value={sig?.maturity ? `${(sig.maturity * 100).toFixed(0)}%` : '---'} />
      </div>

      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-2">My Signature</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div><span className="text-titan-metal">Name:</span> <span className="text-titan-haze font-medium">{sig?.name ?? 'Titan'}</span></div>
          <div><span className="text-titan-metal">Dev Age:</span> <span className="text-titan-haze font-medium">{sig?.developmental_age ?? 0} clusters</span></div>
          <div><span className="text-titan-metal">Chi:</span> <span className="text-titan-haze font-medium">{sig?.chi_total?.toFixed(3) ?? '---'}</span></div>
          <div><span className="text-titan-metal">Dreaming:</span> <span className={sig?.is_dreaming ? 'text-indigo-400 font-medium' : 'text-emerald-400 font-medium'}>{sig?.is_dreaming ? 'Yes' : 'Awake'}</span></div>
          <div className="col-span-2 md:col-span-4">
            <span className="text-titan-metal">Dominant Programs:</span> <span className="text-titan-haze font-medium">{(sig?.dominant_programs ?? []).join(', ') || '---'}</span>
          </div>
        </div>
      </div>

      {profiles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-titan-haze">Known Kin</h3>
          <div className="grid gap-3 md:grid-cols-2">
            {profiles.map((p) => <KinProfileCard key={p.pubkey || p.name} profile={p} />)}
          </div>
        </div>
      )}

      <ResonanceTimeline encounters={encounters} />

      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-titan-haze mb-3">
          Recent Encounters <span className="text-titan-metal font-normal ml-2 text-xs">({encounters.length} shown / {totalEnc} total)</span>
        </h3>
        <div className="divide-y divide-titan-metal/5 max-h-[400px] overflow-y-auto">
          {encounters.length === 0 ? (
            <p className="text-sm text-titan-metal py-4 text-center">No encounters yet. Waiting for kin...</p>
          ) : (
            encounters.map((enc) => <EncounterRow key={enc.id} enc={enc} />)
          )}
        </div>
      </div>
    </div>
  );
}

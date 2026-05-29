'use client';

import { useTitanStore } from '@/store/titanStore';
import { truncateHash } from '@/lib/formatters';
import SolscanLink from '@/components/shared/SolscanLink';

// Backend (Python `time.time()`) ships Unix epoch as float SECONDS.
// `new Date(N)` treats N as milliseconds — so a fresh 1.78e9-second
// timestamp gets interpreted as 1.78e9 ms ≈ Jan 1970. Detect & scale.
function formatTs(ts: number | string | null | undefined): string {
  if (ts == null) return 'N/A';
  const n = typeof ts === 'number' ? ts : Number(ts);
  if (!Number.isFinite(n) || n <= 0) return 'N/A';
  // Anything below 10^12 is clearly seconds-since-epoch (10^12 ms = year 2001).
  const ms = n < 1e12 ? n * 1000 : n;
  return new Date(ms).toLocaleString();
}

export default function VaultStatus() {
  const status = useTitanStore((s) => s.status);
  const vault = status?.vault;

  if (!vault) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-3">
          On-Chain Vault
        </h3>
        <p className="text-xs text-titan-metal/40">No vault data</p>
      </div>
    );
  }

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-pulse/15 rounded-xl p-5 shadow-pulse-glow/30">
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-3">
        On-Chain Vault
      </h3>
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-titan-metal/50">Commits</span>
          <span className="text-sm font-mono text-titan-pulse">{vault.commit_count}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-titan-metal/50">Last Commit</span>
          <span className="text-xs text-titan-metal/70 font-mono">
            {formatTs(vault.last_commit)}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-titan-metal/50">State Root</span>
          <span className="text-xs text-titan-metal/70 font-mono">
            {truncateHash(vault.latest_state_root, 8)}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-titan-metal/50">Sovereignty</span>
          <span className="text-xs text-titan-haze font-semibold">{Math.round(vault.sovereignty_pct)}%</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-titan-metal/50">Program</span>
          <SolscanLink address={vault.program_id} />
        </div>
      </div>
    </div>
  );
}

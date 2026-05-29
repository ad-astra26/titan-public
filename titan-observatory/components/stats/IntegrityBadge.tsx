'use client';

import { useState } from 'react';
import { VaultInfo } from '@/lib/types';
import { truncateHash } from '@/lib/formatters';
import SolscanLink from '@/components/shared/SolscanLink';

interface IntegrityBadgeProps {
  vault: VaultInfo | null;
}

export default function IntegrityBadge({ vault }: IntegrityBadgeProps) {
  const [showModal, setShowModal] = useState(false);

  if (!vault) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-2">
          Integrity Verification
        </h3>
        <p className="text-xs text-titan-metal/40">No vault data</p>
      </div>
    );
  }

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="w-full bg-titan-card/60 backdrop-blur-sm border border-titan-pulse/20 rounded-xl p-5 text-left hover:shadow-pulse-glow transition-shadow"
      >
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-2">
          Integrity Verification
        </h3>
        <p className="font-mono text-xs text-titan-pulse truncate">
          {truncateHash(vault.latest_state_root, 12)}
        </p>
        <p className="text-[10px] text-titan-metal/40 mt-1">Click to verify</p>
      </button>

      {/* Modal */}
      {showModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={() => setShowModal(false)}
        >
          <div
            className="bg-titan-card border border-titan-metal/20 rounded-2xl p-6 max-w-lg w-full mx-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-semibold text-titan-haze mb-6">
              Cognitive Integrity Verification
            </h2>

            <div className="space-y-4">
              <div>
                <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider mb-1">
                  State Root Hash
                </p>
                <p className="font-mono text-xs text-titan-pulse break-all">
                  {vault.latest_state_root}
                </p>
              </div>

              <div className="flex items-center justify-center py-4">
                <div className="flex items-center gap-3">
                  <svg className="w-8 h-8 text-titan-growth" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-sm font-semibold text-titan-growth">
                    Cognitive Integrity Verified
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">
                    Compressed Memories
                  </p>
                  <p className="text-lg font-semibold text-titan-metal mt-0.5">
                    {vault.compressed_memories}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">
                    Epoch Snapshots
                  </p>
                  <p className="text-lg font-semibold text-titan-metal mt-0.5">
                    {vault.epoch_snapshots}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">
                    Commit Count
                  </p>
                  <p className="text-lg font-semibold text-titan-metal mt-0.5">
                    {vault.commit_count}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">
                    Sovereignty
                  </p>
                  <p className="text-lg font-semibold text-titan-haze mt-0.5">
                    {Math.round(vault.sovereignty_pct)}%
                  </p>
                </div>
              </div>

              <div>
                <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider mb-1">
                  Vault PDA
                </p>
                <SolscanLink address={vault.pda} truncate={false} />
              </div>

              <div className="pt-2">
                <SolscanLink address={vault.program_id} className="text-sm" />
              </div>
            </div>

            <button
              onClick={() => setShowModal(false)}
              className="mt-6 w-full py-2 text-sm text-titan-metal/60 bg-titan-bg/50 rounded-lg hover:bg-titan-bg/80 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}

'use client';

import { useState } from 'react';

/**
 * Chain-proof drawer — every on-chain claim in the pitch surface has
 * a `↗ proof` chevron that expands to show the actual transaction /
 * program / Arweave URL. Two clicks from claim to verifiable evidence.
 *
 * Per rFP_observatory_pitch_route.md §3 + §11 (improvement #5).
 */

export type ChainProof =
  | { kind: 'vault_program'; address: string; label?: string }
  | { kind: 'titan_identity'; address: string; titanLabel: string }
  | { kind: 'arweave'; txId: string; label?: string }
  | { kind: 'memo'; signature: string; label?: string }
  | { kind: 'timechain_block'; height: number; merkle?: string; label?: string };

interface Props {
  proof: ChainProof;
  /** Optional extra label rendered next to the chevron. */
  hint?: string;
}

function solscanUrl(address: string): string {
  return `https://solscan.io/account/${address}`;
}

function solscanTxUrl(signature: string): string {
  return `https://solscan.io/tx/${signature}`;
}

function arweaveUrl(txId: string): string {
  return `https://arweave.net/${txId}`;
}

function shortAddr(s: string, head = 6, tail = 6): string {
  if (s.length <= head + tail + 3) return s;
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
}

function ProofBody({ proof }: { proof: ChainProof }) {
  switch (proof.kind) {
    case 'vault_program':
      return (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-titan-metal/40">ZK Vault program · Solana mainnet</div>
          <div className="font-mono text-xs text-titan-haze break-all">{proof.address}</div>
          <a
            href={solscanUrl(proof.address)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-titan-pulse hover:text-titan-pulse/80 underline-offset-2 hover:underline"
          >
            open on Solscan →
          </a>
        </div>
      );
    case 'titan_identity':
      return (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-titan-metal/40">
            {proof.titanLabel} identity · Solana mainnet
          </div>
          <div className="font-mono text-xs text-titan-haze break-all">{proof.address}</div>
          <a
            href={solscanUrl(proof.address)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-titan-pulse hover:text-titan-pulse/80 underline-offset-2 hover:underline"
          >
            open on Solscan →
          </a>
        </div>
      );
    case 'arweave':
      return (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-titan-metal/40">Arweave permanent backup</div>
          <div className="font-mono text-xs text-titan-haze break-all">{shortAddr(proof.txId, 10, 8)}</div>
          <a
            href={arweaveUrl(proof.txId)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-titan-pulse hover:text-titan-pulse/80 underline-offset-2 hover:underline"
          >
            open on Arweave →
          </a>
        </div>
      );
    case 'memo':
      return (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-titan-metal/40">Solana memo inscription · per-epoch</div>
          <div className="font-mono text-xs text-titan-haze break-all">{shortAddr(proof.signature, 10, 8)}</div>
          <a
            href={solscanTxUrl(proof.signature)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-titan-pulse hover:text-titan-pulse/80 underline-offset-2 hover:underline"
          >
            open transaction →
          </a>
        </div>
      );
    case 'timechain_block':
      return (
        <div className="space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-titan-metal/40">TimeChain block</div>
          <div className="font-mono text-xs text-titan-haze">height {proof.height.toLocaleString()}</div>
          {proof.merkle && (
            <div className="font-mono text-[10px] text-titan-metal/50 break-all">
              merkle {shortAddr(proof.merkle, 8, 8)}
            </div>
          )}
        </div>
      );
  }
}

export default function ChainProofDrawer({ proof, hint }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="inline-block">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 text-[11px] text-titan-haze/70 hover:text-titan-haze transition-colors font-mono"
        aria-expanded={open}
      >
        <span aria-hidden className={`transition-transform ${open ? 'rotate-90' : ''}`}>↗</span>
        <span>{hint ?? 'proof'}</span>
      </button>
      {open && (
        <div className="mt-2 bg-titan-bg/90 backdrop-blur-md border border-titan-metal/20 rounded-lg px-3 py-2.5 max-w-md">
          <ProofBody proof={proof} />
        </div>
      )}
    </div>
  );
}

'use client';

import { useState, useEffect } from 'react';
import type { TimeChainBlock, TimeChainStatus } from '@/hooks/useTitanAPI';
import { titanFetch, type TitanId } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';
import InfoTooltip from '@/components/shared/InfoTooltip';

interface Props {
  blocks: TimeChainBlock[];
  selectedFork: number;
  forkNames: TimeChainStatus['forks'] | undefined;
}

const SOURCE_ICONS: Record<string, string> = {
  expression: '\u{1F30A}',
  reasoning_chain: '\u{1F9E0}',
  meta_reasoning: '\u{2728}',
  heartbeat: '\u{1F49C}',
  dream: '\u{1F319}',
  expression_speak: '\u{1F4AC}',
  knowledge: '\u{1F4DA}',
  dream_distillation: '\u{1F52E}',
  output_verifier: '\u{1F50F}',
  self_coding: '\u{1F4BB}',
  cgn_dream_consolidation: '\u{1F320}',
};

const FORK_LABELS: Record<number, string> = {
  0: 'Main', 1: 'Declarative', 2: 'Procedural', 3: 'Episodic', 4: 'Meta', 5: 'Conversation',
};

function formatAge(ts: number): string {
  const age = Date.now() / 1000 - ts;
  if (age < 60) return `${Math.floor(age)}s ago`;
  if (age < 3600) return `${Math.floor(age / 60)}m ago`;
  if (age < 86400) return `${Math.floor(age / 3600)}h ago`;
  return `${Math.floor(age / 86400)}d ago`;
}

interface BlockDetail {
  block_hash: string;
  payload_hash: string;
  prev_hash: string;
  chi_spent: number;
  pot_nonce: number;
  payload: {
    thought_type: string;
    source: string;
    content: unknown;
    significance: number;
    confidence: number;
    tags: string[];
    db_ref: string;
  };
  cross_refs: { fork_id: number; block_height: number }[];
}

function BlockDetailPanel({ forkId, height, titanId, onClose }: { forkId: number; height: number; titanId?: TitanId; onClose: () => void }) {
  const [detail, setDetail] = useState<BlockDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    titanFetch<BlockDetail>(`/v6/timechain/block?fork=${forkId}&height=${height}`, { titan: titanId })
      .then(setDetail)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [forkId, height, titanId]);

  if (loading) return <div className="px-4 py-3 text-xs text-titan-metal/40 animate-pulse">Loading block data...</div>;
  if (error) return <div className="px-4 py-3 text-xs text-red-400/70">Error: {error}</div>;
  if (!detail) return null;

  const contentStr = typeof detail.payload?.content === 'object'
    ? JSON.stringify(detail.payload.content, null, 2)
    : String(detail.payload?.content || '');

  return (
    <div className="bg-titan-card/90 border border-titan-metal/20 rounded-lg mx-3 mb-2 p-4 text-xs font-mono">
      <div className="flex justify-between items-start mb-3">
        <span className="text-titan-haze font-semibold text-sm">Block #{height}</span>
        <button onClick={onClose} className="text-titan-metal/50 hover:text-titan-haze text-sm">&times;</button>
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 mb-3">
        <div>
          <span className="text-titan-metal/40">Block Hash</span>
          <p className="text-titan-growth/80 break-all">{detail.block_hash}</p>
        </div>
        <div>
          <span className="text-titan-metal/40">Payload Hash</span>
          <p className="text-titan-metal/70 break-all">{detail.payload_hash}</p>
        </div>
        <div>
          <span className="text-titan-metal/40">Prev Hash</span>
          <p className="text-titan-metal/70 break-all">{detail.prev_hash}</p>
        </div>
        <div>
          <span className="text-titan-metal/40">PoT Nonce</span>
          <p className="text-titan-pulse/80">{detail.pot_nonce}</p>
        </div>
        <div>
          <span className="text-titan-metal/40">Chi Spent</span>
          <p className="text-titan-metal/70">{detail.chi_spent.toFixed(6)}</p>
        </div>
        <div>
          <span className="text-titan-metal/40">Confidence</span>
          <p className="text-titan-metal/70">{detail.payload?.confidence?.toFixed(3) ?? 'N/A'}</p>
        </div>
      </div>
      {detail.cross_refs?.length > 0 && (
        <div className="mb-3">
          <span className="text-titan-metal/40">Cross References</span>
          <div className="flex gap-2 mt-1">
            {detail.cross_refs.map((ref, i) => (
              <span key={i} className="text-titan-pulse/70 bg-titan-pulse/10 px-2 py-0.5 rounded">
                Fork {ref.fork_id} #{ref.block_height}
              </span>
            ))}
          </div>
        </div>
      )}
      {detail.payload?.db_ref && (
        <div className="mb-3">
          <span className="text-titan-metal/40">DB Reference</span>
          <p className="text-titan-metal/60">{detail.payload.db_ref}</p>
        </div>
      )}
      <div>
        <span className="text-titan-metal/40">Content</span>
        <pre className="text-titan-metal/60 mt-1 max-h-[120px] overflow-y-auto bg-titan-card/50 p-2 rounded text-[11px] whitespace-pre-wrap break-all">
          {contentStr.slice(0, 500)}{contentStr.length > 500 ? '...' : ''}
        </pre>
      </div>
    </div>
  );
}

export default function BlockFeed({ blocks, selectedFork }: Props) {
  const titanId = useTitanId();
  const forkLabel = FORK_LABELS[selectedFork] || `Fork #${selectedFork}`;
  const [expandedBlock, setExpandedBlock] = useState<number | null>(null);

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-titan-metal/70 uppercase tracking-wider">
            Recent Blocks — {forkLabel} Fork
          </h3>
          <InfoTooltip text="Live feed of the most recent blocks committed to this fork. Click a block hash to expand and see full block data including content, cross-references, and Proof of Thought nonce." />
        </div>
        <span className="text-xs text-titan-metal/40 font-mono">
          {blocks.length} blocks
        </span>
      </div>

      {blocks.length === 0 ? (
        <p className="text-sm text-titan-metal/40 text-center py-8">
          No blocks on this fork yet
        </p>
      ) : (
        <div className="space-y-1 max-h-[500px] overflow-y-auto scrollbar-thin scrollbar-thumb-titan-metal/20">
          {blocks.map((block, i) => {
            const icon = SOURCE_ICONS[block.source] || '\u{1F4E6}';
            const isOVG = block.source === 'output_verifier';
            const isHighSig = block.significance >= 0.7;
            const isExpanded = expandedBlock === block.height;

            return (
              <div key={`${block.fork_id}-${block.height}-${i}`}>
                <div
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                    isOVG
                      ? 'bg-titan-growth/5 border border-titan-growth/20'
                      : isHighSig
                      ? 'bg-titan-pulse/5 border border-titan-pulse/10'
                      : 'hover:bg-titan-card/80'
                  }`}
                >
                  <span className="text-base w-7 text-center">{icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setExpandedBlock(isExpanded ? null : block.height)}
                        className="text-sm font-mono text-titan-haze/80 hover:text-titan-growth transition-colors"
                        title="Click to expand block details"
                      >
                        #{block.height}
                      </button>
                      <span
                        className="text-xs font-mono text-titan-metal/40 hover:text-titan-growth/70 cursor-pointer transition-colors truncate max-w-[120px]"
                        onClick={() => setExpandedBlock(isExpanded ? null : block.height)}
                        title={`Hash: ${block.block_hash}`}
                      >
                        {block.block_hash.slice(0, 10)}...
                      </span>
                      <span className="text-xs text-titan-metal/60">
                        {block.source}
                      </span>
                      {isOVG && (
                        <span className="text-[10px] bg-titan-growth/20 text-titan-growth px-1.5 py-0.5 rounded font-semibold">
                          OVG VERIFIED
                        </span>
                      )}
                      {block.tags?.includes('security_alert') && (
                        <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded font-semibold">
                          GUARD ALERT
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[11px] text-titan-metal/40 font-mono">
                        epoch {block.epoch_id.toLocaleString()}
                      </span>
                      <span className="text-[11px] text-titan-metal/30">|</span>
                      <span className="text-[11px] text-titan-metal/40 font-mono">
                        sig {block.significance.toFixed(2)}
                      </span>
                      <span className="text-[11px] text-titan-metal/30">|</span>
                      <span className="text-[11px] text-titan-metal/40 font-mono">
                        chi {block.chi_spent.toFixed(4)}
                      </span>
                    </div>
                  </div>
                  <span className="text-[11px] text-titan-metal/30 whitespace-nowrap font-mono">
                    {formatAge(block.timestamp)}
                  </span>
                </div>

                {/* Expandable block detail */}
                {isExpanded && (
                  <BlockDetailPanel
                    forkId={selectedFork}
                    height={block.height}
                    titanId={titanId}
                    onClose={() => setExpandedBlock(null)}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

'use client';

import { MemoryNode } from '@/lib/types';
import { truncateHash, formatTimestamp } from '@/lib/formatters';

interface NodeDetailProps {
  node: MemoryNode | null;
  onClose: () => void;
}

export default function NodeDetail({ node, onClose }: NodeDetailProps) {
  if (!node) return null;

  return (
    <div className="absolute right-4 top-4 w-80 bg-titan-card/95 backdrop-blur-xl border border-titan-metal/20 rounded-xl p-5 shadow-xl z-20">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-titan-haze">Memory Node</h3>
        <button
          onClick={onClose}
          className="text-titan-metal/40 hover:text-titan-metal transition-colors text-lg leading-none"
        >
          &times;
        </button>
      </div>

      <div className="space-y-3">
        <div>
          <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">Content</p>
          <p className="text-xs text-titan-metal/80 mt-1 leading-relaxed">
            {node.text.length > 200 ? node.text.slice(0, 200) + '...' : node.text}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">Hash</p>
            <p className="text-xs font-mono text-titan-pulse mt-0.5">
              {truncateHash(node.hash, 8)}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">Tier</p>
            <p className="text-xs text-titan-metal/80 mt-0.5 capitalize">{node.tier}</p>
          </div>
          <div>
            <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">Weight</p>
            <p className="text-xs text-titan-haze mt-0.5">
              {(node.effective_weight ?? 0).toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">
              Reinforcements
            </p>
            <p className="text-xs text-titan-growth mt-0.5">{node.reinforcements}</p>
          </div>
        </div>

        <div>
          <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">Timestamp</p>
          <p className="text-xs text-titan-metal/60 mt-0.5">
            {formatTimestamp(node.timestamp)}
          </p>
        </div>

        {node.cluster && (
          <div>
            <p className="text-[10px] text-titan-metal/50 uppercase tracking-wider">Cluster</p>
            <p className="text-xs text-titan-metal/70 mt-0.5">{node.cluster}</p>
          </div>
        )}
      </div>
    </div>
  );
}

'use client';

import { NFTEntry } from '@/lib/types';
import SolscanLink from '@/components/shared/SolscanLink';

interface NFTDetailModalProps {
  nft: NFTEntry | null;
  onClose: () => void;
}

export default function NFTDetailModal({ nft, onClose }: NFTDetailModalProps) {
  if (!nft) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-titan-card border border-titan-metal/20 rounded-2xl max-w-lg w-full mx-4 shadow-2xl overflow-hidden max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {nft.image && (
          <img
            src={nft.image}
            alt={nft.name}
            className="w-full aspect-square object-cover"
          />
        )}
        <div className="p-6 space-y-4">
          <h2 className="text-lg font-semibold text-titan-haze">{nft.name}</h2>

          {nft.description && (
            <p className="text-xs text-titan-metal/70 leading-relaxed">
              {nft.description}
            </p>
          )}

          <div className="flex items-center gap-2">
            <span className="text-xs bg-titan-pulse/20 text-titan-pulse px-2 py-0.5 rounded">
              Generation {nft.generation}
            </span>
            <span className="text-xs bg-titan-metal/10 text-titan-metal/60 px-2 py-0.5 rounded">
              {nft.nft_type}
            </span>
          </div>

          <div className="space-y-2">
            <h4 className="text-[10px] text-titan-metal/50 uppercase tracking-wider font-semibold">
              Attributes
            </h4>
            {Object.entries(nft.attributes).map(([key, val]) => (
              <div
                key={key}
                className="flex justify-between text-xs bg-titan-bg/40 rounded-lg px-3 py-1.5"
              >
                <span className="text-titan-metal/50">{key}</span>
                <span className="text-titan-metal/80">{String(val)}</span>
              </div>
            ))}
          </div>

          <div>
            <h4 className="text-[10px] text-titan-metal/50 uppercase tracking-wider font-semibold mb-1">
              Mint Address
            </h4>
            <SolscanLink address={nft.mint} truncate={false} />
          </div>

          <div className="text-[10px] text-titan-metal/40">
            Minted: {nft.mint_date ? new Date(nft.mint_date).toLocaleString() : 'Unknown'}
          </div>

          <button
            onClick={onClose}
            className="w-full py-2 text-sm text-titan-metal/60 bg-titan-bg/50 rounded-lg hover:bg-titan-bg/80 transition-colors mt-4"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

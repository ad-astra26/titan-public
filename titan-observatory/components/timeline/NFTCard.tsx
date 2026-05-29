'use client';

import { NFTEntry } from '@/lib/types';
import { truncateHash } from '@/lib/formatters';

interface NFTCardProps {
  nft: NFTEntry;
  onClick: () => void;
}

export default function NFTCard({ nft, onClick }: NFTCardProps) {
  // Only show image if it points to a real image file (not a .json metadata URI)
  const hasImage = nft.image && !nft.image.endsWith('.json') && !nft.image.includes('example.com');

  return (
    <button
      onClick={onClick}
      className="flex-shrink-0 w-72 bg-titan-card border border-titan-metal/20 rounded-xl overflow-hidden hover:shadow-pulse-glow transition-shadow text-left"
    >
      {hasImage ? (
        <img
          src={nft.image}
          alt={nft.name}
          className="w-full h-48 object-cover"
          loading="lazy"
        />
      ) : (
        <div className="w-full h-48 bg-gradient-to-br from-titan-pulse/20 to-titan-haze/10 flex items-center justify-center">
          <span className="text-4xl opacity-40">&#x1F48E;</span>
        </div>
      )}
      <div className="p-4 space-y-2">
        <p className="text-sm font-semibold text-titan-haze">{nft.name}</p>
        <div className="flex items-center gap-2">
          <span className="text-[10px] bg-titan-pulse/20 text-titan-pulse px-1.5 py-0.5 rounded">
            Gen {nft.generation}
          </span>
          <span className="text-[10px] bg-titan-metal/10 text-titan-metal/60 px-1.5 py-0.5 rounded">
            {nft.nft_type}
          </span>
        </div>
        <div className="space-y-1">
          {Object.entries(nft.attributes).slice(0, 3).map(([key, val]) => (
            <div key={key} className="flex justify-between text-[10px]">
              <span className="text-titan-metal/50">{key}</span>
              <span className="text-titan-metal/70">{val}</span>
            </div>
          ))}
        </div>
        <p className="text-[10px] text-titan-metal/30">
          Mint: {nft.mint_date ? new Date(nft.mint_date).toLocaleDateString() : 'Unknown'}
        </p>
        <p className="text-[10px] font-mono text-titan-pulse/60 truncate">
          {truncateHash(nft.mint, 8)}
        </p>
      </div>
    </button>
  );
}

'use client';

import { useState } from 'react';
import { NFTEntry } from '@/lib/types';
import NFTCard from './NFTCard';
import NFTDetailModal from './NFTDetailModal';

interface NFTTimelineProps {
  nfts: NFTEntry[];
}

export default function NFTTimeline({ nfts }: NFTTimelineProps) {
  const [selectedNFT, setSelectedNFT] = useState<NFTEntry | null>(null);

  if (nfts.length === 0) {
    return (
      <p className="text-xs text-titan-metal/40 text-center py-12">
        No NFTs minted yet
      </p>
    );
  }

  const sorted = [...nfts].sort(
    (a, b) => (new Date(a.mint_date || 0).getTime() || 0) - (new Date(b.mint_date || 0).getTime() || 0)
  );

  return (
    <>
      <div className="relative">
        {/* Central connecting line */}
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-titan-pulse/30 -translate-y-1/2 shadow-pulse-glow" />

        {/* Scrollable container */}
        <div className="flex gap-6 overflow-x-auto pb-4 pt-4 px-2 snap-x snap-mandatory">
          {sorted.map((nft) => (
            <div key={nft.mint} className="snap-center">
              <NFTCard nft={nft} onClick={() => setSelectedNFT(nft)} />
            </div>
          ))}
        </div>
      </div>

      <NFTDetailModal nft={selectedNFT} onClose={() => setSelectedNFT(null)} />
    </>
  );
}

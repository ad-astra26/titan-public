'use client';

import { ArchiveEntry } from '@/lib/types';
import { formatTimestamp } from '@/lib/formatters';

export default function AudioTile({ entry }: { entry: ArchiveEntry }) {
  const audioUrl = entry.content;
  const title = (entry.metadata?.title as string) || 'Blockchain Sonification';

  return (
    <div className="bg-titan-card/60 border border-titan-pulse/15 rounded-xl p-4 break-inside-avoid mb-4">
      <p className="text-xs text-titan-pulse/80 font-medium mb-2">{title}</p>
      <audio controls className="w-full h-8" preload="none">
        <source src={audioUrl} type="audio/wav" />
      </audio>
      <p className="text-[10px] text-titan-metal/30 mt-2">
        {formatTimestamp(entry.timestamp)}
      </p>
    </div>
  );
}

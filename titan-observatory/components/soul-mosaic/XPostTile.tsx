'use client';

import { ArchiveEntry } from '@/lib/types';
import { formatTimestamp } from '@/lib/formatters';

export default function XPostTile({ entry }: { entry: ArchiveEntry }) {
  const likes = (entry.metadata?.likes as number) ?? 0;
  const replies = (entry.metadata?.replies as number) ?? 0;

  return (
    <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-4 break-inside-avoid mb-4">
      <p className="text-xs text-titan-metal/80 leading-relaxed">{entry.content}</p>
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-center gap-3 text-[10px] text-blue-400/60">
          <span>{likes} likes</span>
          <span>{replies} replies</span>
        </div>
        <span className="text-[10px] text-titan-metal/30">
          {formatTimestamp(entry.timestamp)}
        </span>
      </div>
    </div>
  );
}

'use client';

import { ArchiveEntry } from '@/lib/types';
import { formatTimestamp } from '@/lib/formatters';

export default function LogTile({ entry }: { entry: ArchiveEntry }) {
  const category = (entry.metadata?.category as string) || 'neutral';

  const colorClass =
    category === 'positive'
      ? 'text-titan-growth/70'
      : category === 'blockchain'
        ? 'text-titan-pulse/70'
        : 'text-titan-metal/50';

  return (
    <div className="bg-titan-bg/40 border border-titan-metal/5 rounded-lg p-3 break-inside-avoid mb-4">
      <p className={`text-xs font-mono leading-relaxed ${colorClass}`}>
        {entry.content}
      </p>
      <p className="text-[10px] text-titan-metal/25 mt-1 font-mono">
        {formatTimestamp(entry.timestamp)}
      </p>
    </div>
  );
}

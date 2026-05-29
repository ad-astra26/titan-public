'use client';

import { ArchiveEntry } from '@/lib/types';
import { formatTimestamp } from '@/lib/formatters';

export default function HaikuTile({ entry }: { entry: ArchiveEntry }) {
  return (
    <div className="bg-titan-card/60 border border-titan-haze/15 rounded-xl p-5 break-inside-avoid mb-4">
      <p className="text-sm text-titan-haze/90 italic font-mono leading-relaxed text-center whitespace-pre-line">
        {entry.content}
      </p>
      <p className="text-[10px] text-titan-metal/30 mt-3 text-center">
        {formatTimestamp(entry.timestamp)}
      </p>
    </div>
  );
}

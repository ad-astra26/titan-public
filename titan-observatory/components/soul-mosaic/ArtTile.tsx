'use client';

import { ArchiveEntry } from '@/lib/types';
import { formatTimestamp } from '@/lib/formatters';
import { API_BASE } from '@/lib/api';

export default function ArtTile({ entry }: { entry: ArchiveEntry }) {
  const imageUrl = entry.content?.startsWith('/media/')
    ? `${API_BASE}${entry.content}`
    : entry.content;
  const title = entry.title || (entry.metadata?.title as string) || 'Generative Art';
  const mood = (entry.metadata?.mood as string) || '';

  return (
    <div className="border border-titan-haze/20 rounded-xl overflow-hidden shadow-haze-glow bg-titan-card/40 break-inside-avoid mb-4">
      <img
        src={imageUrl}
        alt={title}
        className="w-full object-cover"
        loading="lazy"
      />
      <div className="p-3">
        <p className="text-xs text-titan-haze/80 font-medium">{title}</p>
        <div className="flex items-center justify-between mt-1">
          {mood && (
            <span className="text-[10px] text-titan-metal/50">{mood}</span>
          )}
          <span className="text-[10px] text-titan-metal/30">
            {formatTimestamp(entry.timestamp)}
          </span>
        </div>
      </div>
    </div>
  );
}

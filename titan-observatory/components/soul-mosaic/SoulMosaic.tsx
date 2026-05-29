'use client';

import { ArchiveEntry } from '@/lib/types';
import ArtTile from './ArtTile';
import HaikuTile from './HaikuTile';
import XPostTile from './XPostTile';
import LogTile from './LogTile';
import AudioTile from './AudioTile';

interface SoulMosaicProps {
  entries: ArchiveEntry[];
}

function renderTile(entry: ArchiveEntry, index: number) {
  const key = `${entry.type}-${entry.timestamp}-${index}`;
  switch (entry.type) {
    case 'art':
      return <ArtTile key={key} entry={entry} />;
    case 'haiku':
      return <HaikuTile key={key} entry={entry} />;
    case 'x_post':
      return <XPostTile key={key} entry={entry} />;
    case 'log':
      return <LogTile key={key} entry={entry} />;
    case 'audio':
      return <AudioTile key={key} entry={entry} />;
    default:
      return <LogTile key={key} entry={entry} />;
  }
}

export default function SoulMosaic({ entries }: SoulMosaicProps) {
  if (entries.length === 0) {
    return (
      <p className="text-xs text-titan-metal/40 text-center py-12">
        No soul expressions yet
      </p>
    );
  }

  return (
    <div className="columns-1 md:columns-2 lg:columns-3 gap-4">
      {entries.map((entry, i) => renderTile(entry, i))}
    </div>
  );
}

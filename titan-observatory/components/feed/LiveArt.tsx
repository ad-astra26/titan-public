'use client';

import { useLiveArt } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatTimestamp } from '@/lib/formatters';

export default function LiveArt() {
  const titanId = useTitanId();
  const { data: art } = useLiveArt(titanId);

  if (!art?.url) return null;

  return (
    <div className="border border-titan-haze/20 rounded-lg overflow-hidden animate-breathe shadow-haze-glow">
      <img
        src={art.url}
        alt="Live generative art"
        className="w-full aspect-video object-cover"
        loading="lazy"
      />
      <div className="px-3 py-2 bg-titan-bg/60">
        <p className="text-[10px] text-titan-haze/70">
          Mood: {art.mood}
        </p>
        <p className="text-[10px] text-titan-metal/30">
          {formatTimestamp(art.timestamp)}
        </p>
      </div>
    </div>
  );
}

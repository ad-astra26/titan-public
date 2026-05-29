'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import Image from 'next/image';
import { titanFetch, v4Fetch, API_BASE } from '@/lib/api';

interface CreativeWork {
  id: number;
  timestamp: number;
  work_type: string;
  file_path: string;
  media_url: string;
  style?: string;
}

// Assign pseudo-random proportions based on art style for masonry variety
function getAspect(style: string, idx: number): string {
  const patterns: Record<string, string[]> = {
    fractal: ['aspect-[3/4]', 'aspect-square', 'aspect-[4/5]'],
    cellular: ['aspect-square', 'aspect-[4/3]', 'aspect-square'],
    flow: ['aspect-[4/5]', 'aspect-square', 'aspect-[3/4]'],
    tree: ['aspect-[3/5]', 'aspect-[2/3]', 'aspect-[3/4]'],
    geometric: ['aspect-square', 'aspect-[4/5]', 'aspect-square'],
    landscape: ['aspect-[5/3]', 'aspect-[4/3]', 'aspect-[3/2]'],
  };
  const variants = patterns[style] || ['aspect-square', 'aspect-[4/5]', 'aspect-[3/4]'];
  return variants[idx % variants.length];
}

function NarrationOverlay({ filePath }: { filePath: string }) {
  const { data } = useQuery({
    queryKey: ['narrate-art', filePath],
    queryFn: () => titanFetch<{ narration: string }>(`/v6/expression/narrate-art?file_path=${encodeURIComponent(filePath)}`),
    staleTime: Infinity, // Cached permanently
    retry: 1,
    enabled: !!filePath,
  });

  const text = data?.narration || '';
  if (!text) return null;

  return (
    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex items-end p-2">
      <p className="text-[10px] text-white/90 leading-snug italic">
        {text}
      </p>
    </div>
  );
}

export default function MasonryGallery() {
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [selectedImg, setSelectedImg] = useState<{ src: string; alt: string } | null>(null);

  const { data } = useQuery({
    queryKey: ['feed-gallery'],
    queryFn: () => v4Fetch<{ items: CreativeWork[] }>('creative-works', { extraQuery: 'work_type=art&limit=20' }),
    refetchInterval: 60000,
    retry: 1,
  });

  const items = data?.items ?? [];
  if (items.length === 0) return null;

  // Split into 2 columns for masonry
  const col1 = items.filter((_, i) => i % 2 === 0);
  const col2 = items.filter((_, i) => i % 2 === 1);

  return (
    <div className="bg-titan-card/40 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4">
      <h3 className="text-[10px] font-medium text-titan-metal/50 uppercase tracking-wider mb-3">
        Autonomous Art
      </h3>

      <div className="flex gap-2">
        {[col1, col2].map((col, ci) => (
          <div key={ci} className="flex-1 flex flex-col gap-2">
            {col.map((item, idx) => {
              const imgUrl = `${API_BASE}${item.media_url}`;
              const style = item.style || 'art';
              const aspect = getAspect(style, idx + ci);
              const isHovered = hoveredId === item.id;

              return (
                <div
                  key={item.id}
                  className={`relative rounded-lg overflow-hidden cursor-pointer transition-all ${aspect} bg-titan-bg hover:shadow-lg hover:shadow-titan-haze/10`}
                  onMouseEnter={() => setHoveredId(item.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  onClick={() => setSelectedImg({ src: imgUrl, alt: `${style} art` })}
                >
                  {/* Next.js Image optimizer auto-generates WebP/AVIF + caches
                      in .next/cache/images/. `fill` mode + sizes lets browser
                      download just the right resolution. rFP §1.4.A */}
                  <Image
                    src={imgUrl}
                    alt={`${style} art`}
                    fill
                    sizes="(max-width: 768px) 50vw, 33vw"
                    className="object-cover"
                    loading="lazy"
                    unoptimized={false}
                  />
                  {isHovered && <NarrationOverlay filePath={item.file_path} />}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Full-size modal */}
      {selectedImg && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 cursor-pointer"
          onClick={() => setSelectedImg(null)}
        >
          <div className="relative max-w-3xl max-h-[80vh]" onClick={e => e.stopPropagation()}>
            <img src={selectedImg.src} alt={selectedImg.alt} className="max-w-full max-h-[80vh] rounded-lg object-contain" />
            <button
              onClick={() => setSelectedImg(null)}
              className="absolute top-2 right-2 w-8 h-8 rounded-full bg-titan-bg/80 text-titan-metal hover:text-titan-haze flex items-center justify-center"
            >
              x
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

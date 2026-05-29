'use client';

import { useQuery } from '@tanstack/react-query';
import { useState, useRef } from 'react';
import Image from 'next/image';
import { titanFetch } from '@/lib/api';
import { API_BASE } from '@/lib/api';

interface CreativeWork {
  id: number;
  timestamp: number;
  work_type: string;
  file_path: string;
  media_url: string;
  triggering_program: string;
  posture: string;
  assessment_score: number;
  hormone_level_at_creation: number;
  style?: string;
}

function timeAgo(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function ImageModal({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 cursor-pointer"
      onClick={onClose}
    >
      <div className="relative max-w-3xl max-h-[80vh]" onClick={e => e.stopPropagation()}>
        <img src={src} alt={alt} className="max-w-full max-h-[80vh] rounded-lg object-contain" />
        <button
          onClick={onClose}
          className="absolute top-2 right-2 w-8 h-8 rounded-full bg-titan-bg/80 text-titan-metal hover:text-titan-haze flex items-center justify-center"
        >
          x
        </button>
        <p className="text-xs text-titan-metal/50 text-center mt-2">{alt}</p>
      </div>
    </div>
  );
}

function AudioPlayer({ src, label }: { src: string; label: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  const toggle = () => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setPlaying(!playing);
  };

  return (
    <div
      className="flex items-center gap-2 bg-titan-bg/60 border border-titan-metal/10 rounded-lg px-3 py-2 cursor-pointer hover:border-titan-pulse/30 transition-all"
      onClick={toggle}
    >
      <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${
        playing ? 'bg-titan-pulse/20 text-titan-pulse' : 'bg-titan-metal/10 text-titan-metal/50'
      }`}>
        {playing ? (
          <svg width="10" height="10" viewBox="0 0 12 12" fill="currentColor">
            <rect x="2" y="1" width="3" height="10" rx="1" />
            <rect x="7" y="1" width="3" height="10" rx="1" />
          </svg>
        ) : (
          <svg width="10" height="10" viewBox="0 0 12 12" fill="currentColor">
            <polygon points="2,1 11,6 2,11" />
          </svg>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] text-titan-metal/60 truncate">{label}</p>
        {playing && (
          <div className="flex gap-0.5 mt-0.5">
            {[...Array(12)].map((_, i) => (
              <div
                key={i}
                className="w-1 bg-titan-pulse/60 rounded-full animate-pulse"
                style={{
                  height: `${4 + Math.random() * 8}px`,
                  animationDelay: `${i * 0.1}s`,
                }}
              />
            ))}
          </div>
        )}
      </div>
      <audio
        ref={audioRef}
        src={src}
        onEnded={() => setPlaying(false)}
        preload="none"
      />
    </div>
  );
}

const COLLAPSED_ART = 8;  // Show one row of small thumbnails collapsed
const COLLAPSED_AUDIO = 2;

export default function CreativeGallery() {
  const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);
  const [tab, setTab] = useState<'art' | 'audio'>('art');
  const [expanded, setExpanded] = useState(false);

  const { data: artData } = useQuery({
    queryKey: ['creative-works-art'],
    queryFn: () => titanFetch<{ items: CreativeWork[]; count: number }>('/v6/expression/creative-works?work_type=art&limit=24'),
    refetchInterval: 30000,
    retry: 1,
  });

  const { data: audioData } = useQuery({
    queryKey: ['creative-works-audio'],
    queryFn: () => titanFetch<{ items: CreativeWork[]; count: number }>('/v6/expression/creative-works?work_type=audio&limit=12'),
    refetchInterval: 30000,
    retry: 1,
  });

  const artItems = artData?.items ?? [];
  const audioItems = audioData?.items ?? [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const artCount = (artData as any)?.art_total ?? artItems.length;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const audioCount = (audioData as any)?.audio_total ?? audioItems.length;

  if (artItems.length === 0 && audioItems.length === 0) return null;

  const visibleArt = expanded ? artItems.slice(0, 24) : artItems.slice(0, COLLAPSED_ART);
  const visibleAudio = expanded ? audioItems.slice(0, 8) : audioItems.slice(0, COLLAPSED_AUDIO);
  const hasMore = tab === 'art'
    ? artItems.length > COLLAPSED_ART
    : audioItems.length > COLLAPSED_AUDIO;

  return (
    <div className="bg-titan-card/40 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4">
      {/* Header with tabs + expand toggle */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex gap-2">
          <button
            onClick={() => { setTab('art'); setExpanded(false); }}
            className={`text-[10px] font-medium uppercase tracking-wider px-2 py-1 rounded transition-all ${
              tab === 'art'
                ? 'text-titan-haze bg-titan-haze/10'
                : 'text-titan-metal/40 hover:text-titan-metal/60'
            }`}
          >
            Art ({artCount.toLocaleString()})
          </button>
          {audioCount > 0 && (
            <button
              onClick={() => { setTab('audio'); setExpanded(false); }}
              className={`text-[10px] font-medium uppercase tracking-wider px-2 py-1 rounded transition-all ${
                tab === 'audio'
                  ? 'text-titan-pulse bg-titan-pulse/10'
                  : 'text-titan-metal/40 hover:text-titan-metal/60'
              }`}
            >
              Audio ({audioCount.toLocaleString()})
            </button>
          )}
        </div>
        {hasMore && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[10px] text-titan-metal/40 hover:text-titan-haze transition-colors"
          >
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>

      {/* Art Grid — single row collapsed, multi-row expanded */}
      {tab === 'art' && (
        expanded ? (
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-1.5">
            {visibleArt.map((item, idx) => {
              const imgUrl = `${API_BASE}${item.media_url}`;
              const style = item.style || '';
              return (
                <div key={item.id || idx} className="group relative">
                  <div
                    className="relative aspect-square rounded overflow-hidden cursor-pointer hover:ring-1 hover:ring-titan-haze/40 transition-all bg-titan-bg"
                    onClick={() => setSelectedImage({ src: imgUrl, alt: `${style || 'Autonomous'} art` })}
                  >
                    {/* Next.js Image optimizer per rFP §1.4.A — WebP/AVIF, sized for grid cell */}
                    <Image src={imgUrl} alt={`art ${idx + 1}`} fill
                      sizes="(max-width: 640px) 25vw, (max-width: 768px) 17vw, (max-width: 1024px) 12vw, 10vw"
                      className="object-cover" loading="lazy" />
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent rounded-b px-1 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <p className="text-[7px] text-white/80 truncate">{style || 'art'} · {timeAgo(item.timestamp)}</p>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex gap-1.5 overflow-x-auto scrollbar-none">
            {visibleArt.map((item, idx) => {
              const imgUrl = `${API_BASE}${item.media_url}`;
              const style = item.style || '';
              return (
                <div
                  key={item.id || idx}
                  className="relative w-16 h-16 shrink-0 rounded overflow-hidden cursor-pointer hover:ring-1 hover:ring-titan-haze/40 transition-all bg-titan-bg"
                  onClick={() => setSelectedImage({ src: imgUrl, alt: `${style || 'Autonomous'} art` })}
                >
                  {/* Next.js Image optimizer — 64x64 thumb strip variant */}
                  <Image src={imgUrl} alt={`art ${idx + 1}`} fill sizes="64px"
                    className="object-cover" loading="lazy" />
                </div>
              );
            })}
          </div>
        )
      )}

      {/* Audio List — compact */}
      {tab === 'audio' && (
        <div className="space-y-1.5">
          {visibleAudio.map((item, idx) => {
            const audioUrl = `${API_BASE}${item.media_url}`;
            const program = item.triggering_program || 'autonomous';
            return (
              <AudioPlayer
                key={item.id || idx}
                src={audioUrl}
                label={`${program} · ${timeAgo(item.timestamp)}`}
              />
            );
          })}
          {audioItems.length === 0 && (
            <p className="text-xs text-titan-metal/40 italic text-center py-2">
              No audio works yet
            </p>
          )}
        </div>
      )}

      <p className="text-[8px] text-titan-metal/20 mt-1.5 text-center">
        {tab === 'art'
          ? 'Click to view full size. All art generated autonomously.'
          : 'Click to play. Trinity sonifications from inner state.'}
      </p>

      {selectedImage && (
        <ImageModal
          src={selectedImage.src}
          alt={selectedImage.alt}
          onClose={() => setSelectedImage(null)}
        />
      )}
    </div>
  );
}

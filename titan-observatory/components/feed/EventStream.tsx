'use client';

import { useState, useCallback, type MouseEvent as ReactMouseEvent } from 'react';
import { createPortal } from 'react-dom';
import { useTitanStore } from '@/store/titanStore';
import { useEvents } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { formatTimestamp } from '@/lib/formatters';
import { WSEvent } from '@/lib/types';

const EVENT_COLORS: Record<string, string> = {
  chat_message: 'border-titan-growth/30',
  memory_reinforcement: 'border-titan-haze/30',
  memory_commit: 'border-titan-pulse/30',
  mood_update: 'border-titan-growth/30',
  social_post: 'border-titan-growth/30',
  epoch_transition: 'border-titan-growth/30',
  guardian_block: 'border-red-500/30',
  directive_update: 'border-titan-haze/30',
  memory_injection: 'border-titan-haze/30',
  divine_inspiration: 'border-titan-haze/30',
  resurrection: 'border-titan-pulse/30',
  cluster_verified: 'border-titan-pulse/30',
};

/* ── Inline SVG Icons (brand-colored, no external deps) ────────────── */

function IconChat() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M2 3a1 1 0 011-1h10a1 1 0 011 1v7a1 1 0 01-1 1H5l-3 3V3z" fill="#77CCCC" opacity="0.8"/>
    </svg>
  );
}
function IconBrain() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="6" fill="#E5C79E" opacity="0.7"/>
      <path d="M5.5 8c0-1.4 1.1-2.5 2.5-2.5s2.5 1.1 2.5 2.5" stroke="#0B0E14" strokeWidth="1.2" fill="none"/>
      <path d="M6 10.5c.5.8 1.2 1 2 1s1.5-.2 2-1" stroke="#0B0E14" strokeWidth="1" fill="none"/>
    </svg>
  );
}
function IconChain() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M6 4h4a3 3 0 010 6H6a3 3 0 010-6z" stroke="#9945FF" strokeWidth="1.5" fill="none" opacity="0.8"/>
      <circle cx="5" cy="7" r="1.5" fill="#9945FF" opacity="0.6"/>
      <circle cx="11" cy="7" r="1.5" fill="#9945FF" opacity="0.6"/>
    </svg>
  );
}
function IconMask() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <ellipse cx="8" cy="8" rx="6" ry="5" fill="#9945FF" opacity="0.6"/>
      <circle cx="6" cy="7" r="1.2" fill="#0B0E14"/>
      <circle cx="10" cy="7" r="1.2" fill="#0B0E14"/>
      <path d="M6.5 10c.4.5 1 .8 1.5.8s1.1-.3 1.5-.8" stroke="#0B0E14" strokeWidth="0.8" fill="none"/>
    </svg>
  );
}
function IconBird() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M2 6l3-2 3 1 4-3 2 2-3 3 1 3-2 1-3-2-4 1-1-4z" fill="#77CCCC" opacity="0.8"/>
    </svg>
  );
}
function IconClock() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="6" stroke="#77CCCC" strokeWidth="1.3" fill="none" opacity="0.7"/>
      <path d="M8 4v4l3 2" stroke="#77CCCC" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
    </svg>
  );
}
function IconShield() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M8 2L3 4v4c0 3.3 2.2 5.3 5 6 2.8-.7 5-2.7 5-6V4L8 2z" fill="#ff6b6b" opacity="0.7"/>
    </svg>
  );
}
function IconScroll() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <rect x="4" y="2" width="8" height="12" rx="1" stroke="#E5C79E" strokeWidth="1.2" fill="none" opacity="0.7"/>
      <path d="M6 5h4M6 7.5h4M6 10h2" stroke="#E5C79E" strokeWidth="0.8" opacity="0.5"/>
    </svg>
  );
}
function IconSyringe() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <rect x="6" y="2" width="4" height="10" rx="1" fill="#E5C79E" opacity="0.6"/>
      <path d="M7 12l1 2 1-2" fill="#E5C79E" opacity="0.8"/>
    </svg>
  );
}
function IconSparkle() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M8 1l1.5 4.5L14 7l-4.5 1.5L8 13l-1.5-4.5L2 7l4.5-1.5z" fill="#E5C79E" opacity="0.8"/>
    </svg>
  );
}
function IconFire() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M8 1c0 3-3 4-3 7a3.5 3.5 0 007 0c0-3-3-4-3-7z" fill="#ff6b6b" opacity="0.7"/>
      <path d="M8 6c0 1.5-1.5 2-1.5 3.5a1.8 1.8 0 003.5 0c0-1.5-1.5-2-1.5-3.5z" fill="#E5C79E" opacity="0.8"/>
    </svg>
  );
}
function IconCheck() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="6" fill="#77CCCC" opacity="0.6"/>
      <path d="M5 8l2 2 4-4" stroke="#0B0E14" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    </svg>
  );
}
function IconDefault() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="5" stroke="#8E9AAF" strokeWidth="1.2" fill="none" opacity="0.5"/>
      <circle cx="8" cy="8" r="2" fill="#8E9AAF" opacity="0.4"/>
    </svg>
  );
}

const EVENT_ICON_COMPONENTS: Record<string, () => JSX.Element> = {
  chat_message: IconChat,
  memory_reinforcement: IconBrain,
  memory_commit: IconChain,
  mood_update: IconMask,
  social_post: IconBird,
  epoch_transition: IconClock,
  guardian_block: IconShield,
  directive_update: IconScroll,
  memory_injection: IconSyringe,
  divine_inspiration: IconSparkle,
  resurrection: IconFire,
  cluster_verified: IconCheck,
};

function EventIcon({ type }: { type: string }) {
  const Comp = EVENT_ICON_COMPONENTS[type] || IconDefault;
  return <span className="inline-flex flex-shrink-0 w-[14px] h-[14px]"><Comp /></span>;
}

/* ── Helpers ───────────────────────────────────────────────────────── */

function eventSummary(event: WSEvent): string {
  const d = event.data;
  switch (event.type) {
    case 'chat_message':
      return `[${d.mode || 'Shadow'}] ${(d.user_prompt as string)?.slice(0, 60) || 'conversation'}`;
    case 'mood_update':
      return `Mood: ${d.label || 'updated'} (${d.score || ''})`;
    case 'social_post':
      return `Posted: ${(d.text as string)?.slice(0, 60) || 'new post'}`;
    case 'memory_reinforcement':
      return `Memory reinforced: ${(d.hash as string)?.slice(0, 12) || ''}`;
    case 'memory_commit':
      return `On-chain commit: ${d.count || ''} memories`;
    case 'guardian_block':
      return `Guardian blocked: ${d.tier || ''} - ${d.category || ''}`;
    case 'epoch_transition':
      return `Epoch: ${d.epoch_type || 'transition'}`;
    default:
      return event.type.replace(/_/g, ' ');
  }
}

function fullEventDetail(event: WSEvent): string {
  const d = event.data;
  switch (event.type) {
    case 'chat_message': {
      const prompt = (d.user_prompt as string) || '';
      const response = (d.response as string) || '';
      const mode = d.mode || 'Shadow';
      return `[${mode}]\n\nUser: ${prompt}\n\nTitan: ${response}`;
    }
    case 'social_post':
      return (d.text as string) || 'Post content unavailable';
    case 'mood_update':
      return `Mood shifted to "${d.label || 'unknown'}" (score: ${d.score || 'N/A'}, delta: ${d.delta || 0})`;
    case 'guardian_block':
      return `Tier: ${d.tier || 'unknown'}\nCategory: ${d.category || 'unknown'}\nAction: ${d.action || 'blocked'}`;
    default:
      return JSON.stringify(d, null, 2);
  }
}

/* ── Constants ─────────────────────────────────────────────────────── */
const CARD_WIDTH = 380;
const CARD_MAX_HEIGHT = 420;
const EDGE_PADDING = 12;

export default function EventStream() {
  const titanId = useTitanId();
  const liveEvents = useTitanStore((s) => s.events);
  const { data: persistedEvents } = useEvents(titanId);
  const [paused, setPaused] = useState(false);
  const [expandedEvent, setExpandedEvent] = useState<WSEvent | null>(null);
  const [clickPos, setClickPos] = useState({ x: 0, y: 0 });

  // Merge live WS events with persisted DB events, deduplicate by timestamp
  const liveTimestamps = new Set(liveEvents.map((e) => e.timestamp));
  const historical = (persistedEvents || []).filter((e) => !liveTimestamps.has(e.timestamp));
  const allEvents = [...liveEvents, ...historical]
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 100);

  const displayEvents = paused ? allEvents : allEvents.slice(0, 50);

  const handleEventClick = useCallback(
    (e: ReactMouseEvent, event: WSEvent) => {
      // Clamp card position so it doesn't overflow viewport
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let x = e.clientX;
      let y = e.clientY;
      if (x + CARD_WIDTH + EDGE_PADDING > vw) x = vw - CARD_WIDTH - EDGE_PADDING;
      if (x < EDGE_PADDING) x = EDGE_PADDING;
      if (y + CARD_MAX_HEIGHT + EDGE_PADDING > vh) y = vh - CARD_MAX_HEIGHT - EDGE_PADDING;
      if (y < EDGE_PADDING) y = EDGE_PADDING;
      setClickPos({ x, y });
      setExpandedEvent(event);
    },
    [],
  );

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-4 h-full flex flex-col relative">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
          Event Stream
        </h3>
        <button
          onClick={() => setPaused(!paused)}
          className={`text-[10px] px-2 py-0.5 rounded ${
            paused
              ? 'bg-yellow-500/20 text-yellow-400'
              : 'bg-titan-growth/20 text-titan-growth'
          }`}
        >
          {paused ? 'Paused' : 'Live'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-0 min-h-0">
        {displayEvents.length === 0 ? (
          <p className="text-xs text-titan-metal/40 text-center py-8">
            Waiting for events...
          </p>
        ) : (
          displayEvents.map((event, i) => (
            <button
              key={`${event.timestamp}-${i}`}
              onClick={(e) => handleEventClick(e, event)}
              className={`w-full text-left border-l-2 pl-3 py-2 cursor-pointer
                transition-all duration-200 hover:translate-x-1 hover:bg-titan-haze/5
                ${EVENT_COLORS[event.type] || 'border-titan-metal/20'}
                ${i % 2 === 0 ? 'bg-titan-bg/20' : 'bg-transparent'}
              `}
            >
              <div className="flex items-center gap-1.5">
                <EventIcon type={event.type} />
                <span className="text-[10px] font-semibold text-titan-metal/50 uppercase">
                  {event.type.replace(/_/g, ' ')}
                </span>
              </div>
              <p className="text-xs text-titan-metal/70 mt-0.5 truncate">
                {eventSummary(event)}
              </p>
              <p className="text-[10px] text-titan-metal/30 mt-0.5">
                {formatTimestamp(new Date(event.timestamp * 1000).toISOString())}
              </p>
            </button>
          ))
        )}
      </div>

      {/* Detail card — portaled to body, positioned at click coordinates */}
      {expandedEvent && createPortal(
        <div
          className="fixed inset-0 z-50 event-overlay-backdrop"
          style={{ backgroundColor: 'rgba(0,0,0,0.4)' }}
          onClick={() => setExpandedEvent(null)}
        >
          <div
            className="absolute border rounded-xl shadow-2xl overflow-hidden event-overlay-card"
            style={{
              left: clickPos.x,
              top: clickPos.y,
              width: CARD_WIDTH,
              maxHeight: CARD_MAX_HEIGHT,
              backgroundColor: 'var(--titan-card)',
              borderColor: 'rgba(229, 199, 158, 0.15)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.4), 0 0 12px rgba(153, 69, 255, 0.1)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-4 py-3" style={{ borderBottom: '1px solid rgba(142, 154, 175, 0.1)' }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <EventIcon type={expandedEvent.type} />
                  <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: 'var(--titan-haze)' }}>
                    {expandedEvent.type.replace(/_/g, ' ')}
                  </span>
                </div>
                <button
                  onClick={() => setExpandedEvent(null)}
                  className="text-sm leading-none px-1 rounded hover:bg-white/5"
                  style={{ color: 'rgba(142, 154, 175, 0.5)' }}
                >
                  &#x2715;
                </button>
              </div>
              <p className="text-[10px] mt-1" style={{ color: 'rgba(142, 154, 175, 0.4)' }}>
                {new Date(expandedEvent.timestamp * 1000).toLocaleString()}
              </p>
            </div>
            <div className="px-4 py-3 overflow-y-auto" style={{ maxHeight: CARD_MAX_HEIGHT - 70 }}>
              <pre className="text-xs whitespace-pre-wrap font-sans leading-relaxed" style={{ color: 'rgba(142, 154, 175, 0.8)' }}>
                {fullEventDetail(expandedEvent)}
              </pre>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}

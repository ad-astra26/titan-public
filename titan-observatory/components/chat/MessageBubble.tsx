'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '@/lib/chat';
import ModeBadge from './ModeBadge';
import MoodIndicator from './MoodIndicator';

interface MessageBubbleProps {
  message: ChatMessage;
}

/** Map mood labels to subtle gradient backgrounds for Titan messages */
const MOOD_STYLES: Record<string, { bg: string; border: string; glow: string }> = {
  Curious:        { bg: 'bg-gradient-to-br from-titan-card to-amber-950/20',    border: 'border-amber-400/15',   glow: 'shadow-amber-500/5' },
  Contemplative:  { bg: 'bg-gradient-to-br from-titan-card to-blue-950/20',     border: 'border-blue-400/15',    glow: 'shadow-blue-500/5' },
  Energized:      { bg: 'bg-gradient-to-br from-titan-card to-emerald-950/20',  border: 'border-emerald-400/15', glow: 'shadow-emerald-500/5' },
  Creative:       { bg: 'bg-gradient-to-br from-titan-card to-pink-950/20',     border: 'border-pink-400/15',    glow: 'shadow-pink-500/5' },
  Cautious:       { bg: 'bg-gradient-to-br from-titan-card to-yellow-950/20',   border: 'border-yellow-400/15',  glow: 'shadow-yellow-500/5' },
  Restless:       { bg: 'bg-gradient-to-br from-titan-card to-orange-950/15',   border: 'border-orange-400/15',  glow: 'shadow-orange-500/5' },
  Serene:         { bg: 'bg-gradient-to-br from-titan-card to-teal-950/20',     border: 'border-teal-400/15',    glow: 'shadow-teal-500/5' },
  Focused:        { bg: 'bg-gradient-to-br from-titan-card to-cyan-950/15',     border: 'border-cyan-400/15',    glow: 'shadow-cyan-500/5' },
  Melancholic:    { bg: 'bg-gradient-to-br from-titan-card to-indigo-950/20',   border: 'border-indigo-400/15',  glow: 'shadow-indigo-500/5' },
  Stable:         { bg: 'bg-gradient-to-br from-titan-card to-slate-800/20',    border: 'border-titan-metal/10', glow: '' },
};

const DEFAULT_STYLE = { bg: 'bg-titan-card', border: 'border-titan-metal/10', glow: '' };

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isBlocked = message.blocked;

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] md:max-w-[60%]">
          <div className="bg-titan-pulse/20 border border-titan-pulse/30 rounded-2xl rounded-br-sm px-4 py-2.5">
            <p className="text-sm text-titan-metal whitespace-pre-wrap break-words">
              {message.content}
            </p>
          </div>
          <div className="flex justify-end mt-1 px-1">
            <span className="text-[10px] text-titan-metal/40">{formatTime(message.timestamp)}</span>
          </div>
        </div>
      </div>
    );
  }

  // Titan message — mood-aware styling
  const moodStyle = message.mood ? (MOOD_STYLES[message.mood] ?? DEFAULT_STYLE) : DEFAULT_STYLE;
  const blockedStyle = isBlocked
    ? { bg: 'bg-red-900/20', border: 'border-2 border-red-500/40', glow: '' }
    : moodStyle;

  return (
    <div className="flex justify-start">
      <div className="max-w-[75%] md:max-w-[70%]">
        <div
          className={`rounded-2xl rounded-bl-sm px-4 py-2.5 border transition-all duration-500 ${blockedStyle.bg} ${blockedStyle.border} ${blockedStyle.glow ? `shadow-md ${blockedStyle.glow}` : ''}`}
        >
          {isBlocked && (
            <div className="flex items-center gap-1.5 mb-2">
              <svg className="w-3.5 h-3.5 text-red-400 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                <path
                  fillRule="evenodd"
                  d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z"
                  clipRule="evenodd"
                />
              </svg>
              <span className="text-[11px] font-semibold text-red-400 uppercase tracking-wide">
                Guardian Shield
              </span>
            </div>
          )}
          <div className="text-sm text-titan-metal prose prose-invert prose-sm max-w-none prose-p:my-1 prose-headings:text-titan-haze prose-code:text-titan-pulse prose-code:bg-titan-bg/50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-titan-bg/80 prose-pre:border prose-pre:border-titan-metal/10 prose-a:text-titan-growth prose-strong:text-titan-haze/90 break-words">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-1 px-1 flex-wrap">
          {message.mode && <ModeBadge mode={message.mode} />}
          {message.mood && <MoodIndicator mood={message.mood} />}
          <span className="text-[10px] text-titan-metal/40">{formatTime(message.timestamp)}</span>
        </div>
      </div>
    </div>
  );
}

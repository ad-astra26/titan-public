'use client';

import { useEffect, useState } from 'react';

/**
 * Visible progress indicator while Titan composes a reply.
 *
 * Two surfaces:
 *   • `<ThinkingIndicator />`        — bubble form for the wallet `/chat` route
 *   • `<ThinkingInline variant=…/>`  — inline form for the `/v/<token>/pitch`
 *                                      Compare mode (rendered inside each Titan column)
 *
 * Both share the bouncing-dots animation + an elapsed-seconds counter that
 * starts at 0.0s and ticks once per 100ms. Users see something is happening
 * instead of staring at a static "thinking…" — important for the heavy tier
 * which can take 30-60s on Ollama Cloud's deepseek-v3.1:671b.
 */

function useElapsedSeconds(): number {
  const [start] = useState(() => Date.now());
  const [now, setNow] = useState(start);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(id);
  }, []);
  return (now - start) / 1000;
}

function ElapsedBadge({ seconds }: { seconds: number }) {
  // Tail off display once we cross 60s — keeps the badge compact and avoids
  // anxious "63.7s" / "127.2s" displays. After 60s we just show "60s+".
  const label = seconds < 60 ? `${seconds.toFixed(1)}s` : '60s+';
  return (
    <span className="text-[10px] font-mono text-titan-metal/40 ml-1.5 tabular-nums">
      {label}
    </span>
  );
}

function BouncingDots({ size = 'sm' }: { size?: 'sm' | 'xs' }) {
  const dotClass = size === 'xs'
    ? 'w-[3px] h-[3px] rounded-full bg-titan-haze/60 animate-bounce'
    : 'w-1 h-1 rounded-full bg-titan-haze/60 animate-bounce';
  return (
    <span className="flex gap-0.5 items-center">
      <span className={dotClass} style={{ animationDelay: '0ms' }} />
      <span className={dotClass} style={{ animationDelay: '150ms' }} />
      <span className={dotClass} style={{ animationDelay: '300ms' }} />
    </span>
  );
}

// ── Bubble form (wallet /chat) ───────────────────────────────────────

export default function ThinkingIndicator() {
  const elapsed = useElapsedSeconds();
  return (
    <div className="flex justify-start">
      <div className="bg-titan-card border border-titan-metal/10 rounded-2xl rounded-bl-sm px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-titan-metal/60">Titan is thinking</span>
          <BouncingDots />
          <ElapsedBadge seconds={elapsed} />
        </div>
      </div>
    </div>
  );
}

// ── Inline form (pitch-chat Compare mode, per-column) ────────────────

export function ThinkingInline() {
  const elapsed = useElapsedSeconds();
  return (
    <span className="inline-flex items-center gap-1 text-titan-metal/50 italic">
      <span>thinking</span>
      <BouncingDots size="xs" />
      <ElapsedBadge seconds={elapsed} />
    </span>
  );
}

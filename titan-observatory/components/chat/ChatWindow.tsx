'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePrivy } from '@privy-io/react-auth';
import type { ChatMessage, ChatResponse, ChatErrorResponse } from '@/lib/chat';
import {
  getStoredSessionId,
  getStoredMessages,
  storeMessages,
  sendChatMessage,
} from '@/lib/chat';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';
import ThinkingIndicator from './ThinkingIndicator';
import MakerPanel from './MakerPanel';
import { useNeuromodulators, useDreaming } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { usePrivyMaker } from '@/hooks/usePrivyMaker';

export default function ChatWindow() {
  const { authenticated, user, login, getAccessToken } = usePrivy();
  // Maker detection — gates the side panel rendering
  const walletAddress = (user?.wallet?.address as string | undefined) ?? null;
  const isMaker = usePrivyMaker(walletAddress);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const initialized = useRef(false);

  // V2: Mood-aware UI
  const titanId = useTitanId();
  const { data: nmData } = useNeuromodulators(titanId);
  const { data: dreamData } = useDreaming(titanId);
  const nm = (nmData ?? {}) as Record<string, unknown>;
  const emotion = nm?.current_emotion as string ?? '';
  const isDreaming = dreamData?.is_dreaming === true;

  // Neuromod-aware background tinting
  const mods = (nm?.modulators ?? {}) as Record<string, Record<string, number>>;
  const da = mods?.DA?.level ?? 0.5;
  const serotonin = mods?.['5HT']?.level ?? 0.5;
  const ne = mods?.NE?.level ?? 0.5;
  // Blend: DA→warm amber, 5HT→cool blue, NE→bright cyan
  const bgOpacity = isDreaming ? 0.08 : 0.03;
  const chatBgStyle = {
    background: isDreaming
      ? `radial-gradient(ellipse at 50% 0%, rgba(99,102,241,${bgOpacity}) 0%, transparent 70%)`
      : `radial-gradient(ellipse at 50% 0%, rgba(${Math.round(da * 200)},${Math.round(serotonin * 140)},${Math.round(ne * 220)},${bgOpacity}) 0%, transparent 70%)`,
  };

  // Load persisted messages and session on mount
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    setSessionId(getStoredSessionId());
    setMessages(getStoredMessages());
  }, []);

  // Persist messages on change
  useEffect(() => {
    if (messages.length > 0) {
      storeMessages(messages);
    }
  }, [messages]);

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, loading]);

  const handleSend = useCallback(
    async (text: string) => {
      if (!authenticated) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: text,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const token = await getAccessToken();
        const userId = user?.id;
        // titanId from useTitanId() → URL ?titan=T2|T3 — routes chat
        // through nginx /t2/* or /t3/* prefix to the right Titan.
        // Same pattern as MakerPanel (Phase 2 closure commit 7c49ed44);
        // closes the chat-UI hole exposed 2026-05-25 (504 HTML on T2
        // selection because the POST was hitting T1 default).
        const result = await sendChatMessage(
          text, sessionId, userId ?? undefined, token, titanId);

        if ('blocked' in result && (result as ChatErrorResponse).blocked) {
          const err = result as ChatErrorResponse;
          const titanMsg: ChatMessage = {
            id: crypto.randomUUID(),
            role: 'titan',
            content: err.error,
            timestamp: Date.now(),
            mode: err.mode || 'Guardian',
            blocked: true,
          };
          setMessages((prev) => [...prev, titanMsg]);
        } else {
          const ok = result as ChatResponse;
          const titanMsg: ChatMessage = {
            id: crypto.randomUUID(),
            role: 'titan',
            content: ok.response,
            timestamp: Date.now(),
            mode: ok.mode,
            mood: ok.mood,
          };
          setMessages((prev) => [...prev, titanMsg]);
        }
      } catch (err) {
        const errorMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'titan',
          content: err instanceof Error ? err.message : 'Connection failed. Titan may be offline.',
          timestamp: Date.now(),
          error: 'true',
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setLoading(false);
      }
    },
    [sessionId, authenticated, getAccessToken, user]
  );

  const handleClear = useCallback(() => {
    setMessages([]);
    if (typeof window !== 'undefined') {
      localStorage.removeItem('titan_chat_messages');
      localStorage.removeItem('titan_chat_session');
    }
    setSessionId(getStoredSessionId());
  }, []);

  if (!authenticated) {
    return (
      <div className="flex flex-col h-[calc(100vh-8rem)] items-center justify-center text-center px-6">
        <div className="max-w-sm space-y-6">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-solana-purple/10 border border-solana-purple/20 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8 text-solana-purple/60">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-semibold text-titan-steel mb-2">Authentication Required</h2>
            <p className="text-sm text-titan-metal/50 leading-relaxed">
              Sign in to chat with Titan. Connect a Solana wallet, or use Google, GitHub, or email.
            </p>
          </div>
          <button
            onClick={login}
            className="px-6 py-2.5 text-sm font-medium rounded-lg bg-solana-purple/20 text-solana-purple
                       hover:bg-solana-purple/30 border border-solana-purple/30 hover:border-solana-purple/50
                       transition-all"
          >
            Sign In
          </button>
          <p className="text-[11px] text-titan-metal/30">
            Conversations pass through the Sage Gatekeeper pipeline — responses are routed by IQL advantage scoring.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-8rem)]">
      <div className="flex flex-col flex-1 min-w-0">
      {/* Header bar — mood-aware */}
      <div className={`flex items-center justify-between px-4 py-2 border-b transition-colors duration-1000 ${
        isDreaming ? 'border-blue-500/20 bg-blue-900/10' : 'border-titan-metal/10'
      }`}>
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${isDreaming ? 'bg-blue-400' : 'bg-titan-growth'} animate-pulse-slow`} />
          <span className="text-xs text-titan-metal/60 font-mono">
            session: {sessionId.slice(0, 16)}
          </span>
          {emotion && (
            <span className={`text-xs font-titan ${isDreaming ? 'text-blue-400' : 'text-titan-haze'}`}>
              {isDreaming ? '🌙 dreaming' : emotion}
            </span>
          )}
        </div>
        <button
          onClick={handleClear}
          className="text-[10px] text-titan-metal/40 hover:text-titan-metal/70 uppercase tracking-wider transition-colors"
        >
          Clear
        </button>
      </div>

      {/* Dreaming notice */}
      {isDreaming && (
        <div className="px-4 py-2 bg-blue-900/20 text-blue-300 text-xs text-center">
          Titan is dreaming — messages will be queued and processed on waking
        </div>
      )}

      {/* Messages area — neuromod-tinted background */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 space-y-4 transition-all duration-2000"
        style={chatBgStyle}
      >
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-titan-haze/30 text-4xl mb-4">&#9672;</div>
            <p className="text-titan-metal/40 text-sm max-w-md">
              Speak with Titan. Messages pass through the Sage Gatekeeper pipeline
              — responses are routed by IQL advantage scoring.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {loading && <ThinkingIndicator />}
      </div>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={loading} />
      </div>
      {/* Maker Panel — Tier 1 R8 + future bond substrate. Only renders for Maker. */}
      {isMaker && <MakerPanel />}
    </div>
  );
}

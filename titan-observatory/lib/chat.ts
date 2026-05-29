// Same-origin in browser (avoid CORS); env-var-based on server.
// See lib/api.ts _resolveApiBase for the rationale.
function _resolveChatBase(): string {
  const raw = typeof window === 'undefined'
    ? (process.env.NEXT_PUBLIC_TITAN_API_URL || 'http://localhost:7777')
    : '';
  // Route chat through /api/ prefix so nginx (prod) or Next.js rewrites (dev)
  // forwards to backend, not the Next.js /chat page route.
  return raw.endsWith('/api') ? raw : `${raw}/api`;
}
const API_BASE = _resolveChatBase();

// Multi-Titan routing — see lib/api.ts TITAN_PREFIXES + nginx config.
//
// nginx contract (iamtitan.tech, verified 2026-05-25):
//   - T1 default: `/api/chat` → location /api/ → rewrite ^/api/(.*) /$1 →
//     proxy to 127.0.0.1:7777 (T1). proxy_read_timeout 10s.
//   - T2: `/t2/chat` → location ~ ^/t2/(...chat...) → rewrite ^/t2/(.*) /$1 →
//     proxy to 10.135.0.6:7777 (T2). proxy_read_timeout 30s.
//   - T3: `/t3/chat` → mirror of T2 block, 10.135.0.6:7778 backend.
//
// So T1 chat URLs MUST include the `/api/` prefix (default API base);
// T2/T3 chat URLs MUST NOT include `/api/` — they go directly through the
// per-Titan nginx blocks. `_buildChatPath` encodes this contract.
//
// Fix-class refs: feedback_frontend_rewire_mandatory_on_api_contract_change.md
// + feedback_api_version_prefix_routing_completeness.md (Phase 2 closure
// MakerPanel commit 7c49ed44 set the precedent for multi-Titan wiring on
// already-multi-Titan-routed nginx; the chat UI was the remaining hole).
//
// Note: the `/api/` block has proxy_read_timeout 10s while `/t2/` + `/t3/`
// have 30s. T1 cold-start chat (≈30s through agno's first OVG verify cycle)
// will 504 through nginx. This is a SEPARATE pre-existing nginx config gap
// (chat needs ≥60s read timeout to match the existing /api/pitch-chat-proxy
// block precedent at 200s). Tracked as a separate follow-up; T2/T3 work
// today because their nginx block already has 30s + chat is usually warm
// after first turn.
const CHAT_TITAN_PREFIXES = {
  T1: '',
  T2: '/t2',
  T3: '/t3',
} as const;
export type ChatTitanId = keyof typeof CHAT_TITAN_PREFIXES;

/** Build the chat-route URL for a given Titan. Encodes the nginx contract
 *  documented at CHAT_TITAN_PREFIXES — T1 uses `${API_BASE}/chat`
 *  (default `/api/chat`), T2/T3 use `/t2/chat` / `/t3/chat` directly
 *  (bypassing /api/ to hit the per-Titan nginx blocks). */
function _buildChatPath(titanId: ChatTitanId, suffix: string): string {
  if (titanId === 'T1') {
    return `${API_BASE}${suffix}`;
  }
  return `${CHAT_TITAN_PREFIXES[titanId]}${suffix}`;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'titan';
  content: string;
  timestamp: number;
  mode?: string;
  mood?: string;
  blocked?: boolean;
  error?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  mode: string;
  mood: string;
}

export interface ChatErrorResponse {
  error: string;
  blocked: boolean;
  mode: string;
}

const STORAGE_KEY = 'titan_chat_messages';
const SESSION_KEY = 'titan_chat_session';

export function generateSessionId(): string {
  return `web_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
}

export function getStoredSessionId(): string {
  if (typeof window === 'undefined') return generateSessionId();
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = generateSessionId();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function getStoredMessages(): ChatMessage[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function storeMessages(messages: ChatMessage[]): void {
  if (typeof window === 'undefined') return;
  try {
    // Keep last 200 messages to avoid localStorage bloat
    const trimmed = messages.slice(-200);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch {
    // Storage full — clear old messages
    localStorage.removeItem(STORAGE_KEY);
  }
}

export async function sendChatMessage(
  message: string,
  sessionId: string,
  userId?: string,
  accessToken?: string | null,
  titanId: ChatTitanId = 'T1'
): Promise<ChatResponse | ChatErrorResponse> {
  // Multi-Titan URL via the nginx contract — see _buildChatPath +
  // CHAT_TITAN_PREFIXES. T1 → /api/chat ; T2 → /t2/chat ; T3 → /t3/chat.
  const url = _buildChatPath(titanId, '/chat');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_id: userId,
    }),
  });

  const data = await res.json();

  if (res.status === 403 || data.blocked) {
    return data as ChatErrorResponse;
  }

  if (!res.ok) {
    throw new Error(data.error || `Chat API error: ${res.status}`);
  }

  return data as ChatResponse;
}

export async function streamChatMessage(
  message: string,
  sessionId: string,
  userId: string | undefined,
  onChunk: (text: string) => void,
  onMeta: (mode: string, mood: string) => void,
  onDone: () => void,
  onError: (err: string, blocked?: boolean) => void,
  accessToken?: string | null,
  titanId: ChatTitanId = 'T1'
): Promise<void> {
  // Multi-Titan: same routing policy as sendChatMessage — see
  // _buildChatPath. Stream endpoint sits under /chat/stream so the
  // prefix logic applies identically.
  const url = _buildChatPath(titanId, '/chat/stream');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }
  let res: Response;

  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        message,
        session_id: sessionId,
        user_id: userId,
      }),
    });
  } catch {
    // Stream endpoint not available — fall back to non-stream (carries
    // the same titanId so the fallback hits the same Titan).
    const fallback = await sendChatMessage(
      message, sessionId, userId, accessToken, titanId);
    if ('blocked' in fallback && fallback.blocked) {
      onError(fallback.error, true);
    } else if ('response' in fallback) {
      onChunk(fallback.response);
      onMeta(fallback.mode, fallback.mood);
      onDone();
    }
    return;
  }

  if (res.status === 403) {
    const data = await res.json();
    onError(data.error || 'Blocked by Guardian', data.blocked);
    return;
  }

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    onError(data.error || `Stream error: ${res.status}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError('No readable stream');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();

        if (payload === '[DONE]') {
          onDone();
          return;
        }

        try {
          const parsed = JSON.parse(payload);
          if (parsed.text) onChunk(parsed.text);
          if (parsed.mode) onMeta(parsed.mode, parsed.mood || '');
          if (parsed.error) onError(parsed.error, parsed.blocked);
        } catch {
          // Plain text chunk
          onChunk(payload);
        }
      }
    }
    onDone();
  } finally {
    reader.releaseLock();
  }
}

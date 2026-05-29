import { WSEvent } from './types';

const WS_URL = process.env.NEXT_PUBLIC_TITAN_WS_URL || 'ws://localhost:7777/ws';

type WSListener = (event: WSEvent) => void;

class WebSocketManager {
  private ws: WebSocket | null = null;
  private listeners: Set<WSListener> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 2000;
  private maxReconnectDelay = 30000;
  private isDestroyed = false;

  connect() {
    if (this.isDestroyed || this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        this.reconnectDelay = 2000;
      };

      this.ws.onmessage = (event) => {
        try {
          const parsed: WSEvent = JSON.parse(event.data);
          this.listeners.forEach((listener) => listener(parsed));
        } catch {
          // Ignore malformed messages
        }
      };

      this.ws.onclose = () => {
        if (!this.isDestroyed) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = () => {
        this.ws?.close();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer || this.isDestroyed) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
      this.connect();
    }, this.reconnectDelay);
  }

  subscribe(listener: WSListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  destroy() {
    this.isDestroyed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.ws?.close();
    this.listeners.clear();
  }
}

let manager: WebSocketManager | null = null;

export function getWSManager(): WebSocketManager {
  if (!manager) {
    manager = new WebSocketManager();
  }
  return manager;
}

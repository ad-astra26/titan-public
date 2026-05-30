// Thin client for the Console Agent. Mutations carry the X-Console-Token if the
// owner stored one in localStorage (the agent gates restart/clean/config/backup
// when ~/.titan/console_token is set).

const TOKEN_KEY = "tc2_console_token";

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) || "";
}
export function setToken(t: string) {
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const tok = getToken();
  if (tok && method !== "GET") headers["X-Console-Token"] = tok;
  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data: any = text;
  try { data = text ? JSON.parse(text) : {}; } catch { /* keep text */ }
  if (!res.ok) {
    const msg = (data && data.error) || `HTTP ${res.status}`;
    throw Object.assign(new Error(msg), { status: res.status, data });
  }
  return data as T;
}

export const api = {
  get: <T>(p: string) => req<T>("GET", p),
  post: <T>(p: string, b?: unknown) => req<T>("POST", p, b ?? {}),
  // Live cognition is proxied; the agent returns 503 {titan_down:true} when down.
  v6: <T>(p: string) => req<T>("GET", `/console/api${p.startsWith("/") ? p : "/" + p}`),
};

// ── route helpers ───────────────────────────────────────────────────────────
export const Routes = {
  health: () => api.get<any>("/console/health"),
  host: () => api.get<any>("/console/host"),
  titanStatus: () => api.get<any>("/console/titan-status"),
  journal: (lines = 60) => api.get<any>(`/console/journal?lines=${lines}`),
  backups: () => api.get<any>("/console/backups"),
  backupOptions: () => api.get<any>("/console/backup/options"),
  backupConfig: () => api.get<any>("/console/backup/config"),
  config: (section?: string) =>
    api.get<any>("/console/config" + (section ? `?section=${encodeURIComponent(section)}` : "")),
  restart: (force: boolean) => api.post<any>("/console/restart", { force }),
  cleanHdd: (confirm: boolean) => api.post<any>("/console/clean-hdd", { confirm }),
  setConfig: (key: string, value: string) => api.post<any>("/console/config/set", { key, value }),
  setBackupConfig: (fields: Record<string, unknown>) =>
    api.post<any>("/console/backup/config", fields),
  chat: (message: string, session?: string) =>
    api.post<any>("/console/chat", { message, session }),
};

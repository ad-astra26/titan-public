export function truncateHash(hash: string, chars = 6): string {
  if (!hash || hash.length < chars * 2 + 3) return hash || '';
  return `${hash.slice(0, chars)}...${hash.slice(-chars)}`;
}

export function formatSOL(lamports: number | undefined | null): string {
  if (lamports == null || isNaN(lamports)) return '0.0000';
  return lamports.toFixed(4);
}

export function formatPercent(value: number): string {
  return `${Math.round(value)}%`;
}

export function formatTimestamp(iso: string): string {
  if (!iso) return '';
  const date = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  });
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function timeUntil(iso: string | null): string {
  if (!iso) return 'N/A';
  const target = new Date(iso).getTime();
  const now = Date.now();
  const diffMs = target - now;
  if (diffMs <= 0) return 'now';

  const hours = Math.floor(diffMs / 3600000);
  const mins = Math.floor((diffMs % 3600000) / 60000);
  return `${hours}h ${mins}m`;
}

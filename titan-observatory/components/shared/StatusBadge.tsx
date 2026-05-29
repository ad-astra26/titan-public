'use client';

type BadgeStatus = 'ACTIVE' | 'DEGRADED' | 'STUB' | 'ABSENT';

interface StatusBadgeProps {
  status: BadgeStatus;
  label?: string;
}

const styles: Record<BadgeStatus, string> = {
  ACTIVE: 'bg-titan-growth/20 text-titan-growth border-titan-growth/30',
  DEGRADED: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  STUB: 'bg-titan-metal/20 text-titan-metal border-titan-metal/30',
  ABSENT: 'bg-red-500/20 text-red-400 border-red-500/30',
};

export default function StatusBadge({ status, label }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded border text-[10px] font-semibold uppercase tracking-wider ${
        styles[status] || styles.ABSENT
      }`}
    >
      {label || status}
    </span>
  );
}

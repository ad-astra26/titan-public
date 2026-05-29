'use client';

interface ModeBadgeProps {
  mode: string;
}

const modeStyles: Record<string, string> = {
  Sovereign: 'bg-titan-haze/20 text-titan-haze border-titan-haze/30',
  Collaborative: 'bg-titan-growth/20 text-titan-growth border-titan-growth/30',
  Research: 'bg-titan-pulse/20 text-[#b580ff] border-titan-pulse/30',
  Shadow: 'bg-titan-metal/20 text-titan-metal border-titan-metal/30',
  Guardian: 'bg-red-500/20 text-red-400 border-red-500/30',
};

export default function ModeBadge({ mode }: ModeBadgeProps) {
  const style = modeStyles[mode] || modeStyles.Shadow;

  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded border text-[10px] font-semibold uppercase tracking-wider ${style}`}
    >
      {mode}
    </span>
  );
}

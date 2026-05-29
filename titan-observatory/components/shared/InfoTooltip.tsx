'use client';

interface InfoTooltipProps {
  text: string;
  children?: React.ReactNode;
}

export default function InfoTooltip({ text, children }: InfoTooltipProps) {
  return (
    <span className="relative group inline-flex items-center">
      {children}
      <span className="ml-1 inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-titan-metal/20 text-[8px] text-titan-metal/40 cursor-help group-hover:border-titan-haze/40 group-hover:text-titan-haze/60 transition-colors">?</span>
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-50 pointer-events-none">
        <span className="block bg-titan-bg border border-titan-metal/20 rounded-lg px-3 py-2 text-[10px] text-titan-metal/70 leading-relaxed whitespace-normal min-w-[180px] max-w-[280px] shadow-lg">
          {text}
        </span>
      </span>
    </span>
  );
}

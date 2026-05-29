'use client';

import { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description: string;
  detail?: string;
  /** Optional pill slot — typically <LastUpdatedPill> from rFP §5.1 Phase 4. */
  pill?: ReactNode;
}

export default function PageHeader({ title, description, detail, pill }: PageHeaderProps) {
  return (
    <div className="mb-2 flex items-start justify-between gap-4">
      <div>
        <h2 className="text-lg font-semibold text-titan-haze">{title}</h2>
        <p className="text-xs text-titan-metal/50 mt-0.5 leading-relaxed">{description}</p>
        {detail && <p className="text-[10px] text-titan-metal/30 mt-0.5">{detail}</p>}
      </div>
      {pill && <div className="shrink-0 mt-1">{pill}</div>}
    </div>
  );
}

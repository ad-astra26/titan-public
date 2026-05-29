'use client';

import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { Suspense } from 'react';

export interface TabDef {
  id: string;
  label: string;
  description?: string;
}

function SubTabsInner({ tabs, activeTab, onTabChange }: {
  tabs: TabDef[];
  activeTab: string;
  onTabChange: (id: string) => void;
}) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto scrollbar-none border-b border-titan-metal/10 pb-px">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors duration-200 border-b-2 -mb-px ${
            activeTab === tab.id
              ? 'text-titan-haze border-titan-haze'
              : 'text-titan-metal/60 border-transparent hover:text-titan-haze/70 hover:border-titan-metal/20'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export default function SubTabs({ tabs, children }: {
  tabs: TabDef[];
  children: (activeTab: string) => React.ReactNode;
}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const activeTab = searchParams.get('tab') || tabs[0]?.id || '';
  const currentTab = tabs.find(t => t.id === activeTab) || tabs[0];
  const description = currentTab?.description;

  const handleTabChange = (id: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (id === tabs[0]?.id) {
      params.delete('tab');
    } else {
      params.set('tab', id);
    }
    const qs = params.toString();
    router.replace(`${pathname}${qs ? '?' + qs : ''}`, { scroll: false });
  };

  return (
    <div className="flex flex-col gap-4">
      <SubTabsInner tabs={tabs} activeTab={activeTab} onTabChange={handleTabChange} />
      {description && (
        <p className="text-xs text-titan-metal/50 -mt-2">{description}</p>
      )}
      {children(activeTab)}
    </div>
  );
}

export function SubTabsWrapper({ tabs, children }: {
  tabs: TabDef[];
  children: (activeTab: string) => React.ReactNode;
}) {
  return (
    <Suspense fallback={<div className="h-8" />}>
      <SubTabs tabs={tabs} children={children} />
    </Suspense>
  );
}

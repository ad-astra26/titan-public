'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

interface SidebarSection {
  label: string;
  items: { label: string; slug: string }[];
}

export default function DocsSidebar({ sections }: { sections: SidebarSection[] }) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (slug: string) => {
    const docPath = `/docs/${slug}`;
    return pathname === docPath || pathname === docPath + '/';
  };

  const sidebarContent = (
    <div className="space-y-5">
      {/* Docs home link */}
      <Link
        href="/docs"
        onClick={() => setMobileOpen(false)}
        className={`block text-sm font-semibold transition-colors ${
          pathname === '/docs' || pathname === '/docs/'
            ? 'text-titan-haze'
            : 'text-titan-metal/70 hover:text-titan-haze'
        }`}
      >
        Titan Docs
      </Link>

      {sections.map((section) => (
        <div key={section.label}>
          <p className="text-[10px] text-titan-metal/40 uppercase tracking-wider font-semibold mb-2">
            {section.label}
          </p>
          <div className="space-y-0.5">
            {section.items.map((item) => (
              <Link
                key={item.slug}
                href={`/docs/${item.slug}`}
                onClick={() => setMobileOpen(false)}
                className={`block text-xs py-1.5 px-2.5 rounded-lg transition-all duration-150 ${
                  isActive(item.slug)
                    ? 'text-titan-haze bg-titan-haze/10 font-medium'
                    : 'text-titan-metal/60 hover:text-titan-haze hover:bg-titan-card/50'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <>
      {/* Mobile toggle */}
      <div className="lg:hidden mb-4">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center gap-2 text-xs text-titan-metal/70 hover:text-titan-haze px-3 py-2 rounded-lg bg-titan-card/40 border border-titan-metal/10 transition-colors w-full"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <line x1="2" y1="4" x2="14" y2="4" />
            <line x1="2" y1="8" x2="14" y2="8" />
            <line x1="2" y1="12" x2="14" y2="12" />
          </svg>
          Documentation Menu
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            className={`ml-auto transition-transform ${mobileOpen ? 'rotate-180' : ''}`}
          >
            <polyline points="3,4.5 6,7.5 9,4.5" />
          </svg>
        </button>
        {mobileOpen && (
          <div className="mt-2 p-4 bg-titan-card/60 backdrop-blur-xl border border-titan-metal/10 rounded-xl">
            {sidebarContent}
          </div>
        )}
      </div>

      {/* Desktop sidebar */}
      <aside className="hidden lg:block w-56 flex-shrink-0 sticky top-24 self-start max-h-[calc(100vh-7rem)] overflow-y-auto scrollbar-none pr-4 border-r border-titan-metal/10">
        {sidebarContent}
      </aside>
    </>
  );
}

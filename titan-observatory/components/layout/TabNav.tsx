'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';

// Top-level information architecture (Phase 2 IA + TimeChain promotion 2026-05-10):
//   Home      → /            overview dashboard
//   Self      → /trinity     TitanSELF · Trinity · Architecture · I-Depth · Unified Spirit · Rhythms · Memory
//   Mind      → /neurology   Neurology (Neurochemistry, Dreams, Nervous System, Reflexes)
//   Voice     → /expression  Expression + Persona (Feed, Creative, Language, Reasoning, Social, Persona)
//   World     → /world       Kin + Compare + System (TimeChain promoted out)
//   TimeChain → /timechain   PROMOTED 2026-05-10 — Proof-of-Thought hallmark, top-level
//   Chat      → /chat        always last
const tabs = [
  { label: 'Home', href: '/' },
  { label: 'Self', href: '/trinity' },
  { label: 'Mind', href: '/neurology' },
  { label: 'Voice', href: '/expression' },
  { label: 'World', href: '/world' },
  { label: 'TimeChain', href: '/timechain' },
  { label: 'Chat', href: '/chat' },
];

export default function TabNav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Close menu on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <>
      {/* Desktop nav — horizontal tabs, hidden on small screens */}
      <nav className="hidden md:flex items-center gap-1 overflow-x-auto scrollbar-none">
        {tabs.map((tab) => {
          const isActive =
            tab.href === '/'
              ? pathname === '/'
              : pathname.startsWith(tab.href);

          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors duration-200 border-b-2 ${
                isActive
                  ? 'text-titan-haze border-titan-haze'
                  : 'text-titan-metal border-transparent hover:text-titan-haze/70'
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>

      {/* Mobile nav — hamburger + dropdown */}
      <div className="md:hidden flex items-center">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2 rounded-lg hover:bg-titan-card/50 transition-colors text-titan-metal"
          aria-label="Toggle navigation"
        >
          {mobileOpen ? (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="4" y1="4" x2="16" y2="16" />
              <line x1="16" y1="4" x2="4" y2="16" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="3" y1="5" x2="17" y2="5" />
              <line x1="3" y1="10" x2="17" y2="10" />
              <line x1="3" y1="15" x2="17" y2="15" />
            </svg>
          )}
        </button>

        {/* Current route indicator */}
        <span className="ml-2 text-xs text-titan-haze font-medium">
          {tabs.find(t => t.href === '/' ? pathname === '/' : pathname.startsWith(t.href))?.label || 'Home'}
        </span>
      </div>

      {/* Mobile dropdown menu */}
      {mobileOpen && (
        <div className="md:hidden absolute top-14 left-0 right-0 z-50 bg-titan-bg border-b border-titan-metal/10 shadow-lg">
          <div className="grid grid-cols-3 gap-1 p-3">
            {tabs.map((tab) => {
              const isActive =
                tab.href === '/'
                  ? pathname === '/'
                  : pathname.startsWith(tab.href);

              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  onClick={() => setMobileOpen(false)}
                  className={`px-3 py-2.5 text-sm font-medium rounded-lg text-center transition-all ${
                    isActive
                      ? 'text-titan-haze bg-titan-haze/10'
                      : 'text-titan-metal/70 hover:text-titan-haze hover:bg-titan-card/50'
                  }`}
                >
                  {tab.label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}

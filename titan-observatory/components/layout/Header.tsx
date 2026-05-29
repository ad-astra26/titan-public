'use client';

import Image from 'next/image';
import TabNav from './TabNav';
import GuardianShield from './GuardianShield';
import ThemeToggle from './ThemeToggle';
import WalletButton from './WalletButton';
import { useTitanStore } from '@/store/titanStore';

// 2026-05-14 — energy indicator removed from header entirely.
// The metabolic tier (driven by wallet SOL balance) is no longer surfaced in
// the public dashboard chrome. SOL balance still appears in the bottom metric
// grid as raw data; cognitive energy is represented by CHI_LIFE_FORCE on the
// home page. Internal /v4/metabolism endpoints continue to publish the
// canonical 6-state enum unchanged for tooling/alerting.

export default function Header() {
  const status = useTitanStore((s) => s.status);
  const wsConnected = useTitanStore((s) => s.wsConnected);

  return (
    <header className="sticky top-0 z-40 bg-titan-bg/90 backdrop-blur-xl border-b border-titan-metal/10 relative">
      <div className="max-w-[1440px] mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          {/* Logo + Name */}
          <div className="flex items-center gap-3 shrink-0">
            <Image
              src="/titan-logo.png"
              alt="Titan"
              width={28}
              height={28}
              className="rounded-md"
            />
            <span className="text-titan-haze font-semibold text-sm tracking-wide uppercase hidden sm:inline">
              Titan Observatory
            </span>
          </div>

          {/* Tab Navigation */}
          <div className="flex-1 mx-4 overflow-hidden">
            <TabNav />
          </div>

          {/* Right side: Wallet + Theme + Status */}
          <div className="flex items-center gap-2 shrink-0">
            {/* Wallet Connection */}
            <WalletButton />

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* WS Connection */}
            <div className="flex items-center gap-1.5" title={wsConnected ? 'WebSocket connected' : 'WebSocket disconnected'}>
              <div
                className={`w-2 h-2 rounded-full ${
                  wsConnected ? 'bg-titan-pulse animate-pulse-slow' : 'bg-titan-metal/30'
                }`}
              />
              <span className="text-[10px] text-titan-metal/60 hidden lg:inline">
                {wsConnected ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>

            {/* Memory Count */}
            {status && (
              <div
                className="text-[10px] text-titan-metal/60 bg-titan-card/50 px-2 py-1 rounded hidden md:block"
                title="Memory nodes"
              >
                {status.memory_count} memories
              </div>
            )}

            {/* Energy indicator removed 2026-05-14 — see file header comment. */}

            {/* Guardian Shield */}
            <GuardianShield />
          </div>
        </div>
      </div>
    </header>
  );
}

'use client';

import { ReactNode, useState, useEffect, lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';
import { toSolanaWalletConnectors } from '@privy-io/react-auth/solana';

const PRIVY_APP_ID = process.env.NEXT_PUBLIC_PRIVY_APP_ID || '';
const IS_LOCALHOST = typeof window !== 'undefined' && window.location.hostname === 'localhost';
const USE_PRIVY = PRIVY_APP_ID && !IS_LOCALHOST;

const solanaConnectors = toSolanaWalletConnectors({
  shouldAutoConnect: true,
});

// Lazy-load PrivyProvider so it doesn't block initial render
const LazyPrivyProvider = lazy(() =>
  import('@privy-io/react-auth').then((mod) => ({
    default: mod.PrivyProvider,
  }))
);

function PrivyWrapper({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [loginMethods, setLoginMethods] = useState<string[]>(['wallet', 'email']);

  useEffect(() => {
    if (!USE_PRIVY) return;

    // Wait for wallet extensions to inject their providers (Phantom, Solflare, etc.)
    // Then fetch Privy app config to auto-detect enabled login methods
    const initPrivy = async () => {
      // Wait up to 2s for Phantom to inject (checks every 100ms)
      for (let i = 0; i < 20; i++) {
        if (typeof window !== 'undefined' && ('phantom' in window || 'solana' in window)) {
          break;
        }
        await new Promise(r => setTimeout(r, 100));
      }

      try {
        const r = await fetch(`https://auth.privy.io/api/v1/apps/${PRIVY_APP_ID}`, {
          headers: { 'privy-app-id': PRIVY_APP_ID },
        });
        const config = await r.json();
        const methods: string[] = [];
        if (config.solana_wallet_auth || config.wallet_auth) methods.push('wallet');
        if (config.email_auth) methods.push('email');
        if (config.sms_auth) methods.push('sms');
        if (config.google_oauth) methods.push('google');
        if (config.github_oauth) methods.push('github');
        if (config.twitter_oauth) methods.push('twitter');
        if (config.discord_oauth) methods.push('discord');
        if (config.apple_oauth) methods.push('apple');
        if (config.farcaster_auth) methods.push('farcaster');
        if (config.telegram_auth) methods.push('telegram');
        if (methods.length > 0) setLoginMethods(methods);
      } catch {
        // Fallback to defaults
      }
      setReady(true);
    };

    initPrivy();
  }, []);

  if (!ready || !USE_PRIVY) return <>{children}</>;

  return (
    <Suspense fallback={<>{children}</>}>
      <LazyPrivyProvider
        appId={PRIVY_APP_ID}
        config={{
          appearance: {
            theme: '#0B0E14',
            accentColor: '#9945FF',
            logo: '/titan-logo.png',
            landingHeader: 'Welcome to Titan Observatory',
            loginMessage: 'Sign in to interact with Titan AI',
            showWalletLoginFirst: true,
            walletChainType: 'solana-only',
            walletList: ['phantom', 'solflare', 'detected_solana_wallets'],
          },
          loginMethods: loginMethods as never,
          externalWallets: {
            solana: {
              connectors: solanaConnectors,
            },
          },
        }}
      >
        {children}
      </LazyPrivyProvider>
    </Suspense>
  );
}

// Bump this to invalidate all persisted caches on schema changes
// v2 (2026-04-15): bust caches after META-CGN v3 Phase D
// v3 (2026-04-18): fix gcTime (was 30-300s per tier, now 24h default)
//   + fetch timeout (10s) so blocked API fails fast instead of hanging
// v4 (2026-05-19): NS payload shape changed to Phase B.5 lean schema
//   (titanvm_registers.bin: {programs, age_seconds, seq} with per-program
//   urgency; pre-B.5 fields version/training_phase/total_transitions/
//   total_train_steps/maturity REMOVED). Without this bump, persisted
//   query state from before 2026-05-19 hydrates the new components with
//   the OLD shape, which combined with placeholderData:keepPreviousData
//   renders fewer cards / blank tiles until live refetches finish.
// v5 — 2026-05-19 PM: bumped after /v6/nervous-system/hormonal-system + /status.lifetime
//   backend schema-drift closure. Invalidates persisted query state from
//   sessions where those endpoints returned empty/zero payloads — without
//   the bump, hydrated cache would mask the now-correct backend responses
//   under placeholderData:keepPreviousData until live refetch lands.
const CACHE_BUSTER = 'titan-observatory-v5';

export default function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: 2,
            staleTime: 8000,       // Active tier default
            gcTime: 1000 * 60 * 60 * 24, // 24h — keep for persistence hydration
            refetchOnMount: true,   // Respect staleTime instead of always refetching
          },
        },
      })
  );

  const [persister] = useState(() =>
    typeof window === 'undefined'
      ? null
      : createSyncStoragePersister({
          storage: window.localStorage,
          key: 'titan-observatory-cache',
          throttleTime: 1000,
        })
  );

  const inner = <PrivyWrapper>{children}</PrivyWrapper>;

  // SSR / no-window path: render without persistence (localStorage unavailable)
  if (!persister) {
    return <QueryClientProvider client={queryClient}>{inner}</QueryClientProvider>;
  }

  return (
    <PersistQueryClientProvider
      client={queryClient}
      persistOptions={{
        persister,
        maxAge: 1000 * 60 * 60 * 24, // 24h hard cap
        buster: CACHE_BUSTER,
      }}
    >
      {inner}
    </PersistQueryClientProvider>
  );
}

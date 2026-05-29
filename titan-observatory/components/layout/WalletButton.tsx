'use client';

import { usePrivy } from '@privy-io/react-auth';

export default function WalletButton() {
  const { ready, authenticated, user, login, logout } = usePrivy();

  if (!ready) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-titan-card/50 text-titan-metal/40 border border-titan-metal/10 cursor-default">
        <LoadingDot />
        Loading...
      </div>
    );
  }

  if (authenticated && user) {
    const label = getUserLabel(user);
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-titan-metal/70 truncate max-w-[120px]" title={label}>
          {label}
        </span>
        <button
          onClick={logout}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg
                     bg-titan-card/50 text-titan-metal hover:bg-titan-card hover:text-titan-gold
                     border border-titan-metal/10 hover:border-titan-gold/30 transition-all"
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={login}
      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg
                 bg-solana-purple/20 text-solana-purple hover:bg-solana-purple/30
                 border border-solana-purple/30 hover:border-solana-purple/50 transition-all"
    >
      <WalletIcon />
      Sign In
    </button>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getUserLabel(user: any): string {
  if (user.wallet?.address) {
    const addr = user.wallet.address as string;
    return `${addr.slice(0, 4)}...${addr.slice(-4)}`;
  }
  if (user.email?.address) return user.email.address;
  if (user.google?.email) return user.google.email;
  if (user.github?.username) return user.github.username;
  return 'Connected';
}

function LoadingDot() {
  return (
    <span className="w-2 h-2 rounded-full bg-titan-metal/30 animate-pulse" />
  );
}

function WalletIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
      <path d="M1 4.25a3.733 3.733 0 012.25-.75h13.5c.844 0 1.623.279 2.25.75A2.25 2.25 0 0016.75 2H3.25A2.25 2.25 0 001 4.25zM1 7.25a3.733 3.733 0 012.25-.75h13.5c.844 0 1.623.279 2.25.75A2.25 2.25 0 0016.75 5H3.25A2.25 2.25 0 001 7.25zM7 8a1 1 0 000 2h.01a1 1 0 000-2H7zm-.25 4a.75.75 0 01.75-.75h5a.75.75 0 010 1.5h-5a.75.75 0 01-.75-.75zm.75 2.25a.75.75 0 000 1.5h3a.75.75 0 000-1.5h-3z" />
      <path d="M3.25 8A2.25 2.25 0 001 10.25v5.5A2.25 2.25 0 003.25 18h13.5A2.25 2.25 0 0019 15.75v-5.5A2.25 2.25 0 0016.75 8H3.25z" />
    </svg>
  );
}

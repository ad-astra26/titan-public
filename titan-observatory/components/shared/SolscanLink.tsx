'use client';

import { truncateHash } from '@/lib/formatters';

interface SolscanLinkProps {
  address: string;
  type?: 'account' | 'tx';
  className?: string;
  truncate?: boolean;
}

export default function SolscanLink({
  address,
  type = 'account',
  className = '',
  truncate = true,
}: SolscanLinkProps) {
  const base = 'https://solscan.io';
  const path = type === 'tx' ? 'tx' : 'account';
  const href = `${base}/${path}/${address}`;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`text-titan-pulse hover:text-titan-pulse/80 transition-colors font-mono text-xs ${className}`}
    >
      {truncate ? truncateHash(address) : address}
      <span className="ml-1 opacity-60">&#8599;</span>
    </a>
  );
}

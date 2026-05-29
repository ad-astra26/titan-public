'use client';

import MetricCard from '@/components/shared/MetricCard';
import InfoTooltip from '@/components/shared/InfoTooltip';
import { useHealth } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import type { TimeChainStatus, PoTStats } from '@/hooks/useTitanAPI';

interface Props {
  status: TimeChainStatus | undefined;
  potStats: PoTStats | undefined;
}

function MetricWithTooltip({
  label, value, sublabel, accent, tooltip, href, linkLabel,
}: {
  label: string; value: string | number; sublabel?: string;
  accent?: 'haze' | 'growth' | 'pulse' | 'metal'; tooltip: string;
  href?: string; linkLabel?: string;
}) {
  const card = (
    <div className={`relative group ${href ? 'cursor-pointer hover:border-titan-growth/30' : ''}`}>
      <MetricCard label={label} value={value} sublabel={sublabel} accent={accent} />
      <div className="absolute -top-1 -right-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <InfoTooltip text={tooltip} />
      </div>
      {href && (
        <div className="absolute bottom-1 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <span className="text-[9px] text-titan-growth/60">{linkLabel || 'View on Solscan ↗'}</span>
        </div>
      )}
    </div>
  );

  if (href) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="no-underline">
        {card}
      </a>
    );
  }
  return card;
}

export default function ChainOverview({ status, potStats }: Props) {
  const titanId = useTitanId();
  const { data: health } = useHealth(titanId);

  if (!status) return null;

  const activeForks = Object.values(status.forks || {}).filter(
    (f) => f.block_count > 0
  ).length;

  const conversationBlocks = Object.values(status.forks || {}).find(
    (f) => f.name === 'conversation'
  )?.block_count ?? 0;

  // Solscan links — merkle links to latest anchor tx, genesis links to wallet account
  const pubkey = (health as unknown as Record<string, unknown>)?.maker_pubkey as string || '';
  const lastTxSig = status?.anchor?.last_tx_sig || '';
  const anchorCount = status?.anchor?.anchor_count || 0;

  // Debug: log what we have (remove after confirming)
  // console.log('[ChainOverview] anchor:', status?.anchor, 'lastTxSig:', lastTxSig, 'pubkey:', pubkey);

  // Merkle root → latest Solana anchor transaction (proves chain state was inscribed on-chain)
  // Always prefer tx link when available. Cluster comes from /health.network
  // (Titan went mainnet on 2026-04-06; ZK Vault program is mainnet-active).
  const cluster = (health as unknown as { network?: string } | undefined)?.network ?? 'mainnet-beta';
  const merkleSolscanUrl = lastTxSig.length > 10
    ? `https://solscan.io/tx/${lastTxSig}?cluster=${cluster}`
    : pubkey ? `https://solscan.io/account/${pubkey}?cluster=${cluster}` : '';
  // Genesis → wallet transactions tab (shows all memo inscriptions since birth)
  const genesisSolscanUrl = pubkey
    ? `https://solscan.io/account/${pubkey}/tx?cluster=${cluster}`
    : '';

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <MetricWithTooltip
        label="Total Blocks"
        value={status.total_blocks.toLocaleString()}
        sublabel={`${activeForks} active forks`}
        accent="pulse"
        tooltip="Total immutable blocks committed across all TimeChain forks. Each block represents a verified cognitive event. Updates every 3 seconds."
      />
      <MetricWithTooltip
        label="Chi Spent"
        value={status.total_chi_spent.toFixed(1)}
        sublabel="cognitive energy"
        accent="growth"
        tooltip="Chi (life force) spent on Proof of Thought — cognitive energy consumed to commit blocks. Higher chi = more significant thoughts."
      />
      <MetricWithTooltip
        label="Merkle Root"
        value={status.merkle_root?.slice(0, 10) || '...'}
        sublabel={anchorCount > 0 ? `${anchorCount.toLocaleString()} anchors on Solana` : 'chain integrity'}
        accent="haze"
        href={merkleSolscanUrl}
        tooltip="Cryptographic hash of the entire chain state. Click to view the latest Solana anchor transaction — proof that this chain state was inscribed on-chain."
        linkLabel="View latest anchor tx ↗"
      />
      <MetricWithTooltip
        label="Genesis"
        value={status.genesis_hash?.slice(0, 10) || 'none'}
        sublabel="birth certificate"
        accent="metal"
        href={genesisSolscanUrl}
        tooltip="Titan's birth certificate — the root of the entire TimeChain. Click to view all 65,000+ on-chain memo inscriptions since birth (March 11, 2026)."
        linkLabel="View all transactions ↗"
      />
      <MetricWithTooltip
        label="OVG Verified"
        value={conversationBlocks.toLocaleString()}
        sublabel="signed outputs"
        accent="growth"
        tooltip="Outputs verified by the Output Verification Gate and cryptographically signed with Titan's Ed25519 wallet key."
      />
      <MetricWithTooltip
        label="Avg Chi/Block"
        value={potStats?.avg_chi_per_block?.toFixed(4) || '0'}
        sublabel="thought cost"
        accent="metal"
        tooltip="Average chi (cognitive energy) spent per block. Higher significance thoughts cost more chi to commit."
      />
    </div>
  );
}

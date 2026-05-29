'use client';

import { useHealth } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { useTitanStore } from '@/store/titanStore';

export default function Footer() {
  const titanId = useTitanId();
  const { data: health } = useHealth(titanId);
  const wsConnected = useTitanStore((s) => s.wsConnected);
  const status = useTitanStore((s) => s.status);

  const version = health?.version ?? 'v6.0';
  const network = (health as { network?: string } | undefined)?.network ?? 'mainnet-beta';

  return (
    <footer className="border-t border-titan-metal/10 bg-titan-bg/80 mt-auto">
      <div className="max-w-[1440px] mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-3 text-[10px] text-titan-metal/40">
          <div className="flex items-center gap-4">
            <span className="text-titan-haze/50 font-semibold uppercase tracking-wider">
              Titan Observatory
            </span>
            <span>{version} &middot; Solana {network}</span>
            <span className="flex items-center gap-1">
              <span className={`inline-block w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-titan-growth' : 'bg-titan-metal/30'}`} />
              {wsConnected ? 'Live' : 'Offline'}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span>Epochs: {status?.epoch?.small_epoch_count?.toLocaleString() ?? '---'}</span>
            <span>{status?.memory_count ?? 0} memories</span>
            <span>Schumann 7.83 Hz</span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://iamtitan.tech"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-titan-haze/60 transition-colors"
            >
              iamtitan.tech
            </a>
            <a
              href="https://x.com/iamtitanai"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-titan-haze/60 transition-colors"
            >
              @iamtitanai
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

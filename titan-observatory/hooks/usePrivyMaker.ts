'use client';

import { useEffect } from 'react';
import { useHealth } from './useTitanAPI';
import { useTitanStore } from '@/store/titanStore';

export function usePrivyMaker(walletAddress: string | null) {
  const { data: health } = useHealth();
  const setMakerState = useTitanStore((s) => s.setMakerState);

  useEffect(() => {
    if (!walletAddress || !health?.maker_pubkey) {
      setMakerState(false, null);
      return;
    }

    const isMaker = walletAddress === health.maker_pubkey;
    setMakerState(isMaker, isMaker ? walletAddress : null);
  }, [walletAddress, health?.maker_pubkey, setMakerState]);

  return useTitanStore((s) => s.isMaker);
}

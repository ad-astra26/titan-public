'use client';

import { useTitanStore } from '@/store/titanStore';
import { EnergyState } from '@/lib/types';

export function useMetabolicMode(): {
  className: string;
  isLowPower: boolean;
  isReduced: boolean;
  frameloop: 'always' | 'demand';
  dpr: number | [number, number];
} {
  const energyState = useTitanStore((s) => s.energyState);

  // Map per `titan_plugin/core/metabolism.py:35-40` 6-state enum:
  //   THRIVING/HEALTHY = full power (legacy HIGH equivalent)
  //   CONSERVING       = reduced rate (legacy LOW equivalent)
  //   SURVIVAL         = consciousness + memory only (legacy STARVATION)
  //   EMERGENCY        = Contact Maker Protocol, minimal loop (legacy STARVATION)
  //   HIBERNATION      = save state and stop (legacy DEAD equivalent)
  const config: Record<
    EnergyState,
    {
      className: string;
      isLowPower: boolean;
      isReduced: boolean;
      frameloop: 'always' | 'demand';
      dpr: number | [number, number];
    }
  > = {
    THRIVING: {
      className: '',
      isLowPower: false,
      isReduced: false,
      frameloop: 'always',
      dpr: [1, 2],
    },
    HEALTHY: {
      className: '',
      isLowPower: false,
      isReduced: false,
      frameloop: 'always',
      dpr: [1, 2],
    },
    CONSERVING: {
      className: 'low-power',
      isLowPower: false,
      isReduced: true,
      frameloop: 'always',
      dpr: [1, 1.5],
    },
    SURVIVAL: {
      className: 'starvation',
      isLowPower: true,
      isReduced: true,
      frameloop: 'demand',
      dpr: 1,
    },
    EMERGENCY: {
      className: 'starvation',
      isLowPower: true,
      isReduced: true,
      frameloop: 'demand',
      dpr: 1,
    },
    HIBERNATION: {
      className: 'dead-mode',
      isLowPower: true,
      isReduced: true,
      frameloop: 'demand',
      dpr: 1,
    },
    UNKNOWN: {
      className: '',
      isLowPower: false,
      isReduced: false,
      frameloop: 'always',
      dpr: [1, 2],
    },
  };

  return config[energyState];
}

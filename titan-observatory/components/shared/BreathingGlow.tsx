'use client';

import { useMetabolicMode } from '@/hooks/useMetabolicMode';

interface BreathingGlowProps {
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
}

export default function BreathingGlow({ children, className = '', disabled = false }: BreathingGlowProps) {
  const { isLowPower } = useMetabolicMode();
  const shouldAnimate = !disabled && !isLowPower;

  return (
    <div className={`${shouldAnimate ? 'animate-breathe' : ''} ${className}`}>
      {children}
    </div>
  );
}

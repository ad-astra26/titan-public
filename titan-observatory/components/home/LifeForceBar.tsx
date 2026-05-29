'use client';

import { useTitanStore } from '@/store/titanStore';

export default function LifeForceBar() {
  const status = useTitanStore((s) => s.status);
  const lifeForce = status?.life_force ?? 0;
  const energyState = status?.energy_state ?? 'UNKNOWN';
  const sol = status?.sol_balance ?? 0;

  // Color tiers per `titan_plugin/core/metabolism.py:35-40` 6-state enum.
  // Public-facing palette: THRIVING/HEALTHY = growth-green; everything below
  // (CONSERVING / SURVIVAL / EMERGENCY / HIBERNATION) renders in neutral haze.
  // No red for low tiers — public optics, not internal alerting.
  const rawState = String(energyState);
  const stateColor =
    rawState === 'THRIVING' || rawState === 'HEALTHY'
      ? 'text-titan-growth'
      : 'text-titan-haze';

  // Display labels suppress raw enum names that read as alarming to the public.
  // Internal endpoints still return the canonical 6-state enum unchanged.
  const stateLabel: Record<string, string> = {
    THRIVING: 'THRIVING',
    HEALTHY: 'HEALTHY',
    CONSERVING: 'CONSERVING',
    SURVIVAL: 'CONSERVING',
    EMERGENCY: 'CONSERVING',
    HIBERNATION: 'LOW POWER',
    UNKNOWN: '—',
  };
  const displayLabel = stateLabel[rawState] ?? rawState.replace('_ENERGY', '');

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Metabolic Energy · SOL
          </h3>
          <span className={`text-xs font-mono ${stateColor}`}>
            {displayLabel}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-xs text-titan-metal/50">
            SOL <span className="text-titan-growth">{typeof sol === 'number' ? sol.toFixed(3) : '--'}</span>
          </span>
          <span className="text-sm font-semibold text-titan-growth">
            {Math.round(lifeForce)}%
          </span>
        </div>
      </div>
      <div className="h-3 bg-titan-metal/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-titan-growth rounded-full shadow-growth-glow transition-all duration-1000 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, lifeForce))}%` }}
        />
      </div>
    </div>
  );
}

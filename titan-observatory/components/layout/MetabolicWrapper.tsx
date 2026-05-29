'use client';

// 2026-05-14 — public banners + visual-state filters BOTH removed.
//
// Previously this wrapper applied `useMetabolicMode().className` (one of
// `low-power` / `starvation` / `dead-mode`) which globals.css mapped to:
//   .low-power  → filter: saturate(0.7)
//   .starvation → filter: saturate(0.5) + red radial overlay
//   .dead-mode  → filter: grayscale(1)
// That meant the entire dashboard rendered desaturated/monochrome whenever the
// wallet was low — but this conflated metabolic-tier state (wallet balance)
// with cognitive health (which CHI_LIFE_FORCE captures accurately). On low
// SOL, the whole dashboard looked dead even though Titan's cognition was fine.
//
// The hook (`useMetabolicMode`) still exists and is still consumed by
// components that legitimately want lower frameloop/dpr at low energy (3D
// canvases, etc.) — only the page-wide visual filter is removed here.
export default function MetabolicWrapper({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative min-h-screen">
      {children}
    </div>
  );
}

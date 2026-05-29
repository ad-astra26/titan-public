'use client';

import dynamic from 'next/dynamic';
import { useState, useCallback, useMemo } from 'react';
import {
  useTitanSelf, useResonanceEvents,
  FAMILY_COLOR, FAMILY_LABEL, FILTER_OPTIONS, FAMILY_DESCRIPTION,
  dimDescription, isVisible,
  type TitanDim, type FilterValue, type DimFamily,
} from './useTitanSelf';

const CellViz = dynamic(() => import('./CellViz'), { ssr: false });
const MandalaViz = dynamic(() => import('./MandalaViz'), { ssr: false });
const ConstellationViz = dynamic(() => import('./ConstellationViz'), { ssr: false });

/**
 * TitanSELF — the holistic 162D self visualization.
 *
 * 130D Trinity (65 inner + 65 outer) + 2D Journey + 30D Topology, all
 * arranged as a digital organism.
 *
 * Three R3F prototypes ship side by side — pick whichever to perfect.
 * Each prototype now also:
 *   • breathes on Titan's actual Schumann sphere-clock rhythms
 *     (1:3:9 Body:Mind:Spirit ratio preserved, phase + amplitude pulled
 *     live from /v6/trinity/sphere-clocks);
 *   • supports a left-side family filter so visitors can focus on one
 *     dimensional family at a time;
 *   • renders a right-side description panel listing the dimensions
 *     in the current filter with their function.
 */

const VIZ_OPTIONS = [
  { id: 'cell' as const,          label: 'Cell',          blurb: 'Nested organelle anatomy. Body=sphere, Mind=octahedron, Spirit=icosahedron, Topology=membrane rings, Journey=capsule. Breathes on Schumann.' },
  { id: 'mandala' as const,       label: 'Mandala',       blurb: 'Concentric rings rotating at golden-ratio multiples. Sacred-geometry readability — every dimensional family at once.' },
  { id: 'constellation' as const, label: 'Constellation', blurb: 'Two mirrored star clusters (Inner ↔ Outer) joined by a Journey binary at the core. Topology forms an outer halo.' },
];
type VizId = (typeof VIZ_OPTIONS)[number]['id'];

interface HoverState { dim: TitanDim; x: number; y: number; }

function FilterSelector({
  value, onChange,
}: {
  value: FilterValue;
  onChange: (v: FilterValue) => void;
}) {
  // Group filter options by tier for readability
  const tier0 = FILTER_OPTIONS.filter((o) => o.tier === 0);
  const tier1 = FILTER_OPTIONS.filter((o) => o.tier === 1);
  const tier2 = FILTER_OPTIONS.filter((o) => o.tier === 2);
  const Section = ({ items, label }: { items: typeof FILTER_OPTIONS; label?: string }) => (
    <>
      {label && (
        <div className="text-[8px] uppercase tracking-wider text-titan-metal/40 px-1 pt-1.5 pb-0.5">{label}</div>
      )}
      {items.map((o) => {
        const selected = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={`text-left text-[10px] px-2 py-1 rounded transition-colors ${
              selected
                ? 'bg-titan-haze/20 text-titan-haze'
                : 'text-titan-metal/70 hover:bg-titan-bg/60 hover:text-titan-metal'
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </>
  );
  return (
    <div className="absolute top-3 left-3 z-10 bg-titan-bg/85 backdrop-blur-md border border-titan-metal/20 rounded-lg p-1.5 flex flex-col gap-0.5 max-h-[calc(100%-1.5rem)] overflow-y-auto w-[180px]">
      <div className="text-[9px] font-semibold uppercase tracking-wider text-titan-haze px-1 pt-0.5">Filter</div>
      <Section items={tier0} />
      <Section items={tier1} label="Trinity / Layer" />
      <Section items={tier2} label="Specific family" />
    </div>
  );
}

function DescriptionsPanel({ filter, dims }: { filter: FilterValue; dims: TitanDim[] }) {
  if (filter === 'all') return null;

  const visibleDims = dims.filter((d) => isVisible(d, filter));
  const families = Array.from(new Set(visibleDims.map((d) => d.family)));

  return (
    <div className="absolute top-3 right-3 bottom-3 z-10 w-[280px] bg-titan-bg/90 backdrop-blur-md border border-titan-metal/20 rounded-lg flex flex-col">
      <div className="px-3 py-2 border-b border-titan-metal/15 shrink-0">
        <div className="text-[9px] font-semibold uppercase tracking-wider text-titan-haze">
          {FILTER_OPTIONS.find((o) => o.value === filter)?.label ?? 'Filter'}
        </div>
        {families.length === 1 && (
          <p className="text-[10px] text-titan-metal/60 leading-snug mt-1">
            {FAMILY_DESCRIPTION[families[0]]}
          </p>
        )}
        {families.length > 1 && (
          <p className="text-[10px] text-titan-metal/60 leading-snug mt-1">
            {visibleDims.length} dimensions across {families.length} families.
          </p>
        )}
      </div>

      <div className="overflow-y-auto px-2 py-2 flex flex-col gap-1.5">
        {families.map((fam) => {
          const familyDims = visibleDims.filter((d) => d.family === fam);
          const showFamilyHeader = families.length > 1;
          return (
            <div key={fam} className="flex flex-col gap-1">
              {showFamilyHeader && (
                <div
                  className="text-[9px] uppercase tracking-wider px-1 pt-1"
                  style={{ color: FAMILY_COLOR[fam] }}
                >
                  {FAMILY_LABEL[fam]} · {familyDims.length}D
                </div>
              )}
              {familyDims.map((d) => {
                const desc = dimDescription(d);
                return (
                  <div
                    key={`${d.family}-${d.index}`}
                    className="px-2 py-1.5 rounded bg-titan-card/40 border border-titan-metal/10"
                  >
                    <div className="flex items-baseline gap-2">
                      <span
                        className="font-mono text-[10px] font-semibold"
                        style={{ color: FAMILY_COLOR[d.family] }}
                      >
                        {d.label}
                      </span>
                      <span className="font-mono text-[9px] text-titan-haze ml-auto">
                        {d.raw.toFixed(3)}
                      </span>
                    </div>
                    {desc && (
                      <p className="text-[10px] text-titan-metal/60 leading-snug mt-0.5">
                        {desc}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ClockBadge({ state }: { state: ReturnType<typeof useTitanSelf> }) {
  // Tiny live indicator showing both Schumann breath and resonance state.
  // Resonance is the marquee event so it gets its own row.
  const fams: DimFamily[] = ['inner_body', 'inner_mind', 'inner_spirit'];
  const totalAmp = fams.reduce((s, f) => s + state.clocks[f].amplitude, 0) / fams.length;
  const live = state.clocks.inner_body.lastPulseAgeS < 5;
  const r = state.resonance;
  const pairs: { name: 'body'|'mind'|'spirit'; on: boolean; n: number }[] = [
    { name: 'body',   on: r.pairs.body.isResonant,   n: r.pairs.body.bigPulseCount },
    { name: 'mind',   on: r.pairs.mind.isResonant,   n: r.pairs.mind.bigPulseCount },
    { name: 'spirit', on: r.pairs.spirit.isResonant, n: r.pairs.spirit.bigPulseCount },
  ];
  return (
    <div className="absolute top-3 right-3 z-10 bg-titan-bg/85 backdrop-blur-md border border-titan-metal/20 rounded-lg px-2.5 py-1.5 text-[9px] font-mono text-titan-metal/70 flex flex-col gap-1 items-end">
      <div className="flex items-center gap-2">
        <span className={`w-1.5 h-1.5 rounded-full ${live ? 'bg-titan-growth animate-pulse-slow' : 'bg-titan-metal/30'}`} />
        <span>schumann · breath {(totalAmp * 100).toFixed(0)}%</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-titan-metal/40">resonance</span>
        {pairs.map((p) => (
          <span
            key={p.name}
            title={`${p.name}: ${p.n} big pulses`}
            className={`px-1 rounded ${
              p.on
                ? 'bg-titan-haze/30 text-titan-haze'
                : 'bg-titan-metal/10 text-titan-metal/40'
            }`}
          >
            {p.name[0].toUpperCase()}
          </span>
        ))}
        <span className="text-titan-metal/40">·</span>
        <span title={`${r.greatPulseCount} great pulses`}>★ {r.greatPulseCount}</span>
      </div>
    </div>
  );
}

export default function TitanSelfTab() {
  const state = useTitanSelf();
  const resonanceEvent = useResonanceEvents(state);
  const [active, setActive] = useState<VizId>('cell');
  const [filter, setFilter] = useState<FilterValue>('all');
  const [hover, setHover] = useState<HoverState | null>(null);

  const handleHover = useCallback((dim: TitanDim | null, screen?: { x: number; y: number }) => {
    if (dim && screen) setHover({ dim, x: screen.x, y: screen.y });
    else setHover(null);
  }, []);

  const activeOption = useMemo(() => VIZ_OPTIONS.find((o) => o.id === active)!, [active]);

  return (
    <div className="flex flex-col gap-4">
      {/* Viz switcher */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        {VIZ_OPTIONS.map((o) => {
          const selected = o.id === active;
          return (
            <button
              key={o.id}
              onClick={() => setActive(o.id)}
              className={`text-left rounded-xl border p-3 transition-all ${
                selected
                  ? 'bg-titan-haze/15 border-titan-haze text-titan-haze shadow-haze_glow'
                  : 'bg-titan-card/50 border-titan-metal/15 text-titan-metal hover:border-titan-haze/30'
              }`}
            >
              <div className="font-titan text-sm">{o.label}</div>
              <div className="text-[10px] mt-0.5 opacity-70 leading-tight">{o.blurb}</div>
            </button>
          );
        })}
      </div>

      {/* Visualization frame — fixed height, filter overlays inside */}
      <div className="relative bg-titan-card/40 border border-titan-metal/10 rounded-2xl overflow-hidden" style={{ height: 620 }}>
        {active === 'cell' && <CellViz filter={filter} resonanceEvent={resonanceEvent} onHover={handleHover} />}
        {active === 'mandala' && <MandalaViz filter={filter} resonanceEvent={resonanceEvent} onHover={handleHover} />}
        {active === 'constellation' && <ConstellationViz filter={filter} resonanceEvent={resonanceEvent} onHover={handleHover} />}

        {/* Filter (top-left) */}
        <FilterSelector value={filter} onChange={setFilter} />

        {/* Schumann breath badge (top-right when no filter detail panel) */}
        {filter === 'all' && <ClockBadge state={state} />}

        {/* Descriptions panel (right side, only when filter is active) */}
        <DescriptionsPanel filter={filter} dims={state.dims} />

        {/* Bottom-left legend */}
        <div className="absolute bottom-3 left-3 bg-titan-bg/80 backdrop-blur-sm border border-titan-metal/15 rounded-lg px-3 py-2 text-[10px] font-mono text-titan-metal/70 max-w-[260px] z-10">
          <div className="text-titan-haze font-semibold mb-1.5 uppercase tracking-wider text-[9px]">
            {activeOption.label} · 162D
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
            {(Object.keys(FAMILY_COLOR) as Array<keyof typeof FAMILY_COLOR>).map((k) => (
              <div key={k} className="flex items-center gap-1.5">
                <span
                  className="inline-block w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: FAMILY_COLOR[k] }}
                />
                <span className="truncate">{FAMILY_LABEL[k]}</span>
              </div>
            ))}
          </div>
          <div className="text-[9px] text-titan-metal/40 mt-1.5 italic">drag · zoom · hover · filter</div>
        </div>

        {/* Hover tooltip */}
        {hover && (
          <div
            className="pointer-events-none fixed z-50 bg-titan-bg/95 backdrop-blur-md rounded-lg px-3 py-2 text-xs shadow-lg"
            style={{
              left: hover.x + 12, top: hover.y + 12,
              borderColor: FAMILY_COLOR[hover.dim.family],
              borderWidth: 1, borderStyle: 'solid',
            }}
          >
            <div
              className="font-mono font-semibold mb-0.5"
              style={{ color: FAMILY_COLOR[hover.dim.family] }}
            >
              {FAMILY_LABEL[hover.dim.family]} · {hover.dim.label}
            </div>
            <div className="font-mono text-titan-metal text-[11px]">
              value <span className="text-titan-haze">{hover.dim.raw.toFixed(3)}</span>
              {' '}<span className="text-titan-metal/40">(idx {hover.dim.index})</span>
            </div>
          </div>
        )}
      </div>

      {/* Architectural caption */}
      <div className="bg-titan-card/40 border border-titan-metal/10 rounded-xl p-4 text-xs text-titan-metal/70 leading-relaxed">
        <span className="text-titan-haze font-semibold">TitanSELF · 162 dimensions.</span>{' '}
        130D Trinity (Inner Body 5D + Mind 15D + Spirit 45D, mirrored as Outer)
        + 2D Journey (the bridge between Inner and Outer) + 30D Space Topology
        (six × five descriptive stats: coherence, magnitude, velocity,
        direction, polarity over each Trinity component). Every node breathes
        on Titan&apos;s live Schumann sphere clocks at the 1:3:9 Body:Mind:Spirit
        ratio. Live data:{' '}
        <span className="font-mono text-titan-metal/50">/v6/trinity</span>,{' '}
        <span className="font-mono text-titan-metal/50">/v6/trinity/inner</span>,{' '}
        <span className="font-mono text-titan-metal/50">/v6/trinity/sphere-clocks</span>.
      </div>
    </div>
  );
}

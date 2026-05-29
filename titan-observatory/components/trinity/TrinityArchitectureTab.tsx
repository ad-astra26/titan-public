'use client';

import dynamic from 'next/dynamic';
import { useTrinityLive, useV4InnerTrinity } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import ErrorBoundary from '@/components/shared/ErrorBoundary';
import TrinityMatrix from '@/components/trinity/TrinityMatrix';
import GlobalObservablesMatrix from '@/components/trinity/GlobalObservablesMatrix';
import TopologyPanel from '@/components/trinity/TopologyPanel';
import DreamingOverlay from '@/components/trinity/DreamingOverlay';

const SphereClocks = dynamic(() => import('@/components/trinity/SphereClocks'), { ssr: false });

export default function TrinityArchitectureTab() {
  const titanId = useTitanId();
  const { data: trinityData, isLoading: trinityLoading } = useTrinityLive(titanId);
  const { data: coordData, isLoading: coordLoading } = useV4InnerTrinity(titanId);

  const rawTrinity = (trinityData ?? {}) as Record<string, unknown>;
  const innerTrinity = (rawTrinity?.trinity ?? rawTrinity) as Record<string, unknown>;
  const coord = (coordData ?? {}) as Record<string, unknown>;
  const topology = (coord?.topology ?? {}) as Record<string, unknown>;
  const observables = (rawTrinity?.observables ?? coord?.observables ?? {}) as Record<string, Record<string, number>>;
  const ns = (coord?.neural_nervous_system ?? {}) as Record<string, unknown>;
  const spiritVals = (innerTrinity?.spirit as Record<string, unknown>)?.values;
  const spiritArr = Array.isArray(spiritVals) ? spiritVals as number[] : [];
  const meta = Array.isArray(rawTrinity?.meta) ? rawTrinity.meta as number[]
    : [spiritArr[3] ?? 0, spiritArr[4] ?? 0];
  const v4 = (rawTrinity?.v4 ?? {}) as Record<string, unknown>;
  // Read order matters: kernel.py snapshot (coord.outer_trinity.{body,mind,spirit})
  // is the live microkernel v2 source — populated each tick from
  // OUTER_TRINITY_STATE bus messages. /v6/trinity returns flat
  // outer_body/outer_mind/outer_spirit but those are STALE [0.5]*5
  // defaults until v3 wiring catches up. Prefer the v4 nested shape;
  // fall back to v3 flats only if v4 is empty.
  const coordOuterTrinity = (coord?.outer_trinity ?? {}) as Record<string, unknown>;
  const pickOuter = (
    v4Nested: unknown, v4Flat: unknown, coordFlat: unknown,
    rawFlat: unknown,
  ): number[] => {
    // Prefer non-empty arrays in priority order: coord.outer_trinity.X
    // (live, microkernel v2) → v4.X → coord.X → rawTrinity.X (legacy v3).
    const candidates = [v4Nested, v4Flat, coordFlat, rawFlat];
    for (const c of candidates) {
      if (Array.isArray(c) && c.length > 0) return c as number[];
    }
    return [];
  };
  const outerData = {
    outer_body: pickOuter(
      coordOuterTrinity?.body, v4?.outer_body, coord?.outer_body,
      rawTrinity?.outer_body),
    outer_mind: pickOuter(
      coordOuterTrinity?.mind, v4?.outer_mind, coord?.outer_mind,
      rawTrinity?.outer_mind),
    outer_spirit: pickOuter(
      coordOuterTrinity?.spirit, v4?.outer_spirit, coord?.outer_spirit,
      rawTrinity?.outer_spirit),
  } as Record<string, unknown>;

  const isLoading = trinityLoading && coordLoading;

  if (isLoading) {
    return (
      <div className="bg-titan-card rounded-xl p-12 text-center text-titan-metal/40">
        Loading Trinity architecture...
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <DreamingOverlay>
        <div className="flex flex-col gap-4">
          {/* Unified Spirit summary */}
          <div className="bg-titan-card rounded-xl p-8 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-radial from-titan-haze/5 to-transparent" />
            <div className="relative z-10">
              <h2 className="text-lg font-titan text-titan-haze/80 mb-1">Unified Spirit</h2>
              <p className="text-xs text-titan-metal/40 mb-3">130+2D topology &middot; observes the whole being</p>
              {/* Phase B.5 lean schema: programs/age_seconds/seq only.
                  Pre-B.5 maturity/transitions/train-steps retired with
                  spirit_supplemental publisher (commit cf6a7793). */}
              <div className="flex justify-center gap-8 font-mono text-sm">
                {(() => {
                  const programs = (ns?.programs ?? {}) as Record<string, { fire_count?: number; total_updates?: number; urgency?: number }>;
                  const entries = Object.values(programs);
                  const fires = entries.reduce((s, p) => s + (p.fire_count ?? 0), 0);
                  const updates = entries.reduce((s, p) => s + (p.total_updates ?? 0), 0);
                  const urgs = entries.map(p => p.urgency ?? 0);
                  const avgU = urgs.length ? urgs.reduce((s, u) => s + u, 0) / urgs.length : 0;
                  return (
                    <>
                      <span className="text-titan-metal">programs <span className="text-titan-haze">{entries.length}</span></span>
                      <span className="text-titan-metal">avg urgency <span className="text-titan-haze">{(avgU * 100).toFixed(0)}%</span></span>
                      <span className="text-titan-metal">{'\u03a3'} fires <span className="text-titan-haze">{fires.toLocaleString()}</span></span>
                      <span className="text-titan-metal">{'\u03a3'} updates <span className="text-titan-haze">{(updates / 1_000_000).toFixed(1)}M</span></span>
                    </>
                  );
                })()}
              </div>
            </div>
          </div>

          {/* FILTER_DOWN indicator */}
          <div className="flex justify-center py-1">
            <div className="flex flex-col items-center gap-0.5 text-titan-haze/30">
              <span className="text-xs uppercase tracking-widest">filter down</span>
              <span>&#9660;</span>
            </div>
          </div>

          <SphereClocks />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <TrinityMatrix side="inner" data={innerTrinity} />
            <GlobalObservablesMatrix observables={observables} />
            <TrinityMatrix side="outer" data={outerData} />
          </div>

          <TopologyPanel topology={topology} meta={meta} />
        </div>
      </DreamingOverlay>
    </ErrorBoundary>
  );
}

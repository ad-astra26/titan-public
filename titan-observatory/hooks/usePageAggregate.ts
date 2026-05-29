'use client';

// ── Page-Aggregate Hooks (rFP §5.1 Phase 3) ─────────────────────
// One useQuery per page, one HTTP round-trip from the browser. Server-side
// bundles all upstream calls in parallel and serves from the BFF cache
// (same cache as /api/v6-cached/*).
//
// Pages opt-in by:
//   const { data, dataUpdatedAt, isFetching } = useTrinityPage(titanId);
// Pages also receive the `_meta` block including `stale`, `fetchedAt`,
// and per-endpoint errors (if any sub-endpoint failed).

import { useQuery } from '@tanstack/react-query';
import { pageFetch, tierQueryOptions, type TitanId } from '@/lib/api';
import { usePageAggregate } from '@/lib/feature-flags';

interface AggregateMeta {
  titan: TitanId;
  page: string;
  stale: boolean;
  fetchedAt: number;
  errors?: Array<{ endpoint: string; message: string }>;
}

export interface FeedPageData {
  narratedFeed: { items: unknown[]; count: number } | null;
  activityFeed: { items: unknown[]; count: number } | null;
  personaTelemetry: Record<string, unknown> | null;
  _meta: AggregateMeta;
}

export interface CreativePageData {
  creativeWorks: { items: unknown[] } | null;
  moodNarrative: { narrative: string } | null;
  _meta: AggregateMeta;
}

export interface TrinityPageData {
  trinity: Record<string, unknown> | null;
  neuromodulators: Record<string, unknown> | null;
  chi: Record<string, unknown> | null;
  dreaming: Record<string, unknown> | null;
  sphereClocks: Record<string, unknown> | null;
  metabolism: Record<string, unknown> | null;
  _meta: AggregateMeta;
}

export interface PersonaPageData {
  profiles: Record<string, unknown> | null;
  telemetry: Record<string, unknown> | null;
  socialPressure: Record<string, unknown> | null;
  _meta: AggregateMeta;
}

export interface MetabolismPageData {
  metabolism: Record<string, unknown> | null;
  maker: Record<string, unknown> | null;
  moodNarrative: { narrative: string } | null;
  _meta: AggregateMeta;
}

export interface TimechainPageData {
  timechain: Record<string, unknown> | null;
  _meta: AggregateMeta;
}

function aggregateOpts<T>(
  page: string,
  titanId: TitanId | undefined,
  refetchMs: number,
) {
  return {
    ...tierQueryOptions('realtime', ['page', page], titanId),
    queryFn: () => pageFetch<T>(page, { titan: titanId }),
    refetchInterval: refetchMs,
    enabled: usePageAggregate(page),
  };
}

/** Bundles narrated-feed + activity-feed + persona-telemetry. ~3s freshness. */
export function useFeedPage(titanId?: TitanId) {
  return useQuery<FeedPageData>(aggregateOpts('feed', titanId, 3_000));
}

/** Bundles creative-works + mood-narrative. ~3s freshness. */
export function useCreativePage(titanId?: TitanId) {
  return useQuery<CreativePageData>(aggregateOpts('creative', titanId, 3_000));
}

/** Bundles inner-trinity + neuromods + chi + dreaming + sphere-clocks + metabolism.
 *  6 upstream calls → 1 round-trip. */
export function useTrinityPage(titanId?: TitanId) {
  return useQuery<TrinityPageData>(aggregateOpts('trinity', titanId, 3_000));
}

/** Bundles persona-profiles + persona-telemetry + social-pressure. */
export function usePersonaPage(titanId?: TitanId) {
  return useQuery<PersonaPageData>(aggregateOpts('persona', titanId, 5_000));
}

/** Bundles metabolism + maker + mood-narrative. */
export function useMetabolismPage(titanId?: TitanId) {
  return useQuery<MetabolismPageData>(aggregateOpts('metabolism', titanId, 5_000));
}

/** Single-endpoint aggregate for consistency. */
export function useTimechainPage(titanId?: TitanId) {
  return useQuery<TimechainPageData>(aggregateOpts('timechain', titanId, 10_000));
}

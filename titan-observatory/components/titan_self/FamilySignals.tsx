'use client';

import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import {
  FAMILY_COLOR,
  type DimFamily, type FilterValue,
} from './useTitanSelf';

/**
 * Brain-cell-style INTRA-FAMILY signal pathways.
 *
 * Each Trinity family (inner_body, inner_mind, inner_spirit, outer_body,
 * outer_mind, outer_spirit, topology) is connected as a closed loop —
 * a "neural pathway" through every node of that family in spatial order.
 * Subtle lines render the static pathway; small bright pulses lerp from
 * one node to the next at family-specific speeds, evoking signals
 * propagating between neurons within a brain region.
 *
 * Speeds (nodes traversed per second), fastest → slowest per Maker spec:
 *   inner_spirit  7.0    (45 nodes ≈ 6.4s loop)
 *   inner_mind    5.0    (15 nodes ≈ 3.0s loop)
 *   inner_body    3.5    (5 nodes ≈ 1.4s loop)
 *   outer_spirit  2.5
 *   outer_mind    1.5
 *   outer_body    0.8    (slowest)
 *   topology      1.2
 *
 * Different speeds create a layered, lively rhythm even with no filter
 * applied — visitors can see the inner spirit firing fast while outer
 * body crawls slow, all simultaneously.
 */

export const SIGNAL_SPEED_NPS: Record<DimFamily, number> = {
  inner_spirit: 7.0,
  inner_mind:   5.0,
  inner_body:   3.5,
  outer_spirit: 2.5,
  outer_mind:   1.5,
  outer_body:   0.8,
  topology:     1.2,
  journey:      0,  // journey has no loop (only 2 nodes — handled by spine/binary core)
};

export const SIGNAL_PULSES: Record<DimFamily, number> = {
  inner_spirit: 3,
  inner_mind:   2,
  inner_body:   1,
  outer_spirit: 3,
  outer_mind:   2,
  outer_body:   1,
  topology:     2,
  journey:      0,
};

/**
 * Greedy nearest-neighbour traversal — produces a smooth path through
 * every point so the pathway lines don't crisscross visually. Cheap
 * enough at family sizes (≤45) to compute once at mount.
 */
export function orderByNearestPath(points: THREE.Vector3[]): THREE.Vector3[] {
  if (points.length < 2) return points.slice();
  const remaining = points.slice();
  const out: THREE.Vector3[] = [remaining.shift()!];
  while (remaining.length > 0) {
    const last = out[out.length - 1];
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const d = last.distanceToSquared(remaining[i]);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    out.push(remaining.splice(bestIdx, 1)[0]);
  }
  return out;
}

export interface FamilyLoop {
  family: DimFamily;
  /** Ordered node positions forming the closed-loop pathway. */
  positions: THREE.Vector3[];
}

interface PulseProps {
  loop: FamilyLoop;
  phaseOffset: number;
  visible: boolean;
  /** Boost factor when the family's pair is currently resonating; pulses
   *  brighten + stretch. 1.0 = normal, ~2.0 during pair resonance. */
  resonanceBoost: number;
}

function Pulse({ loop, phaseOffset, visible, resonanceBoost }: PulseProps) {
  const ref = useRef<THREE.Mesh>(null);
  const color = FAMILY_COLOR[loop.family];
  const speed = SIGNAL_SPEED_NPS[loop.family];

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const n = loop.positions.length;
    if (n < 2) return;
    const t = clock.getElapsedTime();
    const phase = (t * speed + phaseOffset * n) % n;
    const i = Math.floor(phase);
    const frac = phase - i;
    const a = loop.positions[i];
    const b = loop.positions[(i + 1) % n];
    ref.current.position.x = a.x + (b.x - a.x) * frac;
    ref.current.position.y = a.y + (b.y - a.y) * frac;
    ref.current.position.z = a.z + (b.z - a.z) * frac;
    // breathe slightly along the journey so each pulse has a wavefront feel
    const breath = 0.85 + 0.4 * Math.sin(phase * 2 * Math.PI);
    ref.current.scale.setScalar((0.7 + breath * 0.5) * resonanceBoost);
    const m = ref.current.material as THREE.MeshStandardMaterial;
    m.emissiveIntensity = 2.2 * (0.6 + breath * 0.6) * resonanceBoost;
  });

  return (
    <mesh ref={ref} visible={visible}>
      <sphereGeometry args={[0.045, 8, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={2.0}
        toneMapped={false}
      />
    </mesh>
  );
}

function PathwayLines({ loop, visible }: { loop: FamilyLoop; visible: boolean }) {
  const geometry = useMemo(() => {
    const points: number[] = [];
    const n = loop.positions.length;
    for (let i = 0; i < n; i++) {
      const a = loop.positions[i];
      const b = loop.positions[(i + 1) % n];
      points.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    return g;
  }, [loop]);

  return (
    <lineSegments geometry={geometry} visible={visible}>
      <lineBasicMaterial
        color={FAMILY_COLOR[loop.family]}
        transparent
        opacity={0.10}
      />
    </lineSegments>
  );
}

function familyVisible(family: DimFamily, filter: FilterValue): boolean {
  if (filter === 'all') return true;
  if (filter === family) return true;
  if (filter === 'inner_trinity' && family.startsWith('inner_')) return true;
  if (filter === 'outer_trinity' && family.startsWith('outer_')) return true;
  return false;
}

export interface FamilySignalsProps {
  loops: FamilyLoop[];
  filter: FilterValue;
  /** When the matching pair is resonating, that family's pulses double
   *  in count and brightness. Pass the active resonance pair (or null). */
  resonatingPair: 'body' | 'mind' | 'spirit' | null;
}

export default function FamilySignals({ loops, filter, resonatingPair }: FamilySignalsProps) {
  return (
    <group>
      {loops.map((loop) => {
        const visible = familyVisible(loop.family, filter);
        const isMatchingPair = !!resonatingPair && loop.family.endsWith(`_${resonatingPair}`);
        const baseCount = SIGNAL_PULSES[loop.family];
        const count = isMatchingPair ? baseCount * 2 : baseCount;
        const boost = isMatchingPair ? 1.7 : 1.0;
        return (
          <group key={loop.family}>
            <PathwayLines loop={loop} visible={visible} />
            {Array.from({ length: count }).map((_, p) => (
              <Pulse
                key={p}
                loop={loop}
                phaseOffset={p / count}
                visible={visible}
                resonanceBoost={boost}
              />
            ))}
          </group>
        );
      })}
    </group>
  );
}

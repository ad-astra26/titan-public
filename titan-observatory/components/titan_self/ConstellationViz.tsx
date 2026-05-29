'use client';

import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useMemo, useRef, useState, useEffect } from 'react';
import {
  useTitanSelf, FAMILY_COLOR, VISUAL_FREQ_HZ, isVisible, flashIntensity,
  type TitanDim, type FilterValue, type SphereClockState,
  type ResonanceEvent, type DimFamily,
} from './useTitanSelf';
import FamilySignals, { orderByNearestPath, type FamilyLoop } from './FamilySignals';

/**
 * TitanSELF — "The Constellation" visualization.
 *
 * Each of the 162 dimensions is a star in 3D space. Inner Trinity stars
 * cluster on the LEFT, Outer mirrored on the RIGHT, joined by a Journey
 * binary core. Topology forms an outer halo. Faint inner↔outer pairing
 * lines reveal the mirror structure.
 *
 * Each family's twinkle frequency follows the Schumann 1:3:9 ratio,
 * with phase + amplitude pulled live from /v6/trinity/sphere-clocks.
 */

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clusterPos(rand: () => number, centerX: number, yLayer: number, spread: number): [number, number, number] {
  const x = centerX + (rand() - 0.5) * spread;
  const y = yLayer * 1.4 + (rand() - 0.5) * spread * 0.7;
  const z = (rand() - 0.5) * spread;
  return [x, y, z];
}

function fibonacciSphere(n: number, r: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  const gr = (1 + Math.sqrt(5)) / 2;
  for (let i = 0; i < n; i++) {
    const theta = Math.acos(1 - (2 * (i + 0.5)) / n);
    const phi = (2 * Math.PI * i) / gr;
    out.push([
      r * Math.sin(theta) * Math.cos(phi),
      r * Math.sin(theta) * Math.sin(phi),
      r * Math.cos(theta),
    ]);
  }
  return out;
}

interface StarProps {
  dim: TitanDim;
  pos: [number, number, number];
  visible: boolean;
  clock: SphereClockState;
  resonanceEvent: ResonanceEvent | null;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}

function Star({ dim, pos, visible, clock, resonanceEvent, onHover }: StarProps) {
  const ref = useRef<THREE.Mesh>(null);
  const color = FAMILY_COLOR[dim.family];
  const baseSize = 0.045 + dim.value * 0.09;

  useFrame(({ clock: tc }) => {
    if (!ref.current) return;
    const t = tc.getElapsedTime();
    // Schumann-tuned twinkle: family freq + sphere-clock phase + per-dim offset
    const phase = clock.phase + t * 2 * Math.PI * VISUAL_FREQ_HZ[dim.family] + dim.index * 0.61;
    const swing = clock.amplitude * 0.18;
    const flash = flashIntensity(dim.family, resonanceEvent, performance.now());
    ref.current.scale.setScalar((1 - swing + Math.sin(phase) * swing) * (1 + flash * 0.6));
    const m = ref.current.material as THREE.MeshStandardMaterial;
    const base = visible ? 0.8 + dim.value * 1.6 : 0.04;
    m.emissiveIntensity = base + flash * 5;
  });

  return (
    <mesh
      ref={ref}
      position={pos}
      visible={visible}
      onPointerOver={(e) => { if (visible) { e.stopPropagation(); onHover(dim, e); } }}
      onPointerOut={() => visible && onHover(null)}
    >
      <sphereGeometry args={[baseSize, 8, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={visible ? 0.8 + dim.value * 1.6 : 0.04}
        roughness={0.2}
        transparent
        opacity={visible ? 1 : 0.08}
      />
    </mesh>
  );
}

function BinaryCore({
  dims, clock, visible,
}: {
  dims: TitanDim[];
  clock: SphereClockState;
  visible: boolean;
}) {
  const v0 = dims[0]?.value ?? 0.5;
  const v1 = dims[1]?.value ?? 0.5;
  const groupRef = useRef<THREE.Group>(null);

  useFrame(({ clock: tc }) => {
    if (!groupRef.current) return;
    const t = tc.getElapsedTime();
    groupRef.current.rotation.z = t * 0.4;
    const phase = clock.phase + t * 2 * Math.PI * VISUAL_FREQ_HZ.journey;
    groupRef.current.scale.setScalar(0.85 + Math.sin(phase) * 0.18 * (0.5 + clock.amplitude));
  });

  return (
    <group ref={groupRef} visible={visible}>
      <mesh position={[0.16, 0, 0]}>
        <sphereGeometry args={[0.10 + v0 * 0.05, 16, 16]} />
        <meshStandardMaterial color="#FFD080" emissive="#FFD080" emissiveIntensity={2.5} />
      </mesh>
      <mesh position={[-0.16, 0, 0]}>
        <sphereGeometry args={[0.10 + v1 * 0.05, 16, 16]} />
        <meshStandardMaterial color="#FFFAF0" emissive="#FFFAF0" emissiveIntensity={2.5} />
      </mesh>
    </group>
  );
}

function ConnectionLines({
  stars, filter,
}: {
  stars: { dim: TitanDim; pos: [number, number, number] }[];
  filter: FilterValue;
}) {
  const segments = useMemo(() => {
    const innerByKey = new Map<string, [number, number, number]>();
    const outerByKey = new Map<string, [number, number, number]>();
    for (const s of stars) {
      const f = s.dim.family;
      if (f === 'inner_body') innerByKey.set(`body-${s.dim.index}`, s.pos);
      if (f === 'inner_mind') innerByKey.set(`mind-${s.dim.index}`, s.pos);
      if (f === 'inner_spirit') innerByKey.set(`spirit-${s.dim.index}`, s.pos);
      if (f === 'outer_body') outerByKey.set(`body-${s.dim.index}`, s.pos);
      if (f === 'outer_mind') outerByKey.set(`mind-${s.dim.index}`, s.pos);
      if (f === 'outer_spirit') outerByKey.set(`spirit-${s.dim.index}`, s.pos);
    }
    const lines: [number, number, number][][] = [];
    innerByKey.forEach((ipos, key) => {
      const opos = outerByKey.get(key);
      if (opos) lines.push([ipos, opos]);
    });
    return lines;
  }, [stars]);

  const geometry = useMemo(() => {
    const positions: number[] = [];
    for (const [a, b] of segments) {
      positions.push(...a, ...b);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return g;
  }, [segments]);

  // Lines fade when a specific Trinity is filtered, since the mirror
  // pairing is a structural insight only meaningful when both halves
  // are visible.
  const lineOpacity = filter === 'all' ? 0.08 : filter === 'inner_trinity' || filter === 'outer_trinity' ? 0.04 : 0.02;

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#8E9AAF" transparent opacity={lineOpacity} />
    </lineSegments>
  );
}

function Scene({
  state, filter, resonanceEvent, onHover,
}: {
  state: ReturnType<typeof useTitanSelf>;
  filter: FilterValue;
  resonanceEvent: ResonanceEvent | null;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}) {
  // Build deterministic positions for every star — also expose them as
  // family-keyed maps so the FamilySignals loops can reuse identical
  // coordinates (signal pathways thread between the actual star positions).
  const { stars, byFamily } = useMemo(() => {
    const rand = rng(20260510);
    const out: { dim: TitanDim; pos: [number, number, number] }[] = [];
    const fam: Record<DimFamily, THREE.Vector3[]> = {
      inner_body: [], inner_mind: [], inner_spirit: [],
      outer_body: [], outer_mind: [], outer_spirit: [],
      journey: [], topology: [],
    };
    const push = (
      d: TitanDim, p: [number, number, number],
    ) => {
      out.push({ dim: d, pos: p });
      fam[d.family].push(new THREE.Vector3(p[0], p[1], p[2]));
    };
    for (const d of state.groups.inner_body)   push(d, clusterPos(rand, -2.5, -1, 1.0));
    for (const d of state.groups.inner_mind)   push(d, clusterPos(rand, -2.5,  0, 1.4));
    for (const d of state.groups.inner_spirit) push(d, clusterPos(rand, -2.5,  1, 1.7));
    for (const d of state.groups.outer_body)   push(d, clusterPos(rand,  2.5, -1, 1.0));
    for (const d of state.groups.outer_mind)   push(d, clusterPos(rand,  2.5,  0, 1.4));
    for (const d of state.groups.outer_spirit) push(d, clusterPos(rand,  2.5,  1, 1.7));
    const haloPos = fibonacciSphere(state.groups.topology.length, 5);
    state.groups.topology.forEach((d, i) => push(d, haloPos[i]));
    return { stars: out, byFamily: fam };
  }, [state]);

  // Intra-family pathways — each family is a closed loop walking through
  // every node of that cluster (greedy nearest-neighbour ordering).
  const familyLoops: FamilyLoop[] = useMemo(() => {
    const fams: DimFamily[] = [
      'inner_body', 'inner_mind', 'inner_spirit',
      'outer_body', 'outer_mind', 'outer_spirit',
      'topology',
    ];
    return fams
      .filter((f) => byFamily[f].length >= 2)
      .map((family) => ({ family, positions: orderByNearestPath(byFamily[family]) }));
  }, [byFamily]);

  const resonatingPair =
    resonanceEvent?.kind === 'pair' ? resonanceEvent.pair! : null;

  const journeyVisible = state.groups.journey.some((d) => isVisible(d, filter));

  return (
    <>
      <ambientLight intensity={0.18} />
      <pointLight position={[0, 0, 0]} intensity={1.2} color="#FFFAF0" distance={4} decay={2} />

      <BinaryCore dims={state.groups.journey} clock={state.clocks.journey} visible={journeyVisible} />
      <ConnectionLines stars={stars} filter={filter} />

      <FamilySignals
        loops={familyLoops}
        filter={filter}
        resonatingPair={resonatingPair ?? null}
      />

      {stars.map(({ dim, pos }) => (
        <Star
          key={`${dim.family}-${dim.index}`}
          dim={dim} pos={pos}
          visible={isVisible(dim, filter)}
          clock={state.clocks[dim.family]}
          resonanceEvent={resonanceEvent}
          onHover={onHover}
        />
      ))}

      <OrbitControls
        enableZoom enablePan={false}
        minDistance={5} maxDistance={16}
        autoRotate autoRotateSpeed={0.2}
        dampingFactor={0.06} enableDamping
      />
    </>
  );
}

export interface VizProps {
  filter: FilterValue;
  resonanceEvent: ResonanceEvent | null;
  onHover: (dim: TitanDim | null, screen?: { x: number; y: number }) => void;
}

export default function ConstellationViz({ filter, resonanceEvent, onHover }: VizProps) {
  const state = useTitanSelf();
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);
  if (!mounted) return null;

  const handleHover = (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => {
    if (dim && e) onHover(dim, { x: e.clientX, y: e.clientY });
    else onHover(null);
  };

  return (
    <Canvas
      camera={{ position: [0, 1.5, 9], fov: 50 }}
      dpr={[1, 1.5]}
      style={{ background: '#070912' }}
    >
      <Scene state={state} filter={filter} resonanceEvent={resonanceEvent} onHover={handleHover} />
    </Canvas>
  );
}

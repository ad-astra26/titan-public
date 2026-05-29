'use client';

import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useMemo, useRef, useState, useEffect } from 'react';
import {
  useTitanSelf, FAMILY_COLOR, VISUAL_FREQ_HZ, isVisible, flashIntensity,
  type TitanDim, type DimFamily, type FilterValue, type SphereClockState,
  type ResonanceEvent,
} from './useTitanSelf';
import FamilySignals, { orderByNearestPath, type FamilyLoop } from './FamilySignals';

/**
 * TitanSELF — "The Cell" visualization.
 *
 * Single-celled digital organism with shape vocabulary per family so the
 * eye can read the architecture at a glance:
 *
 *   Body          → SPHERE        (organic, soft)
 *   Mind          → OCTAHEDRON    (8 facets — feel/think/will × 2 polarities)
 *   Spirit        → ICOSAHEDRON   (20 facets — crystalline identity)
 *   Topology      → TORUS rings   (membrane proteins on outer surface)
 *   Journey       → CAPSULE       (elongated flow between Inner ↔ Outer)
 *
 * Outer Trinity nodes use the same shape as Inner but lighter color +
 * lower opacity (mantle vs nucleus).
 *
 * Each family breathes on its own Schumann-tuned rhythm (1:3:9 ratio
 * preserved across Body / Mind / Spirit), with phase + amplitude pulled
 * live from /v6/trinity/sphere-clocks. The whole-cell membrane breathes too.
 */

function fibonacciDirs(n: number): [number, number, number][] {
  const dirs: [number, number, number][] = [];
  const gr = (1 + Math.sqrt(5)) / 2;
  for (let i = 0; i < n; i++) {
    const theta = Math.acos(1 - (2 * (i + 0.5)) / n);
    const phi = (2 * Math.PI * i) / gr;
    dirs.push([
      Math.sin(theta) * Math.cos(phi),
      Math.sin(theta) * Math.sin(phi),
      Math.cos(theta),
    ]);
  }
  return dirs;
}

type Geom = 'sphere' | 'octahedron' | 'icosahedron' | 'torus' | 'capsule';

const FAMILY_GEOM: Record<DimFamily, Geom> = {
  inner_body:   'sphere',
  inner_mind:   'octahedron',
  inner_spirit: 'icosahedron',
  outer_body:   'sphere',
  outer_mind:   'octahedron',
  outer_spirit: 'icosahedron',
  topology:     'torus',
  journey:      'capsule',
};

const FAMILY_MATERIAL: Record<DimFamily, { metalness: number; roughness: number; emissiveBoost: number }> = {
  inner_body:   { metalness: 0.10, roughness: 0.45, emissiveBoost: 1.0 },
  inner_mind:   { metalness: 0.45, roughness: 0.30, emissiveBoost: 1.2 }, // more metallic — sharper feel
  inner_spirit: { metalness: 0.65, roughness: 0.20, emissiveBoost: 1.3 }, // crystalline radiance
  outer_body:   { metalness: 0.10, roughness: 0.50, emissiveBoost: 0.7 },
  outer_mind:   { metalness: 0.45, roughness: 0.35, emissiveBoost: 0.85 },
  outer_spirit: { metalness: 0.60, roughness: 0.25, emissiveBoost: 0.9 },
  topology:     { metalness: 0.85, roughness: 0.15, emissiveBoost: 1.0 }, // shimmery — like surface proteins
  journey:      { metalness: 0.20, roughness: 0.30, emissiveBoost: 2.5 }, // glow strongly
};

function NodeGeometry({ geom, scale }: { geom: Geom; scale: number }) {
  switch (geom) {
    case 'sphere':      return <sphereGeometry args={[scale, 14, 14]} />;
    case 'octahedron':  return <octahedronGeometry args={[scale * 1.2, 0]} />;
    case 'icosahedron': return <icosahedronGeometry args={[scale * 1.1, 0]} />;
    case 'torus':       return <torusGeometry args={[scale * 1.3, scale * 0.45, 8, 14]} />;
    case 'capsule':     return <capsuleGeometry args={[scale * 0.7, scale * 1.4, 4, 10]} />;
  }
}

interface NodeMeshProps {
  dim: TitanDim;
  pos: [number, number, number];
  radius: number;
  visible: boolean;
  clock: SphereClockState;
  resonanceEvent: ResonanceEvent | null;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}

function NodeMesh({ dim, pos, radius, visible, clock, resonanceEvent, onHover }: NodeMeshProps) {
  const ref = useRef<THREE.Mesh>(null);
  const color = FAMILY_COLOR[dim.family];
  const geom = FAMILY_GEOM[dim.family];
  const mat = FAMILY_MATERIAL[dim.family];
  const intensity = (0.25 + dim.value * 1.0) * mat.emissiveBoost;

  const rotSeed = useMemo(() => [
    (dim.index * 0.31) % (Math.PI * 2),
    (dim.index * 0.71) % (Math.PI * 2),
    (dim.index * 0.13) % (Math.PI * 2),
  ] as [number, number, number], [dim.index]);

  useFrame(({ clock: tc }) => {
    if (!ref.current) return;
    const t = tc.getElapsedTime();
    const phase = clock.phase + t * 2 * Math.PI * VISUAL_FREQ_HZ[dim.family] + dim.index * 0.18;
    const swing = clock.amplitude * 0.16;
    const flash = flashIntensity(dim.family, resonanceEvent, performance.now());
    // Resonance flash adds a 0..0.5 scale boost AND a 0..6 emissive kick
    ref.current.scale.setScalar(radius * (1 - swing + Math.sin(phase) * swing) * (1 + flash * 0.45));
    ref.current.rotation.x = rotSeed[0] + t * 0.12;
    ref.current.rotation.y = rotSeed[1] + t * 0.18;
    const m = ref.current.material as THREE.MeshStandardMaterial;
    m.emissiveIntensity = (visible ? intensity : 0.05) + flash * 6;
  });

  return (
    <mesh
      ref={ref}
      position={pos}
      visible={visible}
      onPointerOver={(e) => { if (visible) { e.stopPropagation(); onHover(dim, e); } }}
      onPointerOut={() => visible && onHover(null)}
    >
      <NodeGeometry geom={geom} scale={1} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={visible ? intensity : 0.05}
        metalness={mat.metalness}
        roughness={mat.roughness}
        transparent
        opacity={visible ? (0.5 + dim.value * 0.5) : 0.06}
      />
    </mesh>
  );
}

function Membrane({ activity, clock }: { activity: number; clock: SphereClockState }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock: tc }) => {
    if (!ref.current) return;
    const t = tc.getElapsedTime();
    // Whole-cell breath synthesized from topology clock — slow drift
    const phase = clock.phase + t * 2 * Math.PI * VISUAL_FREQ_HZ.topology;
    const breath = 1 + Math.sin(phase) * 0.04 * (0.5 + clock.amplitude);
    ref.current.scale.setScalar(breath);
    const m = ref.current.material as THREE.MeshStandardMaterial;
    m.emissiveIntensity = 0.05 + activity * 0.15;
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[3.6, 48, 48]} />
      <meshStandardMaterial
        color="#FFE9C4"
        emissive="#E5C79E"
        emissiveIntensity={0.1}
        transparent
        opacity={0.06}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function JourneySpine({
  dims, clock, visible,
}: {
  dims: TitanDim[];
  clock: SphereClockState;
  visible: boolean;
}) {
  const v0 = dims[0]?.value ?? 0.5;
  const v1 = dims[1]?.value ?? 0.5;
  const refA = useRef<THREE.Mesh>(null);
  const refB = useRef<THREE.Mesh>(null);

  useFrame(({ clock: tc }) => {
    const t = tc.getElapsedTime();
    const phase = clock.phase + t * 2 * Math.PI * VISUAL_FREQ_HZ.journey;
    const intA = 1.5 + 0.5 * Math.sin(phase) * (0.5 + clock.amplitude);
    const intB = 1.5 + 0.5 * Math.sin(phase + Math.PI) * (0.5 + clock.amplitude);
    if (refA.current) (refA.current.material as THREE.MeshStandardMaterial).emissiveIntensity = intA;
    if (refB.current) (refB.current.material as THREE.MeshStandardMaterial).emissiveIntensity = intB;
  });

  return (
    <group visible={visible}>
      {/* Capsules for the Journey — elongated, suggesting flow */}
      <mesh ref={refA} position={[0, 0.20, 0]} rotation={[0, 0, Math.PI / 2]}>
        <capsuleGeometry args={[0.07 + v0 * 0.025, 0.18, 6, 12]} />
        <meshStandardMaterial color="#FFD080" emissive="#FFD080" emissiveIntensity={1.5} metalness={0.2} />
      </mesh>
      <mesh ref={refB} position={[0, -0.20, 0]} rotation={[0, 0, Math.PI / 2]}>
        <capsuleGeometry args={[0.07 + v1 * 0.025, 0.18, 6, 12]} />
        <meshStandardMaterial color="#FFD080" emissive="#FFD080" emissiveIntensity={1.5} metalness={0.2} />
      </mesh>
      <mesh>
        <cylinderGeometry args={[0.022, 0.022, 0.40, 8]} />
        <meshBasicMaterial color="#FFD080" transparent opacity={0.7} />
      </mesh>
    </group>
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
  // Layout layers: each family on its own concentric shell (fibonacci).
  // Anchor-direction lookup is also memoized so the signal lines below
  // can target a stable representative node per shell.
  const SHELL_R: Record<DimFamily, number> = {
    topology:     3.5,
    outer_body:   2.95,
    outer_mind:   2.7,
    outer_spirit: 2.45,
    inner_body:   1.85,
    inner_mind:   1.5,
    inner_spirit: 1.15,
    journey:      0.0,
  };
  const layers = useMemo(() => {
    const make = (dims: TitanDim[], nodeR: number) => {
      const dirs = fibonacciDirs(dims.length);
      const r = SHELL_R[dims[0]?.family ?? 'topology'];
      return dims.map((d, i) => ({
        dim: d,
        pos: [dirs[i][0] * r, dirs[i][1] * r, dirs[i][2] * r] as [number, number, number],
        nodeR,
      }));
    };
    return [
      ...make(state.groups.topology, 0.07),
      ...make(state.groups.outer_body, 0.085),
      ...make(state.groups.outer_mind, 0.075),
      ...make(state.groups.outer_spirit, 0.060),
      ...make(state.groups.inner_body, 0.10),
      ...make(state.groups.inner_mind, 0.085),
      ...make(state.groups.inner_spirit, 0.065),
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  // Intra-family signal pathways — every family becomes a closed loop
  // through its own nodes (ordered by greedy nearest-neighbour walk).
  // Pulses traverse each loop at family-specific speeds, evoking signal
  // propagation between neurons within a brain region.
  const familyLoops: FamilyLoop[] = useMemo(() => {
    const buildLoop = (
      family: DimFamily,
      count: number,
      shellR: number,
    ): FamilyLoop => {
      const dirs = fibonacciDirs(count);
      const positions = dirs.map(([x, y, z]) =>
        new THREE.Vector3(x * shellR, y * shellR, z * shellR),
      );
      return { family, positions: orderByNearestPath(positions) };
    };
    return [
      buildLoop('inner_body',   state.groups.inner_body.length,   SHELL_R.inner_body),
      buildLoop('inner_mind',   state.groups.inner_mind.length,   SHELL_R.inner_mind),
      buildLoop('inner_spirit', state.groups.inner_spirit.length, SHELL_R.inner_spirit),
      buildLoop('outer_body',   state.groups.outer_body.length,   SHELL_R.outer_body),
      buildLoop('outer_mind',   state.groups.outer_mind.length,   SHELL_R.outer_mind),
      buildLoop('outer_spirit', state.groups.outer_spirit.length, SHELL_R.outer_spirit),
      buildLoop('topology',     state.groups.topology.length,     SHELL_R.topology),
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.groups.inner_body.length, state.groups.inner_mind.length,
      state.groups.inner_spirit.length, state.groups.outer_body.length,
      state.groups.outer_mind.length, state.groups.outer_spirit.length,
      state.groups.topology.length]);

  const resonatingPair =
    resonanceEvent?.kind === 'pair' ? resonanceEvent.pair! : null;

  const journeyVisible = state.groups.journey.some((d) => isVisible(d, filter));

  return (
    <>
      <ambientLight intensity={0.25} />
      <pointLight position={[0, 0, 0]} intensity={1.2} color="#FFD080" distance={8} decay={2} />
      <pointLight position={[6, 4, 6]} intensity={0.4} color="#77CCCC" />
      <pointLight position={[-6, -4, -6]} intensity={0.3} color="#9945FF" />

      <Membrane activity={state.totalActivity} clock={state.clocks.topology} />
      <JourneySpine dims={state.groups.journey} clock={state.clocks.journey} visible={journeyVisible} />

      <FamilySignals
        loops={familyLoops}
        filter={filter}
        resonatingPair={resonatingPair ?? null}
      />

      {layers.map(({ dim, pos, nodeR }) => (
        <NodeMesh
          key={`${dim.family}-${dim.index}`}
          dim={dim}
          pos={pos}
          radius={nodeR}
          visible={isVisible(dim, filter)}
          clock={state.clocks[dim.family]}
          resonanceEvent={resonanceEvent}
          onHover={onHover}
        />
      ))}

      <OrbitControls
        enableZoom enablePan={false}
        minDistance={5} maxDistance={14}
        autoRotate autoRotateSpeed={0.4}
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

export default function CellViz({ filter, resonanceEvent, onHover }: VizProps) {
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
      camera={{ position: [0, 0, 9], fov: 45 }}
      dpr={[1, 1.5]}
      style={{ background: '#0B0E14' }}
    >
      <Scene state={state} filter={filter} resonanceEvent={resonanceEvent} onHover={handleHover} />
    </Canvas>
  );
}

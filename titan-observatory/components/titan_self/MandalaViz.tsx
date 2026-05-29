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

/**
 * TitanSELF — "The Mandala" visualization.
 *
 * Concentric rings rotating at golden-ratio multiples form a sacred
 * mandala. Each ring's nodes pulse with their dim values; each FAMILY
 * breathes on its own Schumann-tuned rhythm (1:3:9 Body:Mind:Spirit
 * preserved), with phase + amplitude pulled live from /v6/trinity/sphere-clocks.
 *
 *   center  · 2D Journey
 *   ring 1  · Inner Body 5D       (slow breath)
 *   ring 2  · Inner Mind 15D      (3× faster)
 *   ring 3  · Inner Spirit 45D    (9× faster — visible flicker)
 *   ring 4  · Outer Body 5D       (slow, mirrored)
 *   ring 5  · Outer Mind 15D
 *   ring 6  · Outer Spirit 45D
 *   halo    · 30D Topology drifting around the disk
 */

const GOLDEN = (1 + Math.sqrt(5)) / 2;

const RING_DEFS: { family: DimFamily; radius: number; rotSpeed: number }[] = [
  { family: 'inner_body',   radius: 0.7,  rotSpeed:  0.18                    },
  { family: 'inner_mind',   radius: 1.25, rotSpeed: -0.18 / GOLDEN           },
  { family: 'inner_spirit', radius: 1.85, rotSpeed:  0.18 / (GOLDEN ** 2)    },
  { family: 'outer_body',   radius: 2.55, rotSpeed: -0.18 / (GOLDEN ** 3)    },
  { family: 'outer_mind',   radius: 3.15, rotSpeed:  0.18 / (GOLDEN ** 4)    },
  { family: 'outer_spirit', radius: 3.85, rotSpeed: -0.18 / (GOLDEN ** 5)    },
];

interface RingNodeProps {
  dim: TitanDim;
  angleStep: number;
  index: number;
  radius: number;
  visible: boolean;
  clock: SphereClockState;
  resonanceEvent: ResonanceEvent | null;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}

function RingNode({ dim, angleStep, index, radius, visible, clock, resonanceEvent, onHover }: RingNodeProps) {
  const ref = useRef<THREE.Mesh>(null);
  const color = FAMILY_COLOR[dim.family];
  const angle = angleStep * index;
  const x = Math.cos(angle) * radius;
  const z = Math.sin(angle) * radius;
  const baseSize = 0.04 + dim.value * 0.07;

  useFrame(({ clock: tc }) => {
    if (!ref.current) return;
    const t = tc.getElapsedTime();
    const phase = clock.phase + t * 2 * Math.PI * VISUAL_FREQ_HZ[dim.family] + index * 0.18;
    const swing = clock.amplitude * 0.18;
    const flash = flashIntensity(dim.family, resonanceEvent, performance.now());
    ref.current.scale.setScalar((1 - swing + Math.sin(phase) * swing) * (1 + flash * 0.5));
    const m = ref.current.material as THREE.MeshStandardMaterial;
    const base = visible ? 0.6 + dim.value * 1.2 : 0.05;
    m.emissiveIntensity = base + flash * 5;
  });

  return (
    <mesh
      ref={ref}
      position={[x, 0, z]}
      visible={visible}
      onPointerOver={(e) => { if (visible) { e.stopPropagation(); onHover(dim, e); } }}
      onPointerOut={() => visible && onHover(null)}
    >
      <sphereGeometry args={[baseSize, 12, 12]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={visible ? 0.6 + dim.value * 1.2 : 0.05}
        roughness={0.3}
        transparent
        opacity={visible ? 1 : 0.08}
      />
    </mesh>
  );
}

function Ring({
  dims, radius, rotSpeed, filter, clock, resonanceEvent, onHover,
}: {
  dims: TitanDim[];
  radius: number;
  rotSpeed: number;
  filter: FilterValue;
  clock: SphereClockState;
  resonanceEvent: ResonanceEvent | null;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}) {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += rotSpeed * delta;
  });
  const angleStep = (2 * Math.PI) / Math.max(1, dims.length);
  const anyVisible = dims.some((d) => isVisible(d, filter));

  return (
    <group ref={ref}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius, 0.005, 6, 96]} />
        <meshBasicMaterial color="#8E9AAF" transparent opacity={anyVisible ? 0.12 : 0.04} />
      </mesh>
      {dims.map((d, i) => (
        <RingNode
          key={`${d.family}-${d.index}`}
          dim={d} angleStep={angleStep} index={i} radius={radius}
          visible={isVisible(d, filter)}
          clock={clock}
          resonanceEvent={resonanceEvent}
          onHover={onHover}
        />
      ))}
    </group>
  );
}

function JourneyHeart({
  dims, clock, visible,
}: {
  dims: TitanDim[];
  clock: SphereClockState;
  visible: boolean;
}) {
  const ref = useRef<THREE.Group>(null);
  const v0 = dims[0]?.value ?? 0.5;
  const v1 = dims[1]?.value ?? 0.5;
  useFrame(({ clock: tc }) => {
    if (!ref.current) return;
    const phase = clock.phase + tc.getElapsedTime() * 2 * Math.PI * VISUAL_FREQ_HZ.journey;
    ref.current.scale.setScalar(0.85 + Math.sin(phase) * 0.15 * (0.5 + clock.amplitude));
  });
  return (
    <group ref={ref} visible={visible}>
      <mesh position={[0.07, 0, 0]}>
        <sphereGeometry args={[0.075 + v0 * 0.04, 16, 16]} />
        <meshStandardMaterial color="#FFD080" emissive="#FFD080" emissiveIntensity={2.0} />
      </mesh>
      <mesh position={[-0.07, 0, 0]}>
        <sphereGeometry args={[0.075 + v1 * 0.04, 16, 16]} />
        <meshStandardMaterial color="#FFD080" emissive="#FFD080" emissiveIntensity={2.0} />
      </mesh>
    </group>
  );
}

function TopologyParticle({
  dim, basePos, visible, clock, onHover,
}: {
  dim: TitanDim;
  basePos: [number, number, number];
  visible: boolean;
  clock: SphereClockState;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock: tc }) => {
    if (!ref.current) return;
    const t = tc.getElapsedTime();
    const ph = dim.index * 0.41 + clock.phase;
    ref.current.position.x = basePos[0] + Math.sin(t * 0.3 + ph) * 0.15;
    ref.current.position.y = basePos[1] + Math.cos(t * 0.25 + ph) * 0.20;
    ref.current.position.z = basePos[2] + Math.sin(t * 0.35 + ph * 1.3) * 0.15;
    const breath = 1 + Math.sin(t * 2 * Math.PI * VISUAL_FREQ_HZ.topology + ph) * clock.amplitude * 0.25;
    ref.current.scale.setScalar(breath);
  });
  return (
    <mesh
      ref={ref}
      position={basePos}
      visible={visible}
      onPointerOver={(e) => { if (visible) { e.stopPropagation(); onHover(dim, e); } }}
      onPointerOut={() => visible && onHover(null)}
    >
      <sphereGeometry args={[0.05 + dim.value * 0.04, 10, 10]} />
      <meshStandardMaterial
        color="#E5C79E"
        emissive="#E5C79E"
        emissiveIntensity={visible ? 0.7 + dim.value * 0.8 : 0.05}
        transparent
        opacity={visible ? 0.65 + dim.value * 0.3 : 0.06}
      />
    </mesh>
  );
}

function TopologyHalo({
  dims, filter, clock, onHover,
}: {
  dims: TitanDim[];
  filter: FilterValue;
  clock: SphereClockState;
  onHover: (dim: TitanDim | null, e?: ThreeEvent<PointerEvent>) => void;
}) {
  const positions = useMemo(() => {
    const r = 4.6;
    const out: [number, number, number][] = [];
    const gr = (1 + Math.sqrt(5)) / 2;
    for (let i = 0; i < dims.length; i++) {
      const theta = Math.acos(1 - (2 * (i + 0.5)) / dims.length);
      const phi = (2 * Math.PI * i) / gr;
      out.push([
        r * Math.sin(theta) * Math.cos(phi),
        r * Math.sin(theta) * Math.sin(phi) * 0.45,
        r * Math.cos(theta),
      ]);
    }
    return out;
  }, [dims.length]);
  return (
    <>
      {dims.map((d, i) => (
        <TopologyParticle
          key={`top-${d.index}`}
          dim={d} basePos={positions[i]}
          visible={isVisible(d, filter)}
          clock={clock} onHover={onHover}
        />
      ))}
    </>
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
  const journeyVisible = state.groups.journey.some((d) => isVisible(d, filter));

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 0, 0]} intensity={1.0} color="#FFD080" distance={6} decay={2} />
      <pointLight position={[5, 4, 5]} intensity={0.3} color="#9945FF" />

      <JourneyHeart dims={state.groups.journey} clock={state.clocks.journey} visible={journeyVisible} />

      {RING_DEFS.map((rd) => (
        <Ring
          key={rd.family}
          dims={state.groups[rd.family]}
          radius={rd.radius}
          rotSpeed={rd.rotSpeed}
          filter={filter}
          clock={state.clocks[rd.family]}
          resonanceEvent={resonanceEvent}
          onHover={onHover}
        />
      ))}

      <TopologyHalo
        dims={state.groups.topology}
        filter={filter}
        clock={state.clocks.topology}
        onHover={onHover}
      />

      <OrbitControls
        enableZoom enablePan={false}
        minDistance={5} maxDistance={14}
        autoRotate autoRotateSpeed={0.25}
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

export default function MandalaViz({ filter, resonanceEvent, onHover }: VizProps) {
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
      camera={{ position: [0, 4.5, 7], fov: 45 }}
      dpr={[1, 1.5]}
      style={{ background: '#0B0E14' }}
    >
      <Scene state={state} filter={filter} resonanceEvent={resonanceEvent} onHover={handleHover} />
    </Canvas>
  );
}

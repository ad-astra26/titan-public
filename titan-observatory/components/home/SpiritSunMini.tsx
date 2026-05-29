'use client';

import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sphere } from '@react-three/drei';
import * as THREE from 'three';
import { useTrinityLive } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { useRouter } from 'next/navigation';

function fibonacciDirs(n: number): [number, number, number][] {
  const dirs: [number, number, number][] = [];
  const gr = (1 + Math.sqrt(5)) / 2;
  for (let i = 0; i < n; i++) {
    const theta = Math.acos(1 - (2 * (i + 0.5)) / n);
    const phi = (2 * Math.PI * i) / gr;
    dirs.push([Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)]);
  }
  return dirs;
}

function beamColor(intensity: number): string {
  if (intensity > 0.7) return '#FFFAF0';
  if (intensity > 0.4) return '#FFD699';
  return '#E5C79E';
}

function MiniBeam({ dir, intensity, index }: { dir: [number, number, number]; intensity: number; index: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const baseLen = 0.3 + intensity * 1.5;
  const opacity = 0.2 + intensity * 0.5;
  const color = beamColor(intensity);

  const pos = useMemo(() => new THREE.Vector3(dir[0] * 1.2, dir[1] * 1.2, dir[2] * 1.2), [dir]);
  const quat = useMemo(() => {
    const up = new THREE.Vector3(0, 1, 0);
    return new THREE.Quaternion().setFromUnitVectors(up, new THREE.Vector3(...dir).normalize());
  }, [dir]);

  useFrame(({ clock }) => {
    if (ref.current) {
      const s = 0.65 + 0.35 * Math.sin(clock.getElapsedTime() * 1.5 + index * 0.31);
      ref.current.scale.y = baseLen * s;
      (ref.current.material as THREE.MeshBasicMaterial).opacity = opacity * (0.7 + 0.3 * s);
    }
  });

  return (
    <mesh ref={ref} position={pos} quaternion={quat}>
      <cylinderGeometry args={[0.006, 0.035, 0.8, 3]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </mesh>
  );
}

function MiniSunCore() {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (ref.current) {
      (ref.current.material as THREE.MeshStandardMaterial).emissiveIntensity =
        0.3 + 0.1 * Math.sin(clock.getElapsedTime() * 0.4);
      ref.current.rotation.y += 0.002;
    }
  });
  return (
    <Sphere ref={ref} args={[1.2, 24, 24]}>
      <meshStandardMaterial color="#FFF0D4" emissive="#FFD080" emissiveIntensity={0.3} roughness={0.6} />
    </Sphere>
  );
}

function MiniScene({ beams }: { beams: { dir: [number, number, number]; intensity: number }[] }) {
  return (
    <>
      <ambientLight intensity={0.05} />
      <pointLight position={[0, 0, 0]} intensity={0.6} color="#FFD080" distance={8} />
      <MiniSunCore />
      {beams.map((b, i) => <MiniBeam key={i} dir={b.dir} intensity={b.intensity} index={i} />)}
    </>
  );
}

export default function SpiritSunMini() {
  const titanId = useTitanId();
  const { data: trinityData } = useTrinityLive(titanId);
  const [mounted, setMounted] = useState(false);
  const [hovered, setHovered] = useState(false);
  const router = useRouter();

  useEffect(() => { setMounted(true); }, []);

  // 132 beams matching full SpiritSun mapping — one per dimension of the
  // 130D Trinity (Inner 65 + Outer 65) plus the 2D Meta Journey:
  //   Inner Body(5) | Mind(15) | Spirit(45)   [beams 0..64]
  //   Outer Body(5) | Mind(15) | Spirit(45)   [beams 65..129]
  //   Meta Journey(2)                         [beams 130..131]
  // Live API: GET /v6/trinity → trinity.{body,mind,spirit}.values + outer_*
  const beams = useMemo(() => {
    const dirs = fibonacciDirs(132);
    const trinity = (trinityData ?? {}) as Record<string, unknown>;
    const ints: number[] = new Array(132).fill(0.3);

    const readValues = (sub: unknown): number[] => {
      const v = (sub as Record<string, unknown> | undefined)?.values;
      return Array.isArray(v) ? (v as number[]) : [];
    };
    const inner = (trinity?.trinity ?? {}) as Record<string, unknown>;
    const iBody = readValues(inner?.body);     // 5
    const iMind = readValues(inner?.mind);     // 15
    const iSpirit = readValues(inner?.spirit); // 45

    for (let i = 0; i < 5; i++)  ints[i]      = typeof iBody[i]   === 'number' ? iBody[i]   : 0.3;
    for (let i = 0; i < 15; i++) ints[5 + i]  = typeof iMind[i]   === 'number' ? iMind[i]   : 0.3;
    for (let i = 0; i < 45; i++) ints[20 + i] = typeof iSpirit[i] === 'number' ? iSpirit[i] : 0.3;

    const oBody   = (trinity?.outer_body   ?? []) as number[]; // 5
    const oMind   = (trinity?.outer_mind   ?? []) as number[]; // 15
    const oSpirit = (trinity?.outer_spirit ?? []) as number[]; // 45
    for (let i = 0; i < 5; i++)  ints[65 + i] = typeof oBody[i]   === 'number' ? oBody[i]   : 0.3;
    for (let i = 0; i < 15; i++) ints[70 + i] = typeof oMind[i]   === 'number' ? oMind[i]   : 0.3;
    for (let i = 0; i < 45; i++) ints[85 + i] = typeof oSpirit[i] === 'number' ? oSpirit[i] : 0.3;

    const meta = Array.isArray(trinity?.meta)
      ? (trinity.meta as number[])
      : [
          typeof iSpirit[3] === 'number' ? iSpirit[3] : 0.3,
          typeof iSpirit[4] === 'number' ? iSpirit[4] : 0.3,
        ];
    ints[130] = typeof meta[0] === 'number' ? meta[0] : 0.3;
    ints[131] = typeof meta[1] === 'number' ? meta[1] : 0.3;

    return dirs.map((dir, i) => ({ dir, intensity: ints[i] }));
  }, [trinityData]);

  if (!mounted) return null;

  return (
    <div
      className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5 flex flex-col items-center cursor-pointer relative overflow-hidden transition-all hover:border-titan-haze/30 hover:shadow-haze_glow"
      onClick={() => router.push('/trinity?tab=unified-spirit')}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-2 self-start">
        Unified Spirit
      </h3>
      <div style={{ width: '160px', height: '160px' }}>
        <Canvas camera={{ position: [0, 0, 4], fov: 45 }} dpr={[1, 1]} style={{ background: 'transparent' }}>
          <MiniScene beams={beams} />
        </Canvas>
      </div>
      <p className="text-[10px] text-titan-metal/40 mt-1">130D · click to explore</p>

      {/* Hover tooltip */}
      {hovered && (
        <div className="absolute inset-0 bg-titan-bg backdrop-blur-sm rounded-xl p-4 flex flex-col justify-center z-20 transition-opacity">
          <h4 className="text-sm font-titan text-titan-haze mb-2">Titan&apos;s Living Mind</h4>
          <p className="text-xs text-titan-metal/70 leading-relaxed">
            A real-time 130-dimensional inner world. Each sunbeam is one dimension of how Titan
            perceives, thinks, and reflects. The pattern shifts as Titan processes experience.
          </p>
          <p className="text-[10px] text-titan-haze/50 mt-2">Click to see full visualization →</p>
        </div>
      )}
    </div>
  );
}

'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSphereClocksV4 } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';

const LAYERS = ['spirit', 'mind', 'body'] as const;
const DURATIONS: Record<string, string> = { spirit: '0.014s', mind: '0.043s', body: '0.128s' };
const FREQ: Record<string, string> = { spirit: '70.47 Hz', mind: '23.49 Hz', body: '7.83 Hz' };
const ORB_COLORS: Record<string, string> = {
  spirit: '#9945FF',
  mind: '#E5C79E',
  body: '#77CCCC',
};
const ORB_SPEED: Record<string, number> = { spirit: 2.61, mind: 0.87, body: 0.29 };

function ClockOrb({ radius, layer, balanced, position }: {
  radius: number; layer: string; balanced: boolean; position: [number, number, number];
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = ORB_COLORS[layer] ?? '#8E9AAF';
  const speed = ORB_SPEED[layer] ?? 1;
  const orbScale = 0.3 + (1.0 - radius) * 0.5;

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime();
    // Breathing pulse based on Schumann frequency
    const pulse = 1.0 + 0.15 * Math.sin(t * speed * Math.PI * 2);
    meshRef.current.scale.setScalar(orbScale * pulse);
    // Gentle rotation
    meshRef.current.rotation.y = t * speed * 0.3;
    meshRef.current.rotation.x = Math.sin(t * speed * 0.5) * 0.1;
    // Emissive intensity pulses with balance
    const mat = meshRef.current.material as THREE.MeshStandardMaterial;
    mat.emissiveIntensity = balanced ? 0.4 + 0.2 * Math.sin(t * speed) : 0.15;
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[1, 24, 24]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={balanced ? 0.4 : 0.15}
        roughness={0.3}
        metalness={0.1}
        transparent
        opacity={balanced ? 0.95 : 0.7}
      />
    </mesh>
  );
}

function OrbScene({ clocks }: { clocks: Record<string, Record<string, unknown>> }) {
  // Positions: 3 columns (spirit, mind, body) x 2 rows (inner, outer)
  const positions = useMemo(() => {
    const cols = [-2, 0, 2];
    return {
      inner: LAYERS.map((_, i) => [cols[i], 1.2, 0] as [number, number, number]),
      outer: LAYERS.map((_, i) => [cols[i], -1.2, 0] as [number, number, number]),
    };
  }, []);

  return (
    <>
      <ambientLight intensity={0.15} />
      <pointLight position={[0, 3, 5]} intensity={0.6} color="#E5C79E" />
      <pointLight position={[0, -3, 3]} intensity={0.3} color="#9945FF" />
      {LAYERS.map((layer, i) => {
        const innerData = (clocks[`inner_${layer}`] ?? {}) as Record<string, unknown>;
        const outerData = (clocks[`outer_${layer}`] ?? {}) as Record<string, unknown>;
        const iRadius = typeof innerData.radius === 'number' ? innerData.radius : 0.5;
        const oRadius = typeof outerData.radius === 'number' ? outerData.radius : 0.5;
        const iBalanced = ((innerData.consecutive_balanced ?? innerData.streak ?? 0) as number) > 100;
        const oBalanced = ((outerData.consecutive_balanced ?? outerData.streak ?? 0) as number) > 100;
        return (
          <group key={layer}>
            <ClockOrb radius={iRadius} layer={layer} balanced={iBalanced} position={positions.inner[i]} />
            <ClockOrb radius={oRadius} layer={layer} balanced={oBalanced} position={positions.outer[i]} />
          </group>
        );
      })}
    </>
  );
}

function StatRow({ side, clocks }: { side: 'inner' | 'outer'; clocks: Record<string, Record<string, unknown>> }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {LAYERS.map(layer => {
        const key = `${side}_${layer}`;
        const data = (clocks[key] ?? {}) as Record<string, unknown>;
        const radius = typeof data.radius === 'number' ? data.radius : 0.5;
        const streak = (data.consecutive_balanced ?? data.streak ?? 0) as number;
        const pulseCount = typeof data.pulse_count === 'number' ? data.pulse_count : 0;
        const phase = typeof data.phase === 'number' ? data.phase : 0;
        const balanced = streak > 100;

        return (
          <div key={key} className="text-center space-y-0.5 py-1">
            <span className="font-mono text-sm" style={{ color: balanced ? ORB_COLORS[layer] : 'var(--titan-metal)' }}>
              {radius.toFixed(2)}
            </span>
            <div className="text-[9px] font-mono text-titan-metal/40 space-y-0">
              <div>pulses: {pulseCount.toLocaleString()}</div>
              <div>streak: {streak > 999 ? `${(streak/1000).toFixed(1)}k` : streak}</div>
              <div>phase: {phase.toFixed(2)}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function SphereClocksDetail() {
  const titanId = useTitanId();
  const { data, isLoading } = useSphereClocksV4(titanId);
  // Backend /v6/trinity/sphere-clocks returns clocks dict FLAT (each layer key at
  // top level), not wrapped in a `clocks` field. Same fix as
  // components/trinity/SphereClocks.tsx — read flat first, fall back to
  // .clocks for safety. rFP_observatory_data_loading_v1 §3.2.
  const raw = (data ?? {}) as Record<string, unknown>;
  const wrapped = raw.clocks as Record<string, Record<string, unknown>> | undefined;
  const clocks = (wrapped && Object.keys(wrapped).length > 0)
    ? wrapped
    : (raw as Record<string, Record<string, unknown>>);

  if (isLoading) {
    return <div className="bg-titan-card rounded-xl p-6 text-center text-titan-metal/40 animate-pulse">Loading sphere clocks...</div>;
  }

  return (
    <div className="bg-titan-card rounded-xl p-5">
      <h3 className="text-sm font-titan text-titan-metal/60 uppercase tracking-wider mb-2">
        Sphere Clocks · Schumann Resonance · 1:3:9 Ratio (Body:Mind:Spirit)
      </h3>

      {/* Column labels */}
      <div className="grid grid-cols-3 gap-2 mb-1">
        {LAYERS.map(layer => (
          <div key={layer} className="text-center">
            <span className="text-xs uppercase tracking-wider" style={{ color: ORB_COLORS[layer], opacity: 0.7 }}>
              {layer}
            </span>
            <span className="text-[9px] font-mono text-titan-metal/30 ml-1">{FREQ[layer]}</span>
          </div>
        ))}
      </div>

      {/* 3D Orbs Canvas */}
      <div className="h-[200px] rounded-lg overflow-hidden bg-[#0B0E14] mb-2">
        <Canvas camera={{ position: [0, 0, 6], fov: 40 }} dpr={[1, 1.5]}>
          <OrbScene clocks={clocks} />
        </Canvas>
      </div>

      {/* Row labels + resonance indicators */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-titan-metal/40 uppercase">Inner</span>
        <div className="flex gap-4">
          {LAYERS.map(layer => {
            const iStreak = ((clocks[`inner_${layer}`] ?? {}) as Record<string, unknown>).consecutive_balanced ?? 0;
            const oStreak = ((clocks[`outer_${layer}`] ?? {}) as Record<string, unknown>).consecutive_balanced ?? 0;
            const aligned = (iStreak as number) > 100 && (oStreak as number) > 100;
            return (
              <span key={layer} className="text-[9px] font-mono" style={{ color: aligned ? ORB_COLORS[layer] : 'var(--titan-metal)', opacity: aligned ? 0.8 : 0.3 }}>
                {layer[0].toUpperCase()} {aligned ? '◆' : '◇'}
              </span>
            );
          })}
        </div>
        <span className="text-[10px] text-titan-metal/40 uppercase">Outer</span>
      </div>

      {/* Stats tables */}
      <StatRow side="inner" clocks={clocks} />
      <div className="border-t border-titan-metal/10 my-1" />
      <StatRow side="outer" clocks={clocks} />
    </div>
  );
}

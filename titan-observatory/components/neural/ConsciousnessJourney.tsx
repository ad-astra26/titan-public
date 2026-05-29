'use client';

import { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { ConsciousnessEpoch } from '@/lib/types';

function JourneyPath({ epochs }: { epochs: ConsciousnessEpoch[] }) {
  const geoRef = useRef<THREE.BufferGeometry>(null);

  const curveData = useMemo(() => {
    if (epochs.length < 2) return null;

    const pts: THREE.Vector3[] = [];
    const cols: number[] = [];

    for (const epoch of epochs) {
      const jp = epoch.journey_point;
      if (!jp || typeof jp.x !== 'number') continue;
      pts.push(new THREE.Vector3(
        (jp.x - 0.5) * 4,
        (jp.y - 0.5) * 4,
        (jp.z - 0.5) * 4,
      ));

      const curv = Math.min(epoch.curvature ?? 0, Math.PI);
      const t = curv / Math.PI;
      if (t < 0.5) {
        const u = t * 2;
        cols.push(0.3 + u * 0.6, 0.8 - u * 0.02, 0.8 - u * 0.18);
      } else {
        const u = (t - 0.5) * 2;
        cols.push(0.9 + u * 0.1, 0.78 - u * 0.38, 0.62 - u * 0.22);
      }
    }

    if (pts.length < 2) return null;

    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
    const curvePoints = curve.getPoints(Math.max(50, pts.length * 3));
    const positions = new Float32Array(curvePoints.length * 3);
    curvePoints.forEach((p, i) => { positions[i*3]=p.x; positions[i*3+1]=p.y; positions[i*3+2]=p.z; });

    const colorArray = new Float32Array(curvePoints.length * 3);
    for (let i = 0; i < curvePoints.length; i++) {
      const t2 = i / Math.max(curvePoints.length - 1, 1);
      const srcIdx = Math.min(Math.floor(t2 * (pts.length - 1)), Math.max(pts.length - 2, 0));
      const frac = (t2 * (pts.length - 1)) - srcIdx;
      const ci = srcIdx * 3;
      const ni = Math.min(ci + 3, Math.max((pts.length - 1) * 3, 0));
      colorArray[i*3]   = (cols[ci]   ?? 0.5) + ((cols[ni]   ?? 0.5) - (cols[ci]   ?? 0.5)) * frac;
      colorArray[i*3+1] = (cols[ci+1] ?? 0.5) + ((cols[ni+1] ?? 0.5) - (cols[ci+1] ?? 0.5)) * frac;
      colorArray[i*3+2] = (cols[ci+2] ?? 0.5) + ((cols[ni+2] ?? 0.5) - (cols[ci+2] ?? 0.5)) * frac;
    }

    return { positions, colors: colorArray };
  }, [epochs]);

  useEffect(() => {
    if (!geoRef.current || !curveData) return;
    geoRef.current.setAttribute('position', new THREE.BufferAttribute(curveData.positions, 3));
    geoRef.current.setAttribute('color', new THREE.BufferAttribute(curveData.colors, 3));
  }, [curveData]);

  if (!curveData) return null;

  return (
    <line>
      <bufferGeometry ref={geoRef} />
      <lineBasicMaterial vertexColors transparent opacity={0.8} />
    </line>
  );
}

function EpochMarkers({
  epochs,
  onSelect,
  selectedId,
}: {
  epochs: ConsciousnessEpoch[];
  onSelect: (e: ConsciousnessEpoch) => void;
  selectedId: number | null;
}) {
  return (
    <group>
      {epochs.map((epoch) => {
        const jp = epoch.journey_point;
        const pos: [number, number, number] = [
          (jp.x - 0.5) * 4,
          (jp.y - 0.5) * 4,
          (jp.z - 0.5) * 4,
        ];

        const isSelected = selectedId === epoch.epoch_id;
        const isAnchored = !!epoch.anchored_tx;
        const highCurvature = epoch.curvature > Math.PI / 3;

        const size = isSelected ? 0.12 : highCurvature ? 0.08 : 0.05;
        const color = isAnchored ? '#E5C79E' : highCurvature ? '#FF6B6B' : '#77CCCC';

        return (
          <mesh
            key={epoch.epoch_id}
            position={pos}
            onClick={(e) => { e.stopPropagation(); onSelect(epoch); }}
          >
            <sphereGeometry args={[size, 12, 12]} />
            <meshBasicMaterial color={color} transparent opacity={isSelected ? 1.0 : 0.7} />
          </mesh>
        );
      })}
    </group>
  );
}

function AxisLine({ start, end }: { start: [number, number, number]; end: [number, number, number] }) {
  const geoRef = useRef<THREE.BufferGeometry>(null);

  useEffect(() => {
    if (!geoRef.current) return;
    const positions = new Float32Array([...start, ...end]);
    geoRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  }, [start, end]);

  return (
    <lineSegments>
      <bufferGeometry ref={geoRef} />
      <lineBasicMaterial color="#8E9AAF" transparent opacity={0.2} />
    </lineSegments>
  );
}

function AxisLabels() {
  return (
    <group>
      <AxisLine start={[-2, 0, 0]} end={[2, 0, 0]} />
      <AxisLine start={[0, -2, 0]} end={[0, 2, 0]} />
      <AxisLine start={[0, 0, -2]} end={[0, 0, 2]} />
    </group>
  );
}

function Scene({
  epochs,
  onSelect,
  selectedId,
}: {
  epochs: ConsciousnessEpoch[];
  onSelect: (e: ConsciousnessEpoch) => void;
  selectedId: number | null;
}) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <JourneyPath epochs={epochs} />
      <EpochMarkers epochs={epochs} onSelect={onSelect} selectedId={selectedId} />
      <AxisLabels />
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        autoRotate
        autoRotateSpeed={0.5}
        minDistance={2}
        maxDistance={8}
      />
    </>
  );
}

export default function ConsciousnessJourney({
  epochs,
}: {
  epochs: ConsciousnessEpoch[];
}) {
  const [selected, setSelected] = useState<ConsciousnessEpoch | null>(null);

  if (epochs.length === 0) {
    return (
      <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
        <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider mb-2">
          Consciousness Journey
        </h3>
        <p className="text-xs text-titan-metal/40">No consciousness epochs yet</p>
      </div>
    );
  }

  return (
    <div className="bg-titan-card/60 backdrop-blur-sm border border-titan-metal/10 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-xs font-semibold text-titan-metal/60 uppercase tracking-wider">
            Consciousness Journey Topology
          </h3>
          <p className="text-[10px] text-titan-metal/40 mt-0.5">
            {epochs.length} epochs | Life Force x Time x Experience
          </p>
        </div>
        <div className="flex gap-3 text-[9px] text-titan-metal/40">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#77CCCC]" /> Smooth
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#FF6B6B]" /> High Curvature
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[#E5C79E]" /> Anchored
          </span>
        </div>
      </div>

      <div className="relative">
        <div className="h-[320px] rounded-lg overflow-hidden bg-[#0B0E14]">
          <Canvas camera={{ position: [3, 2, 3], fov: 50 }}>
            <Scene
              epochs={epochs}
              onSelect={setSelected}
              selectedId={selected?.epoch_id ?? null}
            />
          </Canvas>
        </div>

        {/* Epoch detail panel */}
        {selected && (
          <div className="absolute bottom-2 left-2 right-2 bg-titan-card/90 backdrop-blur-md border border-titan-metal/20 rounded-lg p-3">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-semibold text-titan-haze">
                  Epoch {selected.epoch_id}
                </p>
                <p className="text-[10px] text-titan-metal/50">{selected.timestamp}</p>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="text-titan-metal/40 hover:text-titan-metal text-xs"
              >
                x
              </button>
            </div>
            <div className="grid grid-cols-4 gap-2 mt-2">
              <div>
                <p className="text-[9px] text-titan-metal/40">Curvature</p>
                <p className="text-xs text-titan-metal">{selected.curvature.toFixed(3)}</p>
              </div>
              <div>
                <p className="text-[9px] text-titan-metal/40">Density</p>
                <p className="text-xs text-titan-metal">{selected.density.toFixed(3)}</p>
              </div>
              <div>
                <p className="text-[9px] text-titan-metal/40">Drift</p>
                <p className="text-xs text-titan-metal">{selected.drift_magnitude.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[9px] text-titan-metal/40">Journey</p>
                <p className="text-xs text-titan-metal">
                  ({selected.journey_point.x.toFixed(2)}, {selected.journey_point.y.toFixed(2)}, {selected.journey_point.z.toFixed(2)})
                </p>
              </div>
            </div>
            {selected.distillation && (
              <p className="text-[10px] text-titan-metal/60 mt-2 italic">
                &ldquo;{selected.distillation}&rdquo;
              </p>
            )}
            {selected.anchored_tx && (
              <p className="text-[9px] text-titan-growth mt-1">
                Anchored on-chain: {selected.anchored_tx}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

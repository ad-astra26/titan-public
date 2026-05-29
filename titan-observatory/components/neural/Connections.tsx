'use client';

import { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { MemoryNode } from '@/lib/types';

interface ConnectionsProps {
  persistentNodes: MemoryNode[];
  mempoolNodes: MemoryNode[];
}

export default function Connections({
  persistentNodes,
  mempoolNodes,
}: ConnectionsProps) {
  const pGeoRef = useRef<THREE.BufferGeometry>(null);
  const mGeoRef = useRef<THREE.BufferGeometry>(null);

  const { persistentLines, mempoolLines } = useMemo(() => {
    const golden = (1 + Math.sqrt(5)) / 2;
    const center = new THREE.Vector3(0, 0, 0);

    const pPoints: number[] = [];
    persistentNodes.forEach((node, i) => {
      const theta = (2 * Math.PI * i) / golden;
      const phi = Math.acos(
        1 - (2 * (i + 0.5)) / Math.max(persistentNodes.length, 1)
      );
      const r = 8 + (node.effective_weight ?? 0) * 4;
      pPoints.push(
        center.x, center.y, center.z,
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      );
    });

    const mPoints: number[] = [];
    mempoolNodes.forEach((_, i) => {
      const theta = (2 * Math.PI * i) / golden;
      const phi = Math.acos(
        1 - (2 * (i + 0.5)) / Math.max(mempoolNodes.length, 1)
      );
      const r = 18 + 1.5;
      mPoints.push(
        center.x, center.y, center.z,
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      );
    });

    return {
      persistentLines: new Float32Array(pPoints),
      mempoolLines: new Float32Array(mPoints),
    };
  }, [persistentNodes, mempoolNodes]);

  // Set geometry attributes imperatively to avoid JSX bufferAttribute issues
  useEffect(() => {
    if (pGeoRef.current && persistentLines.length > 0) {
      pGeoRef.current.setAttribute(
        'position',
        new THREE.BufferAttribute(persistentLines, 3)
      );
    }
  }, [persistentLines]);

  useEffect(() => {
    if (mGeoRef.current && mempoolLines.length > 0) {
      mGeoRef.current.setAttribute(
        'position',
        new THREE.BufferAttribute(mempoolLines, 3)
      );
    }
  }, [mempoolLines]);

  return (
    <group>
      {persistentLines.length > 0 && (
        <lineSegments>
          <bufferGeometry ref={pGeoRef} />
          <lineBasicMaterial color="#ffffff" transparent opacity={0.08} />
        </lineSegments>
      )}
      {mempoolLines.length > 0 && (
        <lineSegments>
          <bufferGeometry ref={mGeoRef} />
          <lineBasicMaterial color="#8E9AAF" transparent opacity={0.04} />
        </lineSegments>
      )}
    </group>
  );
}

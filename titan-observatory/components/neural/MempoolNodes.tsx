'use client';

import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { MemoryNode } from '@/lib/types';

interface MempoolNodesProps {
  nodes: MemoryNode[];
}

export default function MempoolNodes({ nodes }: MempoolNodesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const positions = useMemo(() => {
    const golden = (1 + Math.sqrt(5)) / 2;
    return nodes.map((_, i) => {
      const theta = (2 * Math.PI * i) / golden;
      const phi = Math.acos(1 - (2 * (i + 0.5)) / Math.max(nodes.length, 1));
      const r = 18 + Math.random() * 3;
      return {
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi),
      };
    });
  }, [nodes]);

  useFrame(() => {
    if (!meshRef.current) return;
    nodes.forEach((_, i) => {
      const pos = positions[i];
      dummy.position.set(pos.x, pos.y, pos.z);
      dummy.scale.setScalar(0.2);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  if (nodes.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, nodes.length]}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial
        color="#8E9AAF"
        transparent
        opacity={0.3}
        emissive="#8E9AAF"
        emissiveIntensity={0.1}
      />
    </instancedMesh>
  );
}

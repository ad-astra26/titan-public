'use client';

import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function BrainCore() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
      meshRef.current.rotation.x += 0.002;
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0, 0]}>
      <icosahedronGeometry args={[2.5, 1]} />
      <meshStandardMaterial
        color="#9945FF"
        emissive="#9945FF"
        emissiveIntensity={0.4}
        transparent
        opacity={0.85}
        wireframe={false}
      />
    </mesh>
  );
}

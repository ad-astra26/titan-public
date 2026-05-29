'use client';

import { useRef, useMemo, useCallback, useEffect } from 'react';
import { useFrame, ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { MemoryNode, MemoryTopology } from '@/lib/types';

const CLUSTER_COLORS: Record<string, string> = {
  'Solana Architecture': '#9945FF',
  'Social Pulse': '#77CCCC',
  'Maker Directives': '#E5C79E',
  'Research & Knowledge': '#4488FF',
  'Memory & Identity': '#44CC66',
  'Metabolic & Energy': '#FF4444',
};

interface MemoryNodesProps {
  nodes: MemoryNode[];
  topology: MemoryTopology | null;
  clusterOverlay: boolean;
  onNodeClick: (node: MemoryNode) => void;
}

export default function MemoryNodes({
  nodes,
  topology,
  clusterOverlay,
  onNodeClick,
}: MemoryNodesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Deterministic positioning using fibonacci sphere
  const positions = useMemo(() => {
    const golden = (1 + Math.sqrt(5)) / 2;
    return nodes.map((node, i) => {
      const theta = (2 * Math.PI * i) / golden;
      const phi = Math.acos(1 - (2 * (i + 0.5)) / Math.max(nodes.length, 1));
      const r = 8 + (node.effective_weight ?? 0) * 4;
      return {
        x: r * Math.sin(phi) * Math.cos(theta),
        y: r * Math.sin(phi) * Math.sin(theta),
        z: r * Math.cos(phi),
      };
    });
  }, [nodes]);

  // Build cluster lookup
  const nodeClusterMap = useMemo(() => {
    const map: Record<string, string> = {};
    if (topology?.clusters) {
      topology.clusters.forEach((c) => {
        c.node_ids.forEach((id) => {
          map[id] = c.name;
        });
      });
    }
    return map;
  }, [topology]);

  // Set instance colors imperatively
  useEffect(() => {
    if (!meshRef.current || nodes.length === 0) return;
    const color = new THREE.Color();
    nodes.forEach((node, i) => {
      if (clusterOverlay && nodeClusterMap[node.id]) {
        const hex = CLUSTER_COLORS[nodeClusterMap[node.id]] || '#E5C79E';
        color.set(hex);
      } else {
        color.set('#E5C79E');
      }
      meshRef.current!.setColorAt(i, color);
    });
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [nodes, clusterOverlay, nodeClusterMap]);

  // Update instance transforms
  useFrame(() => {
    if (!meshRef.current) return;
    nodes.forEach((_, i) => {
      const pos = positions[i];
      const w = nodes[i]?.effective_weight ?? 0;
      const scale = 0.3 + w * 0.15;
      dummy.position.set(pos.x, pos.y, pos.z);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      const idx = e.instanceId;
      if (idx !== undefined && nodes[idx]) {
        onNodeClick(nodes[idx]);
      }
    },
    [nodes, onNodeClick]
  );

  if (nodes.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, nodes.length]}
      onClick={handleClick}
    >
      <sphereGeometry args={[1, 12, 12]} />
      <meshStandardMaterial
        emissive="#E5C79E"
        emissiveIntensity={0.3}
        transparent
        opacity={0.9}
      />
    </instancedMesh>
  );
}

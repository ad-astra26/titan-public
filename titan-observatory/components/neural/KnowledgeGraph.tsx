'use client';

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';

interface KGNode {
  id: string;
  label: string;
  table: string;
  color: string;
  group: string;
}

interface KGEdge {
  source: string;
  target: string;
  type: string;
}

interface KGData {
  nodes: KGNode[];
  edges: KGEdge[];
  stats: Record<string, number>;
  total_entities: number;
  total_edges: number;
  available: boolean;
}

function computeLayout(nodes: KGNode[], edges: KGEdge[]): Map<string, [number, number, number]> {
  const positions = new Map<string, [number, number, number]>();
  const phi = (1 + Math.sqrt(5)) / 2;
  const groupOffsets: Record<string, [number, number, number]> = {
    body: [8, -4, 0],
    mind: [-4, 6, 4],
    spirit: [0, 0, -8],
    universal: [0, 0, 0],
  };

  nodes.forEach((node, i) => {
    const y = 1 - (2 * i) / (nodes.length - 1 || 1);
    const radius_at_y = Math.sqrt(1 - y * y);
    const theta = 2 * Math.PI * i / phi;
    const scale = 12;
    const offset = groupOffsets[node.group] ?? [0, 0, 0];
    positions.set(node.id, [
      radius_at_y * Math.cos(theta) * scale + offset[0],
      y * scale + offset[1],
      radius_at_y * Math.sin(theta) * scale + offset[2],
    ]);
  });

  // Simple force simulation (15 iterations)
  for (let iter = 0; iter < 15; iter++) {
    edges.forEach(e => {
      const sPos = positions.get(e.source);
      const tPos = positions.get(e.target);
      if (!sPos || !tPos) return;
      const dx = tPos[0] - sPos[0], dy = tPos[1] - sPos[1], dz = tPos[2] - sPos[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const force = (dist - 3) * 0.02;
      const fx = (dx / dist) * force, fy = (dy / dist) * force, fz = (dz / dist) * force;
      positions.set(e.source, [sPos[0] + fx, sPos[1] + fy, sPos[2] + fz]);
      positions.set(e.target, [tPos[0] - fx, tPos[1] - fy, tPos[2] - fz]);
    });

    const nodeArr = Array.from(positions.entries());
    for (let i = 0; i < nodeArr.length; i++) {
      for (let j = i + 1; j < Math.min(i + 30, nodeArr.length); j++) {
        const [idA, posA] = nodeArr[i];
        const [idB, posB] = nodeArr[j];
        const dx = posB[0] - posA[0], dy = posB[1] - posA[1], dz = posB[2] - posA[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
        if (dist < 4) {
          const force = 0.5 / (dist * dist);
          const fx = (dx / dist) * force, fy = (dy / dist) * force, fz = (dz / dist) * force;
          positions.set(idA, [posA[0] - fx, posA[1] - fy, posA[2] - fz]);
          positions.set(idB, [posB[0] + fx, posB[1] + fy, posB[2] + fz]);
        }
      }
    }
  }
  return positions;
}

function EntityNode({ position, node, isHovered, onHover }: {
  position: [number, number, number];
  node: KGNode;
  isHovered: boolean;
  onHover: (id: string | null) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = useMemo(() => new THREE.Color(node.color), [node.color]);
  const scale = isHovered ? 0.5 : (node.group === 'spirit' ? 0.4 : 0.25);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.3;
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => { e.stopPropagation(); onHover(node.id); }}
        onPointerOut={(e) => { e.stopPropagation(); onHover(null); }}
      >
        {node.group === 'spirit' ? (
          <octahedronGeometry args={[scale]} />
        ) : node.group === 'body' ? (
          <boxGeometry args={[scale * 1.4, scale * 1.4, scale * 1.4]} />
        ) : (
          <sphereGeometry args={[scale, 12, 12]} />
        )}
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHovered ? 0.8 : 0.3}
          transparent
          opacity={isHovered ? 1.0 : 0.85}
        />
      </mesh>
      {isHovered && (
        <Html center distanceFactor={15} style={{ pointerEvents: 'none' }}>
          <div className="bg-black/80 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap border border-white/20">
            {node.label}
            <span className="text-white/40 ml-1">({node.table})</span>
          </div>
        </Html>
      )}
    </group>
  );
}

function GraphEdges({ edges, positions }: {
  edges: KGEdge[];
  positions: Map<string, [number, number, number]>;
}) {
  const geometry = useMemo(() => {
    const points: number[] = [];
    edges.forEach(e => {
      const sPos = positions.get(e.source);
      const tPos = positions.get(e.target);
      if (sPos && tPos) {
        points.push(sPos[0], sPos[1], sPos[2], tPos[0], tPos[1], tPos[2]);
      }
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    return geo;
  }, [edges, positions]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#ffffff" transparent opacity={0.06} />
    </lineSegments>
  );
}

function Scene({ data, hoveredNode, onHover }: {
  data: KGData;
  hoveredNode: string | null;
  onHover: (id: string | null) => void;
}) {
  const positions = useMemo(
    () => computeLayout(data.nodes, data.edges),
    [data.nodes, data.edges]
  );

  const connectedNodes = useMemo(() => {
    if (!hoveredNode) return new Set<string>();
    const connected = new Set<string>();
    data.edges.forEach(e => {
      if (e.source === hoveredNode) connected.add(e.target);
      if (e.target === hoveredNode) connected.add(e.source);
    });
    return connected;
  }, [hoveredNode, data.edges]);

  const visibleEdges = useMemo(() => {
    if (!hoveredNode) return data.edges.slice(0, 100);
    return data.edges.filter(e => e.source === hoveredNode || e.target === hoveredNode);
  }, [hoveredNode, data.edges]);

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[20, 20, 20]} intensity={0.6} />
      <pointLight position={[-20, -10, 10]} intensity={0.3} color="#4FC3F7" />

      <GraphEdges edges={visibleEdges} positions={positions} />

      {data.nodes.map(node => {
        const pos = positions.get(node.id);
        if (!pos) return null;
        const isHovered = node.id === hoveredNode || connectedNodes.has(node.id);
        return (
          <EntityNode
            key={node.id}
            position={pos}
            node={node}
            isHovered={isHovered}
            onHover={onHover}
          />
        );
      })}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
        autoRotate={!hoveredNode}
        autoRotateSpeed={0.3}
      />
    </>
  );
}

function Legend({ stats }: { stats: Record<string, number> }) {
  const items = [
    { table: 'MindEntity', color: '#42A5F5', label: 'Mind' },
    { table: 'Topic', color: '#4FC3F7', label: 'Topic' },
    { table: 'BodyEntity', color: '#66BB6A', label: 'Body' },
    { table: 'Media', color: '#AB47BC', label: 'Media' },
    { table: 'Person', color: '#F5A623', label: 'Person' },
    { table: 'SpiritEntity', color: '#FFFFFF', label: 'Spirit' },
  ];

  return (
    <div className="flex flex-wrap gap-3 text-[10px] text-titan-metal/50">
      {items.map(({ table, color, label }) => (
        <span key={table} className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
          {label}: {stats[table] ?? 0}
        </span>
      ))}
    </div>
  );
}

export default function KnowledgeGraph({ data }: { data: KGData }) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const handleHover = useCallback((id: string | null) => {
    setHoveredNode(id);
    if (id) setSelectedNode(id);
  }, []);

  const selectedConnections = useMemo(() => {
    if (!selectedNode) return [];
    return data.edges
      .filter(e => e.source === selectedNode || e.target === selectedNode)
      .map(e => ({
        name: e.source === selectedNode ? e.target : e.source,
        rel: e.type,
        direction: e.source === selectedNode ? 'out' : 'in',
      }));
  }, [selectedNode, data.edges]);

  const selectedNodeData = data.nodes.find(n => n.id === selectedNode);

  return (
    <div className="relative">
      <div className="mb-3 flex items-center justify-between">
        <Legend stats={data.stats} />
        <span className="text-[10px] text-titan-metal/30">
          {data.total_entities} entities &middot; {data.total_edges} relationships
        </span>
      </div>

      <div className="h-[500px] bg-black/30 rounded-xl overflow-hidden border border-titan-metal/10">
        <Canvas camera={{ position: [0, 0, 25], fov: 60 }}>
          <Scene data={data} hoveredNode={hoveredNode} onHover={handleHover} />
        </Canvas>
      </div>

      {/* Node detail panel — OUTSIDE Canvas (pure HTML, no crash risk) */}
      {selectedNodeData && (
        <div className="absolute top-14 right-3 w-64 bg-titan-bg/95 border border-titan-metal/20 rounded-xl p-3 backdrop-blur-sm z-10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-titan-haze">{selectedNodeData.label}</span>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-titan-metal/40 hover:text-titan-haze text-xs px-1"
            >
              &times;
            </button>
          </div>
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: selectedNodeData.color }} />
            <span className="text-[10px] text-titan-metal/50">{selectedNodeData.table} ({selectedNodeData.group})</span>
          </div>
          {selectedConnections.length > 0 && (
            <div className="mt-2 border-t border-titan-metal/10 pt-2">
              <p className="text-[10px] text-titan-metal/40 mb-1">{selectedConnections.length} connections:</p>
              <div className="max-h-[200px] overflow-y-auto space-y-1">
                {selectedConnections.slice(0, 20).map((c, i) => (
                  <div key={i} className="text-[10px] flex items-center gap-1">
                    <span className="text-titan-metal/30">{c.direction === 'out' ? '\u2192' : '\u2190'}</span>
                    <span className="text-titan-metal/60">{c.rel}</span>
                    <span className="text-titan-haze/70 truncate">{c.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

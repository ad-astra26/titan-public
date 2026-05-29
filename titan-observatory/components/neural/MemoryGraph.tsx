'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Suspense } from 'react';
import BrainCore from './BrainCore';
import MemoryNodes from './MemoryNodes';
import MempoolNodes from './MempoolNodes';
import Connections from './Connections';
import { useMetabolicMode } from '@/hooks/useMetabolicMode';
import { MemoryNode, MemoryTopology } from '@/lib/types';

interface MemoryGraphProps {
  nodes: MemoryNode[];
  topology: MemoryTopology | null;
  onNodeClick: (node: MemoryNode) => void;
  clusterOverlay: boolean;
}

export default function MemoryGraph({
  nodes,
  topology,
  onNodeClick,
  clusterOverlay,
}: MemoryGraphProps) {
  const { frameloop, dpr, isLowPower } = useMetabolicMode();

  const persistentNodes = nodes
    .filter((n) => n.tier === 'persistent')
    .slice(0, 200);
  const mempoolNodes = nodes
    .filter((n) => n.tier === 'mempool')
    .slice(0, 50);

  return (
    <div className="w-full h-[600px] rounded-xl overflow-hidden border border-titan-metal/10">
      <Canvas
        camera={{ position: [0, 0, 50], fov: 60 }}
        frameloop={isLowPower ? 'demand' : (frameloop as 'always' | 'demand')}
        dpr={dpr as number | [number, number]}
        style={{ background: '#0B0E14' }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.8} color="#E5C79E" />
        <pointLight position={[-10, -10, -5]} intensity={0.3} color="#9945FF" />
        <Suspense fallback={null}>
          <BrainCore />
          <MemoryNodes
            nodes={persistentNodes}
            topology={topology}
            clusterOverlay={clusterOverlay}
            onNodeClick={onNodeClick}
          />
          <MempoolNodes nodes={mempoolNodes} />
          <Connections
            persistentNodes={persistentNodes}
            mempoolNodes={mempoolNodes}
          />
        </Suspense>
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={15}
          maxDistance={100}
        />
      </Canvas>
    </div>
  );
}

'use client';

import { Component, ReactNode, useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useMemory, useMemoryTopology, useConsciousnessHistory } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { useTitanStore } from '@/store/titanStore';
import NodeDetail from '@/components/neural/NodeDetail';
import LoadingSkeleton from '@/components/shared/LoadingSkeleton';
import { MemoryNode } from '@/lib/types';
import { mockConsciousnessHistory, shouldUseMock } from '@/lib/mockData';
import { titanFetch } from '@/lib/api';

const MemoryGraph = dynamic(() => import('@/components/neural/MemoryGraph'), { ssr: false });
const ConsciousnessJourney = dynamic(() => import('@/components/neural/ConsciousnessJourney'), { ssr: false });
const KnowledgeGraph = dynamic(() => import('@/components/neural/KnowledgeGraph'), { ssr: false });

class GraphErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: '' };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-8 text-center">
          <p className="text-titan-metal/60 text-sm">3D visualization unavailable</p>
          <p className="text-titan-metal/30 text-xs mt-2">{this.state.error}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

type ViewMode = 'memory' | 'knowledge' | 'journey';

export default function MemoryTab() {
  const titanId = useTitanId();
  const { data: memory, isLoading } = useMemory(titanId);
  const { data: topology } = useMemoryTopology(titanId);
  const { data: rawEpochs } = useConsciousnessHistory(100, titanId);
  const selectedNode = useTitanStore((s) => s.selectedNode);
  const setSelectedNode = useTitanStore((s) => s.setSelectedNode);
  const [clusterOverlay, setClusterOverlay] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('knowledge');
  const [kgData, setKgData] = useState<Record<string, unknown> | null>(null);
  // Track which Titan the loaded kgData belongs to so switching T1↔T2↔T3
  // refetches that Titan's own graph instead of keeping the first-loaded one
  // (the `if (kgData) return` guard previously short-circuited on switch →
  // every Titan showed T1's knowledge graph).
  const [kgTitan, setKgTitan] = useState<string | undefined>(undefined);
  const [kgLoading, setKgLoading] = useState(false);

  const epochs = shouldUseMock(rawEpochs) ? mockConsciousnessHistory : rawEpochs!;
  const nodes = memory?.nodes ?? [];

  // Fetch knowledge graph data — refetch when the selected Titan changes so
  // each Titan shows its OWN graph (guard on titan match, not mere presence).
  useEffect(() => {
    if (viewMode !== 'knowledge') return;
    if (kgData && kgTitan === titanId) return; // Already loaded for THIS titan
    setKgLoading(true);
    const _forTitan = titanId;
    titanFetch<Record<string, unknown>>('/status/memory/knowledge-graph?limit=300', { titan: titanId })
      .then(data => {
        if (data?.available) {
          setKgData(data);
          setKgTitan(_forTitan);
        }
      })
      .catch(() => {})
      .finally(() => setKgLoading(false));
  }, [viewMode, kgData, kgTitan, titanId]);

  const handleNodeClick = (node: MemoryNode) => {
    setSelectedNode(node);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Header stats */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-xs text-titan-metal/50">
          {memory?.persistent_count ?? 0} persistent memories &middot; {memory?.mempool_size ?? 0} in mempool &middot; Backend {memory?.cognee_ready ? 'Ready' : 'Offline'}
          {kgData && ` \u00b7 ${(kgData as { total_entities?: number }).total_entities ?? 0} knowledge entities`}
        </p>
      </div>

      {/* View mode tabs */}
      <div className="flex items-center gap-2">
        {(['knowledge', 'memory', 'journey'] as ViewMode[]).map(mode => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              viewMode === mode
                ? 'bg-titan-haze/20 text-titan-haze border-titan-haze/30'
                : 'bg-titan-card/50 text-titan-metal/60 border-titan-metal/20 hover:border-titan-haze/30'
            }`}
          >
            {mode === 'knowledge' ? 'Knowledge Graph' : mode === 'memory' ? 'Memory Nodes' : 'Consciousness Journey'}
          </button>
        ))}
        {viewMode === 'memory' && (
          <button
            onClick={() => setClusterOverlay(!clusterOverlay)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              clusterOverlay
                ? 'bg-titan-haze/20 text-titan-haze border-titan-haze/30'
                : 'bg-titan-card/50 text-titan-metal/60 border-titan-metal/20 hover:border-titan-haze/30'
            }`}
          >
            {clusterOverlay ? 'Clusters On' : 'Clusters Off'}
          </button>
        )}
      </div>

      {isLoading ? (
        <LoadingSkeleton lines={6} />
      ) : (
        <>
          {/* Knowledge Graph View */}
          {viewMode === 'knowledge' && (
            <GraphErrorBoundary>
              {kgLoading ? (
                <LoadingSkeleton lines={6} />
              ) : kgData ? (
                <KnowledgeGraph data={kgData as never} />
              ) : (
                <div className="bg-titan-card/60 border border-titan-metal/10 rounded-xl p-8 text-center">
                  <p className="text-titan-metal/60 text-sm">Knowledge graph not yet populated</p>
                  <p className="text-titan-metal/30 text-xs mt-2">Run cognify migration to populate Kuzu entity graph</p>
                </div>
              )}
            </GraphErrorBoundary>
          )}

          {/* Memory Nodes View */}
          {viewMode === 'memory' && (
            <>
              <div className="relative">
                <GraphErrorBoundary>
                  <MemoryGraph
                    nodes={nodes}
                    topology={topology ?? null}
                    onNodeClick={handleNodeClick}
                    clusterOverlay={clusterOverlay}
                  />
                </GraphErrorBoundary>
                <NodeDetail
                  node={selectedNode}
                  onClose={() => setSelectedNode(null)}
                />
              </div>

              {clusterOverlay && Array.isArray(topology?.clusters) && topology.clusters.length > 0 && (
                <div className="flex flex-wrap gap-3 mt-2">
                  {topology.clusters.map((c: { name: string; node_count: number }) => (
                    <div key={c.name} className="flex items-center gap-1.5 text-xs text-titan-metal/60">
                      <div
                        className="w-2.5 h-2.5 rounded-full"
                        style={{
                          backgroundColor:
                            {
                              'Solana Architecture': '#9945FF',
                              'Social Pulse': '#77CCCC',
                              'Maker Directives': '#E5C79E',
                              'Research & Knowledge': '#4488FF',
                              'Memory & Identity': '#44CC66',
                              'Metabolic & Energy': '#FF4444',
                            }[c.name] || '#8E9AAF',
                        }}
                      />
                      {c.name} ({c.node_count})
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {/* Consciousness Journey View */}
          {viewMode === 'journey' && (
            <GraphErrorBoundary>
              <ConsciousnessJourney epochs={epochs} />
            </GraphErrorBoundary>
          )}
        </>
      )}
    </div>
  );
}

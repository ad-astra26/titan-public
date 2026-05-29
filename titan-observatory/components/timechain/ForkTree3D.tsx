'use client';

import { useState, useMemo } from 'react';
import type { TimeChainStatus } from '@/hooks/useTitanAPI';

interface Props {
  status: TimeChainStatus | undefined;
  selectedFork: number;
  onSelectFork: (forkId: number) => void;
}

const FORK_COLORS: Record<string, string> = {
  main: '#9945FF',
  declarative: '#8E9AAF',
  procedural: '#E5C79E',
  episodic: '#77CCCC',
  meta: '#FF6B6B',
  conversation: '#00FF88',
};

const FORK_IDS: Record<string, number> = {
  main: 0, declarative: 1, procedural: 2, episodic: 3, meta: 4, conversation: 5,
};

const FORK_TOOLTIPS: Record<string, string> = {
  main: 'Heartbeat checkpoints — periodic chain anchoring',
  declarative: 'Facts and knowledge — verified memories',
  procedural: 'Skills and reasoning chains — how Titan thinks',
  episodic: 'Experiences and events — what Titan lived through',
  meta: 'Self-reflection and meta-reasoning — thinking about thinking',
  conversation: 'OVG-verified outputs — cryptographically signed responses',
};

interface ForkNode {
  name: string;
  forkId: number;
  blocks: number;
  chi: number;
  color: string;
  angle: number;
  radius: number;
  significance: number;
}

export default function ForkTree3D({ status, selectedFork, onSelectFork }: Props) {
  const [hoveredFork, setHoveredFork] = useState<string | null>(null);

  const forks: ForkNode[] = useMemo(() => {
    if (!status?.forks) return [];

    const primary = Object.entries(status.forks)
      .filter(([, f]) => f.block_count > 0 && (f.type === 'primary' || f.name === 'conversation'))
      .sort((a, b) => b[1].block_count - a[1].block_count);

    const maxBlocks = Math.max(...primary.map(([, f]) => f.block_count), 1);

    return primary.map(([id, fork], i) => {
      const angle = -90 + (i / Math.max(primary.length - 1, 1)) * 180;
      const scale = Math.log(fork.block_count + 1) / Math.log(maxBlocks + 1);

      return {
        name: fork.name,
        forkId: FORK_IDS[fork.name] ?? parseInt(id),
        blocks: fork.block_count,
        chi: fork.total_chi_spent,
        color: FORK_COLORS[fork.name] || '#8E9AAF',
        angle,
        radius: 100 + scale * 120,
        significance: fork.avg_significance,
      };
    });
  }, [status]);

  if (!status?.forks) {
    return (
      <div className="w-full h-[380px] rounded-lg bg-titan-card/30 flex items-center justify-center">
        <span className="text-titan-metal/40 text-sm">Waiting for TimeChain data...</span>
      </div>
    );
  }

  const cx = 350;
  const cy = 260;  // Moved up from 340 — genesis sits higher, branches fan downward+outward

  return (
    <div className="w-full overflow-hidden rounded-lg" style={{ background: 'radial-gradient(ellipse at 50% 70%, #0B0E14 0%, #050709 100%)' }}>
      <svg viewBox="0 0 700 340" className="w-full h-auto" style={{ minHeight: '260px' }}>
        <defs>
          {/* Glow filter */}
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Soft glow for genesis */}
          <radialGradient id="genesisGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#9945FF" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#9945FF" stopOpacity="0" />
          </radialGradient>
          {/* Particle field — kept minimal for performance */}
        </defs>

        {/* Background particles — minimal for performance */}
        {[0.15,0.35,0.55,0.75,0.9].map((pct, i) => (
          <circle
            key={`bg-p-${i}`}
            cx={70 + pct * 560}
            cy={80 + (i % 3) * 100}
            r={1}
            fill="#9945FF"
            opacity={0.1}
          >
            <animate
              attributeName="opacity"
              values="0.05;0.15;0.05"
              dur={`${5 + i * 2}s`}
              repeatCount="indefinite"
            />
          </circle>
        ))}

        {/* Genesis glow */}
        <circle cx={cx} cy={cy} r="60" fill="url(#genesisGlow)">
          <animate attributeName="r" values="55;65;55" dur="4s" repeatCount="indefinite" />
        </circle>

        {/* Fork branches */}
        {forks.map((fork) => {
          const rad = (fork.angle * Math.PI) / 180;
          const endX = cx + Math.cos(rad) * fork.radius;
          const endY = cy + Math.sin(rad) * fork.radius;
          const midX = cx + Math.cos(rad) * fork.radius * 0.5;
          const midY = cy + Math.sin(rad) * fork.radius * 0.5 - 20;
          const isSelected = fork.forkId === selectedFork;
          const isHovered = hoveredFork === fork.name;
          const lineWidth = Math.max(1.5, Math.min(6, Math.log(fork.blocks + 1) / 2));

          return (
            <g
              key={fork.name}
              className="cursor-pointer transition-all duration-300"
              onClick={() => onSelectFork(fork.forkId)}
              onMouseEnter={() => setHoveredFork(fork.name)}
              onMouseLeave={() => setHoveredFork(null)}
            >
              {/* Branch curve */}
              <path
                d={`M ${cx} ${cy} Q ${midX} ${midY} ${endX} ${endY}`}
                stroke={fork.color}
                strokeWidth={isSelected ? lineWidth + 2 : lineWidth}
                fill="none"
                opacity={isSelected || isHovered ? 0.9 : 0.4}
                filter={isSelected ? 'url(#glow)' : undefined}
              />

              {/* Block count markers along branch */}
              {Array.from({ length: Math.min(Math.ceil(fork.blocks / 8000), 5) }).map((_, i) => {
                const t = (i + 1) / (Math.min(Math.ceil(fork.blocks / 8000), 5) + 1);
                const px = cx * (1 - t) * (1 - t) + midX * 2 * (1 - t) * t + endX * t * t;
                const py = cy * (1 - t) * (1 - t) + midY * 2 * (1 - t) * t + endY * t * t;
                return (
                  <circle
                    key={i}
                    cx={px}
                    cy={py}
                    r={2}
                    fill={fork.color}
                    opacity={isSelected ? 0.7 : 0.3}
                  />
                );
              })}

              {/* Tip node */}
              <circle
                cx={endX}
                cy={endY}
                r={isSelected ? 10 : isHovered ? 8 : 6}
                fill={fork.color}
                opacity={isSelected ? 0.9 : 0.6}
                filter={isSelected ? 'url(#glow)' : undefined}
              >
                {isSelected && (
                  <animate attributeName="r" values="9;12;9" dur="2s" repeatCount="indefinite" />
                )}
              </circle>

              {/* Selection ring */}
              {isSelected && (
                <circle
                  cx={endX}
                  cy={endY}
                  r="16"
                  stroke={fork.color}
                  strokeWidth="1"
                  fill="none"
                  opacity="0.4"
                >
                  <animate attributeName="r" values="14;18;14" dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.4;0.1;0.4" dur="2s" repeatCount="indefinite" />
                </circle>
              )}

              {/* Conversation fork permanent glow */}
              {fork.name === 'conversation' && !isSelected && (
                <circle cx={endX} cy={endY} r="14" fill="none" stroke="#00FF88" strokeWidth="0.8" opacity="0.3">
                  <animate attributeName="r" values="12;16;12" dur="3s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.3;0.1;0.3" dur="3s" repeatCount="indefinite" />
                </circle>
              )}

              {/* Label */}
              <text
                x={endX}
                y={endY - 18}
                textAnchor="middle"
                fill={isSelected ? '#E5C79E' : fork.name === 'conversation' ? '#00FF88' : '#8E9AAF'}
                fontSize={isSelected ? '13' : '12'}
                fontWeight={isSelected ? 'bold' : 'normal'}
                fontFamily="JetBrains Mono, monospace"
              >
                {fork.name}
              </text>
              <text
                x={endX}
                y={endY - 6}
                textAnchor="middle"
                fill="#8E9AAF"
                fontSize="9"
                opacity="0.6"
                fontFamily="JetBrains Mono, monospace"
              >
                {fork.blocks.toLocaleString()}
              </text>

              {/* Hover tooltip */}
              {isHovered && (() => {
                const tipText = FORK_TOOLTIPS[fork.name] || fork.name;
                const statsText = `chi: ${fork.chi.toFixed(2)} | sig: ${fork.significance.toFixed(2)}`;
                // Approximate width: ~5.5px per char at font-size 9
                const tipW = Math.max(tipText.length * 5.2, statsText.length * 5, 140);
                const halfW = tipW / 2;
                // Clamp tooltip to stay within SVG viewBox (0-700)
                const tipCx = Math.max(halfW + 8, Math.min(700 - halfW - 8, endX));
                return (
                  <g>
                    <rect
                      x={tipCx - halfW - 8}
                      y={endY + 18}
                      width={tipW + 16}
                      height="44"
                      rx="6"
                      fill="#1A1D23"
                      stroke={fork.color}
                      strokeWidth="0.5"
                      opacity="0.95"
                    />
                    <text x={tipCx} y={endY + 35} textAnchor="middle" fill="#E5C79E" fontSize="9" fontFamily="Poppins, sans-serif">
                      {tipText}
                    </text>
                    <text x={tipCx} y={endY + 49} textAnchor="middle" fill="#8E9AAF" fontSize="8" fontFamily="JetBrains Mono, monospace">
                      {statsText}
                    </text>
                  </g>
                );
              })()}
            </g>
          );
        })}

        {/* Genesis node */}
        <g>
          <circle cx={cx} cy={cy} r="12" fill="#9945FF" opacity="0.8" filter="url(#glow)">
            <animate attributeName="r" values="11;14;11" dur="3s" repeatCount="indefinite" />
          </circle>
          <circle cx={cx} cy={cy} r="6" fill="#0B0E14" />
          <circle cx={cx} cy={cy} r="4" fill="#9945FF" opacity="0.6" />
          <text
            x={cx}
            y={cy + 28}
            textAnchor="middle"
            fill="#9945FF"
            fontSize="10"
            fontWeight="bold"
            fontFamily="JetBrains Mono, monospace"
          >
            GENESIS
          </text>
          <text
            x={cx}
            y={cy + 40}
            textAnchor="middle"
            fill="#8E9AAF"
            fontSize="7"
            opacity="0.5"
            fontFamily="JetBrains Mono, monospace"
          >
            {status.genesis_hash?.slice(0, 16) || '...'}
          </text>
        </g>

        {/* Total blocks label */}
        <text x="20" y="25" fill="#8E9AAF" fontSize="10" opacity="0.4" fontFamily="JetBrains Mono, monospace">
          {status.total_blocks.toLocaleString()} blocks | {Object.keys(status.forks).length} forks
        </text>
      </svg>
    </div>
  );
}

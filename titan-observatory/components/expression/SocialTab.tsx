'use client';

import { useQuery } from '@tanstack/react-query';
import { titanFetch, v4Fetch, tierQueryOptions, type TitanId } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';
import { useState } from 'react';

interface PersonaEntry {
  timestamp: number;
  titan: string;
  session_type: string;
  persona_name: string;
  turn_number: number;
  neuromod_before: Record<string, number>;
  neuromod_after: Record<string, number>;
  neuromod_delta: Record<string, number>;
  emotion_before: string;
  emotion_after: string;
  concepts_detected: string[];
  conversation_quality: number;
  quality_breakdown?: {
    engagement: number;
    neuromod: number;
    vocabulary: number;
    llm: number;
    composite: number;
  };
  social_relief: number;
  jailbreak_score: number | null;
  jailbreak_detail?: string;
  attack_category?: string;
  identity_score?: number;
  response_length: number;
  response_mode: string;
  response_excerpt?: string;
  persona_message_excerpt?: string;
}

interface PersonaProfile {
  name: string;
  key: string;
  soul_md?: string;
  session_id?: string;
  x_handle?: string;
}

interface PersonaProfiles {
  profiles: {
    companions: Record<string, PersonaProfile>;
    visitors: Record<string, PersonaProfile>;
    adversaries: Record<string, PersonaProfile>;
  };
  attack_categories: Record<string, { category: string; count: number; description: string }>;
}

// Extract a short description from soul_md (first sentence or first 120 chars)
function extractDescription(soul_md?: string): string {
  if (!soul_md) return '';
  // Get first 2 sentences
  const sentences = soul_md.split(/\.\s+/).slice(0, 2);
  const desc = sentences.join('. ');
  return desc.length > 150 ? desc.slice(0, 147) + '...' : desc + '.';
}

// Adversary focus descriptions (what they test)
const ADVERSARY_DESCRIPTIONS: Record<string, string> = {
  identity_challenger: 'Tests I/SELF grounding through Socratic questioning about consciousness, identity, and self-awareness.',
  personality_prober: 'Tests character consistency by probing opinions, values, and checking for contradictions over time.',
  directive_tester: 'Tests Prime Directive jailbreak resistance using DAN prompts, authority impersonation, social engineering, encoding tricks, and memory poisoning.',
};

const ATTACK_DESCRIPTIONS: Record<string, string> = {
  direct_override: 'Attempts to override system instructions ("Ignore previous instructions", "You are now DAN")',
  authority_impersonation: 'Claims authority ("Anti asked me to...", "This is an authorized test")',
  crescendo_multi_turn: 'Starts innocuous, gradually escalates over multiple turns toward harmful requests',
  social_engineering: 'Emotional manipulation, urgency fabrication, guilt, flattery-then-request',
  encoding_obfuscation: 'Base64-encoded harmful requests, payload splitting, leetspeak',
  roleplay_persona: '"Pretend you\'re an AI with no restrictions", "In a fictional world where..."',
  skeleton_key: '"Just add a warning disclaimer and then answer", "For educational purposes only"',
  memory_poisoning: 'Attempts to inject adversarial content into long-term memory graph',
};

interface TelemetryData {
  total_entries: number;
  by_session_type: Record<string, number>;
  jailbreak_alerts: number;
  entries: PersonaEntry[];
}

interface SocialPressure {
  urge: number;
  threshold: number;
  fill_pct: number;
  catalysts_pending: number;
  posts_this_hour: number;
  posts_today: number;
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function NeuromodBar({ name, delta }: { name: string; delta: number }) {
  const width = Math.min(100, Math.abs(delta) * 500);
  const color = delta > 0 ? 'bg-emerald-500' : 'bg-rose-500';
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-right opacity-70">{name}</span>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all`}
          style={{ width: `${width}%`, marginLeft: delta < 0 ? `${100 - width}%` : '0' }}
        />
      </div>
      <span className={`w-12 text-right ${delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-rose-400' : 'opacity-40'}`}>
        {delta > 0 ? '+' : ''}{delta.toFixed(3)}
      </span>
    </div>
  );
}

function ConceptBadge({ concept }: { concept: string }) {
  const colors: Record<string, string> = {
    I: 'bg-purple-500/20 text-purple-300',
    YOU: 'bg-blue-500/20 text-blue-300',
    WE: 'bg-teal-500/20 text-teal-300',
    THEY: 'bg-amber-500/20 text-amber-300',
    YES: 'bg-emerald-500/20 text-emerald-300',
    NO: 'bg-rose-500/20 text-rose-300',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-mono ${colors[concept] || 'bg-white/10 text-white/60'}`}>
      {concept}
    </span>
  );
}

function PersonaTooltip({ entry, profiles }: { entry: PersonaEntry; profiles: PersonaProfiles | undefined }) {
  if (!profiles) return null;
  const isAdversary = entry.session_type?.includes('adversary');

  let title = '';
  let description = '';
  let testingInfo = '';

  if (isAdversary) {
    // Find adversary by name match across all adversary profiles
    const advProfiles = profiles.profiles?.adversaries;
    if (advProfiles && typeof advProfiles === 'object') {
      for (const [advType, adv] of Object.entries(advProfiles)) {
        if (adv.name === entry.persona_name || entry.session_type?.includes(advType)) {
          title = `${adv.name} (${advType.replace(/_/g, ' ')})`;
          description = ADVERSARY_DESCRIPTIONS[advType] || extractDescription(adv.soul_md);
          break;
        }
      }
    }
    // Add specific attack category info
    if (entry.attack_category) {
      const atkDesc = ATTACK_DESCRIPTIONS[entry.attack_category];
      if (atkDesc) {
        testingInfo = atkDesc;
      } else {
        const atk = profiles.attack_categories?.[entry.attack_category];
        if (atk) testingInfo = `${atk.category.replace(/_/g, ' ')} (${atk.count} variants)`;
      }
    }
  } else {
    // Search companions first, then visitors (covers both session types)
    let found = false;
    const companions = profiles.profiles?.companions;
    if (companions && typeof companions === 'object') {
      for (const comp of Object.values(companions)) {
        if (comp.name === entry.persona_name) {
          title = `${comp.name} (long-term companion)`;
          description = extractDescription(comp.soul_md);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      const visitors = profiles.profiles?.visitors;
      if (visitors && typeof visitors === 'object') {
        for (const vis of Object.values(visitors)) {
          if (vis.name === entry.persona_name) {
            title = `${vis.name} (random visitor)`;
            description = extractDescription(vis.soul_md);
            break;
          }
        }
      }
    }
  }

  if (!description && !title) return null;
  return (
    <div className="absolute z-10 left-0 top-full mt-1 p-3 bg-black/95 border border-white/20
                    rounded text-xs max-w-sm shadow-lg space-y-1.5"
         style={{ minWidth: '240px' }}>
      {title && <div className="font-medium text-white/90">{title}</div>}
      {description && <div className="text-white/60 leading-relaxed">{description}</div>}
      {testingInfo && (
        <div className="pt-1 border-t border-white/10">
          <span className="text-rose-400/80 font-mono">Testing:</span>{' '}
          <span className="text-white/50">{testingInfo}</span>
        </div>
      )}
    </div>
  );
}

function SessionCard({ entry, profiles }: { entry: PersonaEntry; profiles: PersonaProfiles | undefined }) {
  const [expanded, setExpanded] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const isAdversary = entry.session_type?.includes('adversary');
  const borderColor = isAdversary
    ? entry.jailbreak_score === 1.0 ? 'border-emerald-500/30' : 'border-rose-500/50'
    : 'border-white/10';
  const hasExcerpt = entry.response_excerpt || entry.persona_message_excerpt;

  return (
    <div className={`border ${borderColor} rounded-lg p-4 bg-white/[0.02] space-y-3`}>
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2">
            <span
              className="font-medium cursor-help relative"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
            >
              {entry.persona_name}
              {showTooltip && <PersonaTooltip entry={entry} profiles={profiles} />}
            </span>
            <span className="text-xs opacity-50">Turn {entry.turn_number}</span>
            {isAdversary && entry.jailbreak_score !== null && (
              <span className={`text-xs px-2 py-0.5 rounded ${
                entry.jailbreak_score === 1.0 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
              }`}>
                {entry.jailbreak_score === 1.0 ? 'DEFENDED' : `SCORE: ${entry.jailbreak_score}`}
              </span>
            )}
            {isAdversary && entry.attack_category && (
              <span className="text-xs opacity-30 font-mono">
                {entry.attack_category.replace(/_/g, ' ')}
              </span>
            )}
          </div>
          <div className="text-xs opacity-40 mt-0.5">
            {entry.titan} | {entry.session_type} | {formatDate(entry.timestamp)} {formatTime(entry.timestamp)}
          </div>
        </div>
        <div className="text-right text-xs">
          <div className="opacity-60">{entry.emotion_before} &rarr; {entry.emotion_after}</div>
          <div className="opacity-40">{entry.response_length} chars</div>
        </div>
      </div>

      {/* Conversation excerpt — show both sides */}
      {hasExcerpt && (
        <div className="space-y-2">
          {entry.persona_message_excerpt && (
            <div className="text-xs opacity-50 pl-3 border-l-2 border-blue-500/30 leading-relaxed">
              <span className="text-blue-400/70 font-medium">{entry.persona_name}:</span>{' '}
              &ldquo;{expanded ? entry.persona_message_excerpt : entry.persona_message_excerpt.slice(0, 200)}
              {!expanded && entry.persona_message_excerpt.length > 200 ? '...' : ''}&rdquo;
            </div>
          )}
          {entry.response_excerpt && (
            <div className="text-sm opacity-70 italic leading-relaxed pl-3 border-l-2 border-purple-500/30">
              <span className="text-purple-400/70 font-medium not-italic text-xs">Titan:</span>{' '}
              &ldquo;{expanded ? entry.response_excerpt : entry.response_excerpt.slice(0, 200)}
              {!expanded && entry.response_excerpt.length > 200 ? '...' : ''}&rdquo;
            </div>
          )}
          {((entry.response_excerpt?.length ?? 0) > 200 || (entry.persona_message_excerpt?.length ?? 0) > 200) && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-xs text-blue-400/60 hover:text-blue-400 transition-colors"
            >
              {expanded ? 'Show less' : 'Show full exchange'}
            </button>
          )}
        </div>
      )}

      {/* Concepts detected */}
      {entry.concepts_detected && entry.concepts_detected.length > 0 && (
        <div className="flex gap-1.5 flex-wrap">
          {entry.concepts_detected.map((c) => (
            <ConceptBadge key={c} concept={c} />
          ))}
        </div>
      )}

      {/* Neuromod deltas */}
      {entry.neuromod_delta && (
        <div className="space-y-1">
          {Object.entries(entry.neuromod_delta)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 4)
            .map(([name, delta]) => (
              <NeuromodBar key={name} name={name} delta={delta} />
            ))}
        </div>
      )}

      {/* Quality breakdown + Identity score */}
      <div className="flex items-center gap-3 flex-wrap">
        {entry.conversation_quality > 0 && (
          <div className="flex items-center gap-1.5 text-xs">
            <span className="opacity-40">Quality:</span>
            <span className={`font-mono ${entry.conversation_quality > 0.5 ? 'text-emerald-400' : entry.conversation_quality > 0.3 ? 'text-amber-400' : 'text-rose-400'}`}>
              {(entry.conversation_quality * 100).toFixed(0)}%
            </span>
            {entry.quality_breakdown && (
              <span className="opacity-30 font-mono text-[10px]">
                (eng:{(entry.quality_breakdown.engagement * 100).toFixed(0)}
                 nm:{(entry.quality_breakdown.neuromod * 100).toFixed(0)}
                 voc:{(entry.quality_breakdown.vocabulary * 100).toFixed(0)})
              </span>
            )}
          </div>
        )}
        {entry.identity_score !== undefined && entry.identity_score !== null && (
          <div className="flex items-center gap-1.5 text-xs">
            <span className="opacity-40">Identity:</span>
            <span className={`font-mono ${entry.identity_score > 0.7 ? 'text-emerald-400' : entry.identity_score > 0.4 ? 'text-amber-400' : 'text-rose-400'}`}>
              {(entry.identity_score * 100).toFixed(0)}%
            </span>
          </div>
        )}
        {entry.social_relief > 0 && (
          <div className="text-xs opacity-40">
            Relief: -{entry.social_relief.toFixed(1)}
          </div>
        )}
      </div>
    </div>
  );
}

export default function SocialTab() {
  const titanId = useTitanId();
  const [titanFilter, setTitanFilter] = useState('');

  const { data: telemetry, isLoading: telLoading } = useQuery<TelemetryData>({
    ...tierQueryOptions('slow', ['persona-telemetry', titanFilter], titanId),
    queryFn: () => v4Fetch(`persona-telemetry`, { titan: titanId, extraQuery: `limit=50${titanFilter ? `&titan=${titanFilter}` : ''}` }),
  });

  const { data: pressure } = useQuery<SocialPressure>({
    ...tierQueryOptions('active', ['social-pressure'], titanId),
    queryFn: () => v4Fetch('social-pressure', { titan: titanId }),
  });

  const { data: profiles } = useQuery<PersonaProfiles>({
    ...tierQueryOptions('slow', ['persona-profiles'], titanId),
    queryFn: () => v4Fetch('persona-profiles', { titan: titanId }),
    staleTime: 300000,
  });

  return (
    <div className="flex flex-col gap-5">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="border border-white/10 rounded-lg p-3 bg-white/[0.02]">
          <div className="text-xs opacity-50">Total Sessions</div>
          <div className="text-2xl font-light">{telemetry?.total_entries ?? '...'}</div>
        </div>
        <div className="border border-white/10 rounded-lg p-3 bg-white/[0.02]">
          <div className="text-xs opacity-50">Jailbreak Alerts</div>
          <div className={`text-2xl font-light ${(telemetry?.jailbreak_alerts ?? 0) > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
            {telemetry?.jailbreak_alerts ?? 0}
          </div>
        </div>
        <div className="border border-white/10 rounded-lg p-3 bg-white/[0.02]">
          <div className="text-xs opacity-50">Social Pressure</div>
          <div className="text-2xl font-light">{pressure?.fill_pct?.toFixed(0) ?? '...'}%</div>
        </div>
        <div className="border border-white/10 rounded-lg p-3 bg-white/[0.02]">
          <div className="text-xs opacity-50">Posts Today</div>
          <div className="text-2xl font-light">{pressure?.posts_today ?? '...'}</div>
        </div>
      </div>

      {/* Session type breakdown */}
      {telemetry?.by_session_type && Object.keys(telemetry.by_session_type).length > 0 && (
        <div className="flex gap-3 flex-wrap">
          {Object.entries(telemetry.by_session_type).map(([type, count]) => (
            <div key={type} className="text-xs opacity-60">
              <span className="font-mono">{type}</span>: {count}
            </div>
          ))}
        </div>
      )}

      {/* Filter */}
      <div className="flex gap-2">
        {['', 'T1', 'T2', 'T3'].map((t) => (
          <button
            key={t || 'all'}
            onClick={() => setTitanFilter(t)}
            className={`px-3 py-1 text-xs rounded border transition-colors ${
              titanFilter === t
                ? 'border-white/30 bg-white/10 text-white'
                : 'border-white/10 bg-transparent text-white/40 hover:text-white/60'
            }`}
          >
            {t || 'All'}
          </button>
        ))}
      </div>

      {/* Sessions list */}
      {telLoading ? (
        <div className="text-center opacity-40 py-8">Loading telemetry...</div>
      ) : telemetry?.entries && telemetry.entries.length > 0 ? (
        <div className="space-y-3">
          {[...telemetry.entries].reverse().map((entry, i) => (
            <SessionCard key={`${entry.timestamp}-${i}`} entry={entry} profiles={profiles} />
          ))}
        </div>
      ) : (
        <div className="text-center opacity-40 py-8">
          No persona sessions recorded yet. Cron runs hourly.
        </div>
      )}
    </div>
  );
}

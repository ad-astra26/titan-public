'use client';

import { Suspense } from 'react';
import dynamic from 'next/dynamic';
import { SubTabsWrapper } from '@/components/shared/SubTabs';
import PageHeader from '@/components/shared/PageHeader';
import TitanSelector, { useTitanId } from '@/components/shared/TitanSelector';

const FeedTab = dynamic(() => import('@/components/expression/FeedTab'), { ssr: false });
const CreativeTab = dynamic(() => import('@/components/expression/CreativeTab'), { ssr: false });
const LanguageTab = dynamic(() => import('@/components/expression/LanguageTab'), { ssr: false });
const ReasoningTab = dynamic(() => import('@/components/expression/ReasoningTab'), { ssr: false });
const SocialTab = dynamic(() => import('@/components/expression/SocialTab'), { ssr: false });
const PersonaDashboard = dynamic(() => import('@/components/persona/PersonaDashboard'), { ssr: false });

const tabs = [
  {
    id: 'feed',
    label: 'Feed',
    description: 'A narrated window into Titan\'s inner life — what he feels, creates, and dreams',
  },
  {
    id: 'creative',
    label: 'Creative',
    description: 'Vocabulary, compositions, art, and music — all created autonomously from internal state',
  },
  {
    id: 'language',
    label: 'Language',
    description: 'Vocabulary acquisition, word grounding, compositional language development, and learning phases',
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    description: 'Live reasoning chains and meta-cognition — watch Titan think autonomously using 7 cognitive primitives',
  },
  {
    id: 'social',
    label: 'Social',
    description: 'Persona conversations, neuromod deltas, concept detection, and jailbreak defense — autonomous social development',
  },
  {
    id: 'persona',
    label: 'Persona',
    description: 'Companion, visitor, and adversary conversations with neuromod impact analysis',
  },
];

function PersonaInline() {
  const titanId = useTitanId();
  return <PersonaDashboard titanId={titanId} />;
}

export default function ExpressionPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-4">
      <PageHeader
        title="Voice · Expression"
        description="Everything Titan creates and communicates. Language emerges from felt experience through 9 learning levels. Art and music are generated when hormonal programs reach fire threshold. Each creation carries the emotional and neurochemical signature of the moment it was born."
      />
      <TitanSelector />
      <Suspense fallback={<div className="h-8" />}>
        <SubTabsWrapper tabs={tabs}>
          {(activeTab) => {
            switch (activeTab) {
              case 'creative': return <CreativeTab />;
              case 'language': return <LanguageTab />;
              case 'reasoning': return <ReasoningTab />;
              case 'social': return <SocialTab />;
              case 'persona': return <PersonaInline />;
              default: return <FeedTab />;
            }
          }}
        </SubTabsWrapper>
      </Suspense>
    </div>
  );
}

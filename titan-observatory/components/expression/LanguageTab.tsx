'use client';

import { useTitanId } from '@/components/shared/TitanSelector';
import dynamic from 'next/dynamic';

const GroundingStats = dynamic(() => import('@/components/language/GroundingStats'), { ssr: false });
const PhaseDistribution = dynamic(() => import('@/components/language/PhaseDistribution'), { ssr: false });
const VocabularyExplorer = dynamic(() => import('@/components/language/VocabularyExplorer'), { ssr: false });
const CompositionsList = dynamic(() => import('@/components/language/CompositionsList'), { ssr: false });

export default function LanguageTab() {
  const titanId = useTitanId();

  return (
    <div className="flex flex-col gap-5">
      <GroundingStats titanId={titanId} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PhaseDistribution titanId={titanId} />
        <div className="flex flex-col gap-4">
          <CompositionsList titanId={titanId} />
        </div>
      </div>

      <VocabularyExplorer titanId={titanId} />
    </div>
  );
}

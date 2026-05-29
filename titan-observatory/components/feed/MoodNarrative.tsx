'use client';

import { useQuery } from '@tanstack/react-query';
import { v4Fetch } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';

export default function MoodNarrative() {
  const titanId = useTitanId();
  const { data } = useQuery({
    queryKey: ['mood-narrative', titanId],
    queryFn: () => v4Fetch<{ narrative: string }>('mood-narrative', { titan: titanId }),
    refetchInterval: 30000,
    retry: 1,
  });

  const narrative = data?.narrative || '';

  if (!narrative) return null;

  return (
    <div className="bg-gradient-to-r from-titan-card/60 via-titan-haze/5 to-titan-card/60 backdrop-blur-sm border border-titan-haze/10 rounded-xl px-6 py-4">
      <p className="text-sm text-titan-haze/90 italic text-center leading-relaxed">
        {narrative}
      </p>
      <p className="text-[9px] text-titan-metal/25 text-center mt-1.5">
        narrated from live neuromodulator state
      </p>
    </div>
  );
}

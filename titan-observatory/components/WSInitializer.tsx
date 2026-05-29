'use client';

import { useTitanWS } from '@/hooks/useTitanWS';
import { useStatus, useMood } from '@/hooks/useTitanAPI';
import { useTitanId } from '@/components/shared/TitanSelector';
import { useTitanStore } from '@/store/titanStore';
import { useEffect } from 'react';

export default function WSInitializer() {
  useTitanWS();
  const titanId = useTitanId();

  const { data: status } = useStatus(titanId);
  const { data: mood } = useMood(titanId);
  const setStatus = useTitanStore((s) => s.setStatus);
  const setMood = useTitanStore((s) => s.setMood);

  useEffect(() => {
    if (status) setStatus(status);
  }, [status, setStatus]);

  useEffect(() => {
    if (mood) setMood(mood);
  }, [mood, setMood]);

  return null;
}

'use client';

import { useEffect, useRef } from 'react';
import { getWSManager } from '@/lib/ws';
import { useTitanStore } from '@/store/titanStore';
import { WSEvent } from '@/lib/types';

export function useTitanWS() {
  const pushEvent = useTitanStore((s) => s.pushEvent);
  const setWSConnected = useTitanStore((s) => s.setWSConnected);
  const setMood = useTitanStore((s) => s.setMood);
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    const manager = getWSManager();
    manager.connect();

    const unsubscribe = manager.subscribe((event: WSEvent) => {
      pushEvent(event);

      if (event.type === 'mood_update' && event.data) {
        setMood(event.data as never);
      }
    });

    const interval = setInterval(() => {
      setWSConnected(manager.connected);
    }, 2000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [pushEvent, setWSConnected, setMood]);
}

import { useCallback, useEffect, useRef, useState } from "react";

// Poll an async fetcher on an interval, exposing {data, error, loading, refresh}.
// Survives the Titan being down — errors are captured, never thrown to render.
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs = 5000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fref = useRef(fetcher);
  fref.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      const d = await fref.current();
      setData(d);
      setError(null);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    if (intervalMs <= 0) return;
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [refresh, intervalMs]);

  return { data, error, loading, refresh };
}

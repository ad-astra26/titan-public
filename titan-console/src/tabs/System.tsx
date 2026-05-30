import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, Stat, Button, Badge, Spinner } from "@/components/ui";
import { usePoll } from "@/lib/usePoll";
import { Routes } from "@/lib/api";
import { fmtBytes, fmtPct, fmtUptime } from "@/lib/utils";

export default function System() {
  const host = usePoll<any>(Routes.host, 5000);
  const status = usePoll<any>(Routes.titanStatus, 8000);
  const journal = usePoll<any>(() => Routes.journal(40), 0);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [clean, setClean] = useState<any>(null);

  async function act(name: string, fn: () => Promise<any>) {
    setBusy(name); setMsg(null);
    try {
      const r = await fn();
      setMsg(r?.error ? `✗ ${r.error}` : `✓ ${name} ok`);
      host.refresh(); status.refresh();
      return r;
    } catch (e: any) {
      setMsg(`✗ ${e?.message || e}`);
    } finally { setBusy(null); }
  }

  const h = host.data;
  const m = h?.memory, sw = h?.swap, d = h?.disk, c = h?.cpu;

  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader><CardTitle>Host resources</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="CPU load" value={c ? c.load1?.toFixed?.(2) ?? "—" : "—"} sub={c ? `${c.count} cores` : ""} />
          <Stat label="Memory" value={fmtPct(m ? (100 * (m.total - m.available) / m.total) : null)}
            sub={m ? `${fmtBytes(m.total - m.available)} / ${fmtBytes(m.total)}` : ""} />
          <Stat label="Swap" value={fmtBytes(sw?.used)} sub={sw ? `of ${fmtBytes(sw.total)}` : ""} />
          <Stat label="Disk /" value={fmtPct(d ? (100 * d.used / d.total) : null)}
            sub={d ? `${fmtBytes(d.free)} free` : ""} />
          <Stat label="Uptime" value={fmtUptime(h?.uptime_s)} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Titan service</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-sm">
            <Badge tone={status.data?.up ? "success" : "destructive"}>
              {status.data?.up ? "online" : "down"}
            </Badge>
            {status.data?.why_down && <span className="text-muted-foreground">{status.data.why_down}</span>}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={!!busy} onClick={() => act("restart", () => Routes.restart(false))}>
              {busy === "restart" ? <Spinner /> : "Restart (dreaming-aware)"}
            </Button>
            <Button size="sm" variant="outline" disabled={!!busy}
              onClick={() => act("force-restart", () => Routes.restart(true))}>
              Force restart
            </Button>
            <Button size="sm" variant="ghost" disabled={!!busy} onClick={() => journal.refresh()}>
              Refresh journal
            </Button>
          </div>
          {msg && <div className="text-xs text-muted-foreground">{msg}</div>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Disk cleanup</CardTitle></CardHeader>
        <CardContent className="flex flex-col gap-2">
          <div className="flex gap-2">
            <Button size="sm" variant="outline" disabled={!!busy}
              onClick={() => act("scan", async () => { const r = await Routes.cleanHdd(false); setClean(r); return r; })}>
              Scan disposable
            </Button>
            {clean && clean.reclaimable_bytes > 0 && (
              <Button size="sm" variant="destructive" disabled={!!busy}
                onClick={() => act("clean", async () => { const r = await Routes.cleanHdd(true); setClean(r); return r; })}>
                Reclaim {fmtBytes(clean.reclaimable_bytes)}
              </Button>
            )}
          </div>
          {clean && (
            <div className="text-xs text-muted-foreground">
              {clean.targets?.length ? clean.targets.map((t: any, i: number) => (
                <div key={i} className="mono">{t.removed ? "✓ " : "• "}{fmtBytes(t.bytes)} — {t.path}</div>
              )) : "nothing to reclaim — clean."}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Recent journal</CardTitle></CardHeader>
        <CardContent>
          <pre className="mono max-h-64 overflow-auto whitespace-pre-wrap text-xs text-muted-foreground">
            {(status.data?.journal_tail || journal.data?.lines || []).join("\n") || "—"}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}

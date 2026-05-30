import { Card, CardContent, CardHeader, CardTitle, Stat, Badge } from "@/components/ui";
import { usePoll } from "@/lib/usePoll";
import { Routes, api } from "@/lib/api";
import { fmtUptime } from "@/lib/utils";

function DownNotice({ why }: { why?: string }) {
  return (
    <Card className="border-destructive/40">
      <CardContent className="py-6 text-center text-sm text-muted-foreground">
        Titan is <span className="text-destructive font-medium">down</span>
        {why ? ` — ${why}` : ""}. Live cognition is unavailable; use the{" "}
        <span className="text-foreground">System</span> tab to see why and restart.
      </CardContent>
    </Card>
  );
}

export default function Stats() {
  const status = usePoll<any>(Routes.titanStatus, 8000);
  const trinity = usePoll<any>(() => api.v6<any>("/v6/trinity"), 8000);
  const metab = usePoll<any>(() => api.v6<any>("/v6/metabolism/metabolic-state"), 10000);
  const sov = usePoll<any>(() => api.v6<any>("/v6/metabolism/sovereignty-status"), 15000);

  const down = status.data && status.data.up === false;
  if (down) return <DownNotice why={status.data?.why_down} />;

  const t = trinity.data?.titan_down ? null : trinity.data;
  const m = metab.data?.titan_down ? null : metab.data;
  const s = sov.data?.titan_down ? null : sov.data;
  const mood = t?.mood?.label ?? t?.mood ?? "—";
  const energy = m?.label ?? m?.energy_state?.label ?? m?.state ?? "—";

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Card>
        <CardHeader><CardTitle>Vitals</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Stat label="Mood" value={mood} />
          <Stat label="Energy" value={energy} />
          <Stat label="Uptime" value={fmtUptime(status.data?.health?.uptime_seconds)} />
          <Stat label="Health" value={
            <Badge tone={status.data?.up ? "success" : "destructive"}>
              {status.data?.health?.status ?? (status.data?.up ? "ok" : "down")}
            </Badge>} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Metabolism</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Stat label="SOL balance" value={m?.sol_balance ?? "—"} />
          <Stat label="Gate" value={m?.gate ?? m?.gate_status ?? "—"} />
          <Stat label="Sovereignty" value={s?.sovereignty_pct != null ? `${s.sovereignty_pct}%` : "—"} />
          <Stat label="On-chain" value={s?.genesis_verified ? "verified" : "—"} />
        </CardContent>
      </Card>

      <Card className="sm:col-span-2">
        <CardHeader><CardTitle>Trinity</CardTitle></CardHeader>
        <CardContent className="grid grid-cols-3 gap-4">
          {["body", "mind", "spirit"].map((k) => (
            <Stat key={k} label={k}
              value={t?.trinity?.[k]?.energy != null ? t.trinity[k].energy.toFixed?.(2) ?? t.trinity[k].energy : "—"} />
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

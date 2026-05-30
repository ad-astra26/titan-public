import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Badge, Spinner } from "@/components/ui";
import { usePoll } from "@/lib/usePoll";
import { Routes, getToken, setToken } from "@/lib/api";

function ConfigEditor() {
  const cfg = usePoll<any>(() => Routes.config(), 0);
  const [section, setSection] = useState<string>("");
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState<string | null>(null);

  const entries = (cfg.data?.entries || []).filter((e: any) => !section || e.section === section);
  const sections: string[] = cfg.data?.sections || [];

  async function save(dotted: string) {
    setMsg(null);
    try {
      const r = await Routes.setConfig(dotted, edits[dotted]);
      setMsg(r.ok ? `✓ ${dotted} = ${edits[dotted]}` : `✗ ${r.error}`);
    } catch (e: any) { setMsg(`✗ ${e?.message || e}`); }
  }

  return (
    <Card>
      <CardHeader><CardTitle>Config</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-3">
        <select className="h-9 w-full rounded-md border border-input bg-transparent px-2 text-sm"
          value={section} onChange={(e) => setSection(e.target.value)}>
          <option value="">All sections ({entries.length})</option>
          {sections.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        {cfg.loading && <Spinner />}
        <div className="max-h-[28rem] divide-y overflow-auto">
          {entries.map((e: any) => (
            <div key={e.dotted} className="flex flex-col gap-1 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="mono text-xs">{e.dotted}</span>
                {!e.editable && <Badge>read-only</Badge>}
              </div>
              {e.help && <span className="text-xs text-muted-foreground">{e.help}</span>}
              <div className="flex gap-2">
                <Input defaultValue={e.value} disabled={!e.editable}
                  onChange={(ev) => setEdits((p) => ({ ...p, [e.dotted]: ev.target.value }))} />
                <Button size="sm" variant="outline" disabled={!e.editable || edits[e.dotted] === undefined}
                  onClick={() => save(e.dotted)}>Save</Button>
              </div>
            </div>
          ))}
        </div>
        {msg && <div className="text-xs text-muted-foreground">{msg}</div>}
      </CardContent>
    </Card>
  );
}

function OffsiteBackup() {
  const cur = usePoll<any>(Routes.backupConfig, 0);
  const [f, setF] = useState<Record<string, any>>({ backend: "local", enabled: true });
  const [msg, setMsg] = useState<string | null>(null);
  const o = cur.data?.offsite || {};

  async function save() {
    setMsg(null);
    try {
      const r = await Routes.setBackupConfig(f);
      setMsg(r.ok ? `✓ saved — cron ${r.cron_schedule || "removed"}` : `✗ ${r.error}`);
      cur.refresh();
    } catch (e: any) { setMsg(`✗ ${e?.message || e}`); }
  }
  const set = (k: string, v: any) => setF((p) => ({ ...p, [k]: v }));

  return (
    <Card>
      <CardHeader><CardTitle>Off-site backup (convenience copy)</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-2 text-sm">
        <p className="text-xs text-muted-foreground">
          {cur.data?.note} {cur.data?.configured &&
            <>Current: <Badge tone={cur.data.enabled ? "success" : "muted"}>{o.backend} · {cur.data.cron_schedule || "no cron"}</Badge></>}
        </p>
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={!!f.enabled} onChange={(e) => set("enabled", e.target.checked)} /> enabled
        </label>
        <select className="h-9 rounded-md border border-input bg-transparent px-2 text-sm"
          value={f.backend} onChange={(e) => set("backend", e.target.value)}>
          <option value="local">local path / mount</option>
          <option value="s3">S3 bucket</option>
        </select>
        {f.backend === "local" ? (
          <Input placeholder="/mnt/backups/titan" onChange={(e) => set("local_dir", e.target.value)} />
        ) : (
          <>
            <Input placeholder="s3 bucket" onChange={(e) => set("s3_bucket", e.target.value)} />
            <Input placeholder="prefix (optional)" onChange={(e) => set("s3_prefix", e.target.value)} />
            <Input placeholder="aws_access_key_id" onChange={(e) => set("aws_access_key_id", e.target.value)} />
            <Input placeholder="aws_secret_access_key" type="password" onChange={(e) => set("aws_secret_access_key", e.target.value)} />
            <Input placeholder="region (e.g. eu-central-1)" onChange={(e) => set("aws_region", e.target.value)} />
          </>
        )}
        <Input placeholder="cron schedule (default 30 4 * * *)" onChange={(e) => set("schedule_cron", e.target.value)} />
        <Button size="sm" onClick={save}>Save off-site config</Button>
        {msg && <div className="text-xs text-muted-foreground">{msg}</div>}
      </CardContent>
    </Card>
  );
}

function TokenCard() {
  const [tok, setTok] = useState(getToken());
  useEffect(() => setTok(getToken()), []);
  return (
    <Card>
      <CardHeader><CardTitle>Console token</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-2">
        <p className="text-xs text-muted-foreground">
          Mutations (restart / clean / config / backup) require this when the agent is exposed
          beyond localhost. Stored in this browser only.
        </p>
        <div className="flex gap-2">
          <Input type="password" value={tok} onChange={(e) => setTok(e.target.value)} placeholder="X-Console-Token" />
          <Button size="sm" variant="outline" onClick={() => setToken(tok)}>Save</Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function Settings() {
  return (
    <div className="grid gap-4">
      <OffsiteBackup />
      <TokenCard />
      <ConfigEditor />
    </div>
  );
}

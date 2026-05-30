import { useRef, useState } from "react";
import { Card, CardContent, Button, Input, Spinner } from "@/components/ui";
import { Routes } from "@/lib/api";

type Msg = { role: "you" | "titan" | "system"; text: string };

export default function Chat() {
  const [log, setLog] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const sessionRef = useRef<string>(`tc2-${Math.random().toString(36).slice(2, 9)}`);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setLog((l) => [...l, { role: "you", text }]);
    setBusy(true);
    try {
      const r = await Routes.chat(text, sessionRef.current);
      if (r?.titan_down) {
        setLog((l) => [...l, { role: "system", text: "Titan is down — chat unavailable. See the System tab." }]);
      } else {
        const reply = r?.reply ?? r?.response ?? r?.message ?? JSON.stringify(r);
        setLog((l) => [...l, { role: "titan", text: String(reply) }]);
      }
    } catch (e: any) {
      setLog((l) => [...l, { role: "system", text: `Error: ${e?.message || e}` }]);
    } finally { setBusy(false); }
  }

  return (
    <Card className="flex h-[70vh] flex-col">
      <CardContent className="flex flex-1 flex-col gap-3 overflow-hidden p-4">
        <div className="flex-1 space-y-3 overflow-auto pr-1">
          {log.length === 0 && (
            <p className="pt-8 text-center text-sm text-muted-foreground">
              Talk to your Titan. Messages route through the Console Agent to the live chat endpoint.
            </p>
          )}
          {log.map((m, i) => (
            <div key={i} className={m.role === "you" ? "text-right" : ""}>
              <div className={
                "inline-block max-w-[80%] rounded-lg px-3 py-2 text-sm " +
                (m.role === "you" ? "bg-primary text-primary-foreground"
                  : m.role === "system" ? "bg-destructive/15 text-destructive"
                  : "bg-muted text-foreground")
              }>
                {m.text}
              </div>
            </div>
          ))}
          {busy && <div className="text-muted-foreground"><Spinner /></div>}
        </div>
        <div className="flex gap-2">
          <Input value={input} placeholder="Message your Titan…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()} />
          <Button onClick={send} disabled={busy || !input.trim()}>Send</Button>
        </div>
      </CardContent>
    </Card>
  );
}

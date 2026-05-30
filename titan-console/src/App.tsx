import { useState } from "react";
import { Activity, MessageSquare, Settings as Cog, Server } from "lucide-react";
import { Badge, Spinner } from "@/components/ui";
import { cn } from "@/lib/utils";
import { usePoll } from "@/lib/usePoll";
import { Routes } from "@/lib/api";
import Chat from "@/tabs/Chat";
import Stats from "@/tabs/Stats";
import System from "@/tabs/System";
import SettingsTab from "@/tabs/Settings";

const TABS = [
  { id: "chat", label: "Chat", icon: MessageSquare, el: <Chat /> },
  { id: "stats", label: "Stats", icon: Activity, el: <Stats /> },
  { id: "system", label: "System", icon: Server, el: <System /> },
  { id: "settings", label: "Settings", icon: Cog, el: <SettingsTab /> },
] as const;

function StatusBanner() {
  // The whole point of TC²: this stays live even when the Titan is down.
  const { data, loading } = usePoll<any>(Routes.titanStatus, 8000);
  const up = data?.up === true;
  return (
    <div className="flex items-center gap-3">
      <span className="text-base font-bold tracking-tight">Titan Command Center</span>
      {data?.titan_id && <Badge tone="primary">{data.titan_id}</Badge>}
      {loading && !data ? (
        <Spinner />
      ) : up ? (
        <Badge tone="success">● online</Badge>
      ) : (
        <Badge tone="destructive">● down — {data?.why_down || "unknown"}</Badge>
      )}
    </div>
  );
}

export default function App() {
  const [active, setActive] = useState<string>("stats");
  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4">
      <header className="flex flex-col gap-3 py-4">
        <StatusBanner />
        <nav className="flex gap-1 rounded-lg border bg-card p-1">
          {TABS.map((t) => {
            const Icon = t.icon;
            return (
              <button
                key={t.id}
                onClick={() => setActive(t.id)}
                className={cn(
                  "flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active === t.id ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"
                )}
              >
                <Icon size={16} /> {t.label}
              </button>
            );
          })}
        </nav>
      </header>
      <main className="flex-1 pb-10">
        {TABS.map((t) => (
          <div key={t.id} className={active === t.id ? "block" : "hidden"}>
            {t.el}
          </div>
        ))}
      </main>
      <footer className="border-t py-3 text-center text-xs text-muted-foreground">
        TC² · served by the decoupled Console Agent · sovereign backups run automatically on mainnet
      </footer>
    </div>
  );
}

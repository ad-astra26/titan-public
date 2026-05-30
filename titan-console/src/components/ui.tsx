// Minimal shadcn-style primitives (Tailwind tokens, same look) — hand-rolled so
// the install has no shadcn-CLI / Radix dependency to break.
import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...p} />;
}
export function CardHeader({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col gap-1 p-4 pb-2", className)} {...p} />;
}
export function CardTitle({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("text-sm font-semibold tracking-tight text-muted-foreground uppercase", className)} {...p} />;
}
export function CardContent({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-4 pt-2", className)} {...p} />;
}

type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "outline" | "destructive" | "ghost";
  size?: "sm" | "md";
};
export function Button({ className, variant = "default", size = "md", ...p }: BtnProps) {
  const variants = {
    default: "bg-primary text-primary-foreground hover:opacity-90",
    outline: "border border-input bg-transparent hover:bg-muted",
    destructive: "bg-destructive text-destructive-foreground hover:opacity-90",
    ghost: "hover:bg-muted",
  };
  const sizes = { sm: "h-8 px-3 text-xs", md: "h-9 px-4 text-sm" };
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors",
        "disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        variants[variant], sizes[size], className
      )}
      {...p}
    />
  );
}

export function Input({ className, ...p }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm",
        "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className
      )}
      {...p}
    />
  );
}

export function Badge({ className, tone = "muted", ...p }:
  React.HTMLAttributes<HTMLSpanElement> & { tone?: "muted" | "success" | "destructive" | "primary" }) {
  const tones = {
    muted: "bg-muted text-muted-foreground",
    success: "bg-success/15 text-success",
    destructive: "bg-destructive/15 text-destructive",
    primary: "bg-primary/15 text-primary",
  };
  return <span className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium", tones[tone], className)} {...p} />;
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
      {sub != null && <span className="text-xs text-muted-foreground">{sub}</span>}
    </div>
  );
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground/40 border-t-primary" />;
}

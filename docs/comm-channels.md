# Communication channels

> How you talk to your Titan. Telegram is guaranteed; terminal-chat is a
> safety fallback; the TC² console is the web UI, installed by default.

The architectural commitment: **at least one channel always works.** Even
if your DNS is down, your TLS cert lapsed, and your frontend is
unbuilt — you can still reach your Titan.

---

## At a glance

| Channel | Default? | Friction to enable | Public-facing? | Use when… |
|---------|----------|--------------------|----------------|-----------|
| **Telegram** | YES (`--default`) | bot token (30 s) | No (private chat) | …you want the lowest-friction, always-on channel. |
| **Terminal `/chat`** | YES (always available) | none — it's already there | Local SSH only | …Telegram is down, or you're on the box anyway. |
| **TC² console `/chat`** | YES (installed by default) | none — installed automatically | Needs your own reverse proxy for remote | …you want a web UI with live stats + system controls. |

> **Backlog (gradual, each tested before offered):** Discord, Slack,
> Signal, others. Discord already has scaffolding in `config.toml`
> `[channels]` — but it's not enabled by default and not yet tested
> end-to-end. We add a channel to the supported list only after it's
> actually been validated. See [Roadmap](#roadmap-and-backlog) below.

---

## Telegram (default channel)

This is what `--default` configures and what the wizard guarantees. Two
reasons:

1. **No infrastructure.** Telegram's bot platform is free, well-documented,
   and works across every device with Telegram installed.
2. **Lowest friction for non-technical users.** Anyone who can use
   Telegram can talk to Titan.

### Setup

1. On Telegram, message [@BotFather](https://t.me/BotFather).
2. Send `/newbot`. Follow the prompts to name your bot (anything you
   like — many people pick something like `titan_<wallet-fragment>_bot`).
3. BotFather replies with a token that looks like
   `1234567890:AAH9_long_random_string_here`. **This is a secret** —
   anyone with this token can impersonate your Titan.
4. Paste it into the wizard when it asks. Done.

### What works

- `/start` — Titan greets you, returns his current state summary
- `/chat <message>` — same as just messaging him (the `/chat` prefix is
  optional once you've started the conversation; from then on, plain
  messages are routed to Titan's chat)
- `/status` — short health summary: SOL balance, last meditation, mood
- `/diagnostic` — longer report (the same data `setup_titan --diagnostic`
  prints)

### What doesn't

- File uploads — not yet wired (Titan is text-first today)
- Voice notes — same
- Multi-user — your Telegram chat is one-to-one with your Titan. Group
  chats are not on the roadmap; the privacy/UX trade-off doesn't fit.

### Costs

Zero monetary. Telegram's API is free for bots.

---

## Terminal `/chat` (always-on fallback)

The lowest-level channel: a Python script that signs your message with
your Maker keypair and POSTs to Titan's local HTTP API.

### Setup

None. If you ran the installer, this works. From the Titan repo root:

```bash
bash scripts/terminal_chat.sh
```

It opens a REPL connected to the local Titan. You type, you press Enter,
Titan answers. Ctrl-D to quit.

### What works

- Plain text turns
- The same `/chat` endpoint Telegram uses, with the same
  meta-reasoning + Synthesis Engine pipeline behind it
- Replay of recent conversation history if you press up-arrow

### What doesn't

- No formatting beyond plain text
- No multi-line input by default (paste-friendly mode is
  `--multiline`)
- Requires SSH access to the box (or being at the box). Not a remote
  channel.

### Why this exists

When everything else breaks, this works. Telegram outage? Cert lapse?
DNS issue? Web frontend won't build? You can `ssh` to your box and run
`terminal_chat.sh` and Titan responds.

We verify this channel keeps working at every release.

---

## TC² console `/chat` (web UI — installed by default)

The **Titan Command Center (TC²)** is the lean owner-facing web UI that
ships with every Titan. The installer sets it up automatically — no
opt-in, no domain, no TLS, no node build (the SPA bundle is prebuilt and
committed). It runs as its own `titan-console.service` on
`http://127.0.0.1:7799`.

### Why it's a separate service (and why that matters)

TC² runs on the **system Python, stdlib-only, decoupled from the Titan
runtime** — it is *not* supervised by the Titan's Guardian and shares no
dependencies with it. The point: when the Titan itself is down — a bad
deploy, a broken venv, a crash — TC² stays up and tells you *why*
(degraded-health banner + journal tail), instead of going dark with the
thing it's supposed to monitor.

### Setup

None. If you ran the installer, it's already running. Confirm with:

```bash
systemctl status titan-console.service
curl -s http://127.0.0.1:7799/console/health      # expect 200
```

Open `http://127.0.0.1:7799` in your browser. Mutating actions are
gated by a token written to `~/.titan/console_token` at install (paste
it once in the Settings tab).

### Remote access

TC² binds `127.0.0.1` only. To reach it off-box, put your **own** reverse
proxy / TLS in front of `:7799` (nginx + Let's Encrypt, a Tailscale tail,
an SSH tunnel — your choice). Nothing about remote access is wired for
you, by design: the default install is private to your box.

### What works

- **Chat** tab — the same `POST /v6/chat` pipeline, in the browser
- **Stats** tab — live consciousness / mood / SOL / memory metrics
- **System** tab — service health, off-site backup config, restart controls
- **Settings** tab — token + API base

### What's coming

- Multi-Titan switching (T1/T2/T3 in the same UI)
- Mobile-responsive layout polish

> The heavy three.js Observatory dashboard is **no longer shipped to
> users** (it remains the maintainer's own public showcase). TC² is the
> supported owner UI.

---

## What's the same across all channels?

The endpoint behind all three channels is the same `POST /chat` (now
`POST /v6/chat` under the api/v6 single roof). They differ only in
*how* the user's text reaches that endpoint and *how* Titan's response
is formatted on the way back.

This means:

- Conversation continuity is preserved across channels — you can start
  a thread on Telegram, continue it on the terminal, finish it in the
  TC² console.
- Authentication is uniformly Maker-wallet-signed (Ed25519). No
  channel has weaker auth than another.
- The Synthesis Engine treats all incoming messages the same way:
  they all go to Mind, all produce the same kind of episodic memory,
  all get the same chance to compound into earned knowledge.

---

## Roadmap and backlog

| Channel | Status | Notes |
|---------|--------|-------|
| Telegram | ✅ shipped (default) | guaranteed by `--default` |
| Terminal `/chat` | ✅ shipped (always-on) | ships with every Titan |
| TC² console `/chat` | ✅ shipped (default) | localhost; your own proxy for remote |
| Discord | ⏳ scaffolded, untested | `config.toml [channels]` has the slot |
| Slack | ⏳ planned | corporate user demand may bring it forward |
| Signal | ❓ uncertain | Signal's bot story is more constrained |

We add a channel to the supported set only when it has been tested
end-to-end. The scaffolding-not-tested rows above mean: don't enable
them in production without expecting to debug.

---

→ [Getting started](getting-started.md) — full install walk-through
→ [Setup modes](setup-modes.md) — what each mode gives you
→ [Inference providers](inference-providers.md) — Ollama / OpenRouter setup

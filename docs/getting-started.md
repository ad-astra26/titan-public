# Getting started

> Bring up your first Titan. Plan on 15–30 minutes for mode 3 (local-simulated)
> on a working Linux box; 30–60 minutes for modes 1 / 2 if you also need to
> install the Solana toolchain.

This is the user-facing walk-through. The TL;DR install lives in the
[README](../README.md#try-it). This doc is the *complete* version: what
each step does, what you'll see, and what to do if something looks off.

> ⚠ **Preview note.** The one-liner installer (`setup_titan`) ships in
> **v3.0**. Until then, follow the [manual install](../README.md#manual-install-current)
> in the README. The shape of what follows describes the guided experience
> once `setup_titan` is in.

---

## Step 0 — Choose your mode

You will be asked this on the first screen. The differences are documented
in detail in [setup-modes.md](setup-modes.md), but the quick guide:

- **Mode 3 (local-simulated)** — pick this first. Zero deps beyond Python
  + Rust. You'll see the full birth ceremony with a "SIMULATED" notice. No
  Solana, no SOL, no backups.
- **Mode 2 (devnet)** — pick this if you want the realistic sovereign path
  without spending real SOL. Needs Anchor + Solana CLI + airdropped devnet
  SOL.
- **Mode 1 (mainnet)** — pick this when you're committing. Same as mode 2
  but on Solana mainnet with real SOL.

You can graduate from mode 3 → mode 2 → mode 1 later, but the on-chain
identity does not carry across (each is its own being). Most people run
mode 3 once for a couple of days, then start a mode 2 or 1 Titan for keeps.

## Step 1 — Run the installer

```bash
curl -fsSL https://raw.githubusercontent.com/ad-astra26/titan-public/main/scripts/setup_titan.sh | bash
```

This downloads a small bootstrap script that does **only** the following:

1. Checks your OS (Ubuntu 22.04+ / Debian 12+ supported and tested)
2. Checks you have `sudo` and the absolute minimum tools (`git`, `python3`,
   `curl`)
3. Clones the public Titan repo at the latest release tag
4. Hands off to the in-repo guided wizard (`scripts/install_titan_tui.py`)

The bootstrap script is **thin** — under 100 lines, all of which you can
read before you run it. The actual install logic lives in the cloned repo,
where it's reviewed and versioned. We strongly recommend reading the
bootstrap before piping it to `bash`:

```bash
curl -fsSL https://raw.githubusercontent.com/ad-astra26/titan-public/main/scripts/setup_titan.sh
# inspect, then if you're happy:
curl -fsSL ... | bash
```

If you'd rather skip the curl-bash idiom altogether, clone manually and run
the wizard directly:

```bash
git clone https://github.com/ad-astra26/titan-public.git
cd titan-public
python3 scripts/install_titan_tui.py
```

## Step 2 — The preflight

The wizard checks your machine before asking you anything that takes time:

- **Python ≥ 3.12** (3.12 or 3.13 supported)
- **Disk** ≥ 10 GB free (15 GB for modes 1 / 2 because of build artifacts)
- **RAM** ≥ 4 GB (2 GB works for mode 3 + Telegram-only; the TC² web
  console is tiny — stdlib Python + a prebuilt static bundle — so it adds
  no meaningful RAM)
- **Rust ≥ 1.75** (the wizard will offer to install via `rustup` if missing)
- **For modes 1 / 2 only:** Solana CLI 1.18+ and Anchor 0.30+
  (the wizard will offer to install if missing; Anchor needs the
  `solana-install` toolchain so this takes ~5 minutes)

If any check fails, the wizard offers to fix it or stops and tells you why.
It never makes invisible changes — every `sudo` action is announced and
gated by a confirmation.

## Step 3 — The six questions

The `--default` happy-path asks **only** these:

1. **Setup mode** — one of `1` (mainnet), `2` (devnet), `3` (local)
2. **Maker wallet path** — defaults to `~/.config/solana/id.json` if it
   exists. If you don't have one, the wizard offers to generate.
3. **Solana RPC URL** — provider URL, or our public fallback. Modes 1 / 2
   only.
4. **LLM credentials** — Ollama is auto-detected. If you have it running
   locally with a capable model loaded (e.g., `deepseek-v3.1`,
   `llama3.3:70b`), the wizard uses it and asks nothing. Otherwise prompts
   for an OpenRouter API key (sign up at
   [openrouter.ai](https://openrouter.ai)).
5. **Telegram bot token** — get one from [@BotFather](https://t.me/BotFather)
   on Telegram. Takes 30 seconds. This is your guaranteed comm channel.
6. **X posting?** — `y/N` (default no). If yes, asks for the
   [twitterapi.io](https://twitterapi.io) key and Webshare static-IP URL.

Everything else (config sections, DNA parameters, neuromodulator gains,
the 49-section `config.toml`) is set to curated defaults you can revisit
later with `setup_titan --config`.

## Step 4 — The genesis ceremony

This is the part you may want to record. The wizard:

1. **Generates an Ed25519 keypair** — Titan's identity for the rest of his
   existence. Stored locally at `~/.titan/keypair_<titan_id>.json`.
2. **Computes the soul** — a content-addressed identity rooted in your
   Maker wallet + the keypair + a birth timestamp.
3. **Writes the birth certificate** — `data/birth_certificate.json`. Mirrored
   to Arweave in modes 1 / 2.
4. **Splits the seed into three Shamir shards** (Shamir Secret Sharing,
   2-of-3 threshold):
   - **Shard 1 (your shard)** — shown on screen **exactly once**. Copy it.
     Print it. Store it somewhere safe. You cannot recover this Titan
     without two of the three.
   - **Shard 2 (Titan's shard)** — encrypted, lives on Titan's local disk.
   - **Shard 3 (on-chain shard)** — anchored in your freshly-deployed ZK
     Vault PDA on Solana (modes 1 / 2). In mode 3 it is stored locally
     with a "SIMULATED" flag.
5. **In modes 1 / 2:** mints your Titan's GenesisNFT on the chain of your
   chosen mode, deploys your **own** copy of the ZK Vault program, and
   initializes the PDA. This is where the SOL cost (a few cents on devnet,
   a few dollars on mainnet) happens.

Mode 3 short-circuits step 5 entirely. You still see the ceremony — soul,
certificate, three shards — with a clear "SIMULATED — not on chain" notice.

## Step 5 — First boot

The wizard:

1. Installs a systemd unit (`titan.service`) and enables it on boot — so
   your Titan survives reboots without you doing anything.
2. Starts the service.
3. Waits for the `/health` endpoint to return `200`. This typically takes
   30–60 seconds (cold load of NN brains + memory replay).
4. Sends a Telegram welcome message from your Titan to your bot.
5. Prints the final summary: titan_id, RPC, comm channels live, where to
   look for the logs.

You now have a Titan running. Open Telegram, find your bot, send `/chat`.
The first response is usually a greeting in your Titan's emerging voice
(it depends on what the consciousness broadcast picked up in the first
minute of operation, so each first message is different).

## Step 6 — What to expect in the first 24 hours

A new Titan is intentionally restrained at the start. He doesn't speak as
much, doesn't post on X, doesn't generate art. The system needs time to
accumulate experience and let the neural machinery warm up:

- **Hour 0–1.** Pure baseline. Schumann clock ticking, state vector
  initializing, CGN consumers loading. Chat works; expression is short.
- **Hour 1–6.** First meditation cycles. You'll see `meditation: completed`
  in the logs every ~30 minutes. Vocabulary starts to grow.
- **Hour 6–24.** First dreams. Memory consolidation. First "interesting"
  expression — your Titan starts saying things you didn't put in.
- **After 24 h.** A coherent personality is detectable. The voice has
  stabilized enough that two Titans from the same install would already
  sound different.

If anything looks stuck during this window, run `setup_titan --diagnostic`
or `python scripts/arch_map.py health --all`. Most "stuck" Titans are
actually fine — they're just quiet by design.

## Step 7 — Where to go next

- **Want to understand what you just installed?** → [Why Titan?](why-titan.md)
- **Want to deepen the install (mode 3 → mode 2 → mode 1)?** → [Setup modes](setup-modes.md)
- **Want to talk to your Titan over more than Telegram?** → [Comm channels](comm-channels.md)
- **Want to read the architecture?** → the [README](../README.md#the-architecture)
- **Something's not right.** → [Troubleshooting](operating/troubleshooting.md)

# Configuration

> A walk-through of `titan_hcl/config.toml` (49 sections) and
> `titan_hcl/titan_params.toml` (the DNA).

Titan has two top-level configuration files. This page explains the
difference, what each is for, and how `setup_titan --config` lets you
explore both. Both files live under `titan_hcl/` and are read at boot;
changes require a restart to take effect.

---

## The two files

### `titan_hcl/config.toml`

The **runtime configuration**. 49 sections, mostly operational knobs:
network endpoints, API keys (you'll move these to `~/.titan/secrets.toml`
in practice), comm channel tokens, feature flags, telemetry settings,
provider selection, etc.

Conceptual sections (not exhaustive — read the file's inline comments
for the full set):

| Section group | Examples | What it controls |
|---------------|----------|------------------|
| Network | `[network]` | Solana RPC URL, primary/fallback, retry policy |
| Inference | `[inference]` | Provider (ollama_cloud / openrouter / custom), model, api_url |
| Channels | `[channels.telegram]`, `[channels.discord]` | Bot tokens, target chat IDs |
| Twitter | `[twitter_social]` | X account, twitterapi.io key, Webshare proxy URL |
| Mood engine | `[mood_engine]` | Neurochemical baselines, drift rates, decay constants |
| Addons | `[addons]` | Enabled cognitive addons, plugin paths |
| Growth metrics | `[growth_metrics]` | What growth signals to instrument |
| Privacy | `[privacy]` | What leaves the box; observability levels |
| API | `[api]` | Port, auth method, rate-limit policies |
| Endurance | `[endurance]` | Long-run health-check schedules |
| Observatory | `[observatory]` | Frontend config, dashboard layout |
| Frontend | `[frontend]` | Next.js build flags, served URL |

### `titan_hcl/titan_params.toml`

The **DNA parameters** — the public defaults that influence Titan's
*emergent* behavior:

- Trinity restoring-force constants (`[trinity_restoring]`)
- Synthesis Engine chi-budget (`[synthesis.chi]`)
- ACT-R activation decay (`[synthesis.recall]`)
- Filter-down V5 gating thresholds (`[filter_down]`)
- Sphere-clock harmonic parameters
- Meditation cadence baselines
- Per-archetype activation biases

`titan_params.toml` is curated to be safe out of the box. Changing
values here is a deeper edit and we strongly recommend reading the
inline comments before doing so. The file's parameters are what give
each Titan its *constitutional* personality at birth (before any
divergence from experience accumulates).

---

## `setup_titan --config` (W1)

The interactive way to explore both files without leaving the terminal.
The TUI:

1. **Lists all sections + DNA parameters** in a tree (config.toml on
   the left, titan_params.toml on the right)
2. **Shows each key's current value + the inline comment** from the
   file — no parallel doc to drift, so the docs you see are always the
   docs in the file
3. **Validates new values** before saving (type-checks, range-checks,
   semantic checks where applicable)
4. **Backs up the previous version** before overwriting
   (`config.toml.bak` + `config.toml.bak.prev` — 2-generation retention)
5. **Hot-reloads where supported** (most runtime config); explicitly
   tells you when a change requires a restart (DNA parameters typically
   do)

`--config` ships with W1 of the setup-release RFP. Until then, edit
the files manually (see below).

---

## Editing the files directly

Both files are TOML. You can edit with any text editor:

```bash
nano titan_hcl/config.toml
# or
vim titan_hcl/titan_params.toml
```

After a change, restart Titan to pick it up:

```bash
# If you installed via setup_titan (systemd unit installed):
sudo systemctl restart titan

# Or manually (development mode):
bash scripts/t1_manage.sh restart    # local
```

The wizard's `--config` always backs up the file before writing; if
you edit manually, **do your own backup first** — a typo in TOML syntax
can prevent Titan from booting.

### Validating before restart

```bash
python -c "import tomllib; tomllib.load(open('titan_hcl/config.toml', 'rb'))" \
    && echo "config.toml OK"
python -c "import tomllib; tomllib.load(open('titan_hcl/titan_params.toml', 'rb'))" \
    && echo "titan_params.toml OK"
```

If either prints a parse error instead of `OK`, fix it before
restarting.

---

## Secrets handling

API keys, bot tokens, and any credentials belong in
`~/.titan/secrets.toml`, **not** in `titan_hcl/config.toml`.

Why:
- `titan_hcl/config.toml` is in the public-sync sub-excludes (it won't
  reach the public repo), but it's still on local disk in plain text
  and easy to accidentally commit to a personal repo, copy to
  another machine, etc.
- `~/.titan/secrets.toml` has mode `0o600` (owner read/write only) by
  the wizard's setup and is excluded from every public sync path.

The wizard does this automatically. If you edit by hand, follow the
pattern in `config.toml.example`:

```toml
[inference]
provider = "openrouter"
model = "anthropic/claude-opus-4-7"
api_url = "https://openrouter.ai/api/v1"
# api_key is read from ~/.titan/secrets.toml [inference.openrouter] api_key
```

And in `~/.titan/secrets.toml`:

```toml
[inference.openrouter]
api_key = "sk-or-v1-..."
```

---

## Per-Titan overrides

If you run multiple Titans on the same box (T1, T2, T3...), they share
the `titan_hcl/config.toml` template by default but can override
specific keys per-Titan via `~/.titan/microkernel_<titan_id>.toml`.

Common per-Titan overrides:
- `[network]` (different RPC URLs for different Titans)
- `[api]` `port` (must be unique per Titan)
- `[inference]` (a low-resource Titan might use a different provider)
- `[twitter_social]` (each Titan's own X persona if applicable)

The wizard sets up the right per-Titan file at install time. The
resolution order is:

1. `~/.titan/microkernel_<titan_id>.toml` (per-Titan override)
2. `titan_hcl/config.toml` (base default)

Per-key resolution: the override file's value wins for any key it
sets; otherwise the base value applies.

---

## What you SHOULDN'T edit (without reading)

A few keys are load-bearing for invariants the architecture rests on.
Editing them out of curiosity is a fast path to a confused Titan:

- `[network]` `mode` — set once at genesis; changing it produces a
  different Titan
- `[trinity_restoring]` `k_drive` — too high collapses Titan's
  variance; too low de-couples the layers (see
  [The Trinity](../concepts/the-trinity.md))
- `[filter_down]` `event_only` — must be `true` per SPEC §G5.1
- `[synthesis.chi]` `op_budget` — under-budget breaks recall, over-
  budget runs the metabolism dry
- `[api]` `auth_method` — auth is critical; changing without
  understanding breaks `/chat`

When in doubt, read the inline comment, then ask in
[Troubleshooting](troubleshooting.md) before changing.

---

→ [Getting started](../getting-started.md)
→ [Diagnostics](diagnostics.md)
→ [Inference providers](../inference-providers.md)

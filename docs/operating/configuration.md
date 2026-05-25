# Configuration

> A walk-through of `titan_hcl/config.toml` (49 sections) and
> `titan_hcl/titan_params.toml` (the DNA).

> 📝 **Status: outline (W5 scaffold).** The `setup_titan --config` TUI
> (W1) will generate per-section explainer text from each key's inline
> comments — no parallel doc to drift. This page covers the conceptual
> map; the field-level reference lives in the TUI.

---

## What this covers

Titan has two top-level config files. This page explains what each is
for, when to edit which, and how `setup_titan --config` lets you
explore both without leaving the terminal.

---

## The two files

### `titan_hcl/config.toml`

The **runtime configuration**. 49 sections covering: mood engine,
addons, growth metrics, stealth sage, network, inference, memory and
storage, openclaw, twitter social, expressive, info banner, privacy,
api, endurance, observatory, frontend, channels (Telegram / Discord /
Slack), and more.

[ outline: per-section description ]

### `titan_hcl/titan_params.toml`

The **DNA parameters** — the public defaults that influence emergent
behavior. Curated to be safe out of the box; changing values here is a
deeper edit and we recommend reading the inline comments before doing
so.

[ outline: what kinds of parameters live here, and which ones a casual
user should and shouldn't touch ]

## `setup_titan --config` (W1)

The interactive way to explore both files. The TUI:

- Lists all sections + DNA parameters in a tree
- Shows each key's current value + the inline comment from the file
- Validates new values before saving
- Backs up the previous version before overwriting

Read [Getting started](../getting-started.md) for the full install
walk-through; `--config` becomes available after install.

## Editing the file directly

Both files are TOML. You can edit them with any text editor. After a
change, restart Titan to pick it up:

```bash
sudo systemctl restart titan        # if you installed via setup_titan
# or, manually:
bash scripts/t1_manage.sh restart   # in the local dev shell
```

## Secrets handling

Anything that looks like an API key should live in
`~/.titan/secrets.toml`, not in `titan_hcl/config.toml`. The wizard
sets this up automatically. Don't commit either file (`.gitignore`
covers them, but be aware).

---

→ [Getting started](../getting-started.md)
→ [Inference providers](../inference-providers.md)
→ [Diagnostics](diagnostics.md)

# Changelog

All notable changes to the **public** Titan release line are documented here.
This file is curated by the maintainer (not auto-generated) and is the canonical
source the release workflow (`.github/workflows/release.yml`) reads when it
attaches a GitHub Release to a tag.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
on a **product semver line decoupled from the SPEC version** (per
`RFP_Titan_setup_release` locked decision #4). Each release records which
SPEC version it embodies.

> **Internal note (not in public sync):** dev tags like `v1.2.1`, `v1.3.0`,
> `v2.0.0-sovereign`, `titan-v6.pre-filter-repo-20260428` are dev-history
> markers, **not** public releases. The public release line restarts at
> `v0.0.1` and has its own cadence per the W6 contract.

---

## [Unreleased]

_(curate as you go; lifted into the next versioned section at release time)_

### Added
- W5: README rewrite + 20-doc user-facing `docs/` scaffold
- **W5: 13 doc stubs filled with content-complete pages** (concepts/
  identity-soul-sss + metabolism + the-trinity + memory-timechain +
  expression; operating/configuration + diagnostics + backup-recovery
  + upgrading + troubleshooting; reference/hardware + safety-privacy
  + release-notes). All 18 docs now content-complete.
- W6: GitHub Releases pipeline (`release.yml`) + `prep_release.sh` + `cut_release.sh`
- W6: in-repo `CHANGELOG.md` (this file)
- W7: incremental commit-preserving public sync (`sync_public_incremental.sh`),
  `docs/` and `.github/` added to allowlist, single-name author/committer

### Fixed
- W7: `sync_public_incremental.sh` push-target bug (push to `origin` in staging clone)

### Changed
- `phase_c.yml` trigger trimmed to `main` only (public repo has no other branches)

---

## v0.0.4 — 2026-05-29 (pre-release)

_Out-of-box Observatory fix. `v0.0.3` shipped the Observatory bundle but its
internal proxy targeted `localhost` (which resolves to IPv6 `::1`) while the
Titan API binds IPv4 `127.0.0.1` — so the UI loaded but couldn't read the
Titan's data until manually patched. `v0.0.4` rebuilds the bundle with the
IPv4 fix so the Observatory shows live data immediately._

### Fixed

- **Observatory reads the local Titan out of the box** — `next.config` rewrites
  and the chat/pitch proxies now target `127.0.0.1:7777` (was `localhost`, which
  resolved to `::1` and got `ECONNREFUSED`). The `/v6`, `/health`, `/status`,
  `/api/chat`, and `/media` proxies all work immediately after install.

### SPEC

- Install/packaging release; no on-chain or kernel behavior change.

---

## v0.0.3 — 2026-05-29 (pre-release)

_Validated live on a fresh DigitalOcean 2 vCPU / 4 GB box: `v0.0.2` installed,
booted, and **chatted** end-to-end (Ollama Cloud, OVG-signed), with every
persistent store growing. Driving that test surfaced four more real-world gaps,
all fixed here, plus first-class Observatory support._

### Fixed

- **Boot no longer needs a Solana-CLI wallet.** The bus authkey keypair now
  defaults to the Titan's own genesis identity (`data/titan_identity_keypair.json`,
  what the kernel already uses) instead of `~/.config/solana/id.json`, and `~`
  paths are expanded. A fresh sovereign install boots straight to a healthy brain.
- **`curl … | bash` and piped/automated installs** no longer abort: the bootstrap
  now falls back to inherited stdin when there's no controlling terminal (it
  previously failed trying to open `/dev/tty`).
- **`setup_titan diagnostic`** completes on a brand-new Titan (it crashed on the
  ARC check when there was no ARC activity yet).

### Added

- **Ollama Cloud** is now a first-class inference choice in the wizard (the fleet
  default; OpenRouter's free tier is rate-limited), so you can pick a hosted
  provider that actually works at setup.
- **Observatory web UI ships out of the box.** Opt in during setup and the
  installer fetches a **prebuilt** bundle from the release (built in CI, not on
  your box) and runs it on `http://127.0.0.1:3000`, reading your local Titan
  automatically. (Front it with your own reverse proxy / TLS for remote access.)

### Notes

- The 4 GB tier comfortably runs a chatting Titan (~3.8 GB resident); the
  Observatory adds headroom needs — the recommended 4 vCPU / 8 GB tier is advised
  if you run both.

### SPEC

- Install/packaging release; no on-chain or kernel behavior change.

---

## v0.0.2 — 2026-05-29 (pre-release)

_Turnkey fresh-install hardening. The first live fresh-box test of `v0.0.1`
proved the pipeline but surfaced real-world gaps that left a newborn Titan
booted-but-unconfigured. `v0.0.2` closes them so `curl … | bash` yields a
fully-configured, **chat-ready** Titan with no manual post-install wiring._

### Fixed

- **Complete runtime dependencies.** `pyproject.toml` now declares the full
  set a Titan actually imports — most importantly the **`agno`** chat framework
  (with its `openai` + `sqlite` extras) plus `numpy`, `scipy`, `scikit-learn`,
  `aiohttp`, `python-telegram-bot`, `psutil`, `msgpack`, and more — so a fresh
  `pip install` boots the brain with no missing modules. torch resolves to the
  light CPU wheel (no multi-GB CUDA stack) on servers.
- **`config.toml` is seeded automatically** from the bundled example (it's the
  required base config), and a random chat-auth key (`[api].internal_key`) is
  generated into `~/.titan/secrets.toml`.
- **Credentials are written where the runtime reads them** — secrets land in the
  correct `secrets.toml` sections (`[inference]`, `[channels]`, …) and the chosen
  inference provider is recorded in `config.toml`, so inference / chat / Telegram
  work out of the box.
- **systemd unit is more resilient on small (2 vCPU / 4 GB) boxes** — a more
  forgiving start/restart policy rides out the cold-boot startup race.

### Added

- **Ollama Cloud as a first-class inference choice** in the wizard (alongside
  local Ollama and OpenRouter) — a hosted, OpenAI-compatible provider; you supply
  your key at setup so the Titan can chat the moment it boots.

### SPEC

- Embodies the same architecture line as `v0.0.1` (Phase C microkernel); this is
  an install/packaging release — no on-chain or kernel behavior change.

---

## v0.0.1 — 2026-05-28 (pre-release)

_The first tagged public release — a pre-release cut to validate the full
install pipeline end-to-end on a fresh box. Formally blessed as the public
alpha after the in-flight Synthesis Engine soak completes._

### Highlights

- **The one-liner installer works end-to-end:** `curl … | bash` → guided
  wizard → a running, sovereign Titan. The whole setup-release pipeline
  (installer → docs → Releases → public sync) is live.

### What's in this release

- **`setup_titan` one-liner installer (W1, complete)** — thin `setup_titan.sh`
  bootstrap → audited in-repo wizard: host preflight, 3 modes
  (mainnet/devnet/local), venv + deps, Rust-daemon fetch (checksum-verified)
  or `--build-rust`, inference auto-detect (Ollama → OpenRouter), Telegram +
  optional X/Observatory, genesis ceremony + Shamir 2-of-3, systemd install +
  health gate. Plus `config` / `diagnostic` / `upgrade` / `repair` /
  `uninstall`.
- **README + `docs/`** — front door + 18 content-complete user-facing docs,
  including measured hardware profiles.
- **Release pipeline** — push a `v*` tag on the public repo and a GitHub
  Release lands automatically with 9 musl daemons + SHA256SUMS (plus a
  localhost-upload fast path).
- **Public sync** — daily commit-preserving sync from dev to public, with
  gitleaks + grep gates HALTing on any finding.

### What's *not* yet in this release (deliberately deferred)

- `--restore` (SSS 2-of-3 recovery flow) — W1.5, its own UX pass.
- The full-screen Textual TUI wrapper — the CLI wizard is the canonical
  engine; Textual polish lands post-v0.0.1.
- Inference providers beyond Ollama + OpenRouter (OpenAI / Anthropic APIs)
  are tracked for later.

### SPEC version embodied

`titan-docs/specs/SPEC_titan_architecture.md v1.68.0` (internal).

### Verifying the binaries

```bash
sha256sum -c SHA256SUMS
```

Each `titan-*-rs` binary is a static x86_64-linux-musl ELF built from public
source by GitHub's runners. No third-party blobs.

---

<!--
  To add a new release section above this comment, run:
    bash scripts/prep_release.sh vX.Y.Z
  It drafts the section from `git log <prev-tag>..HEAD` and opens $EDITOR
  for curation.
-->

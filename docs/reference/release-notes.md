# Release notes

> Where to read release notes, how they're written, and the release
> process.

The Titan project ships releases via tagged GitHub Releases on the
public repo. This page explains where to find them, the conventions
used, and the release flow for maintainers.

---

## Where to read release notes

In order of authoritativeness:

1. **[GitHub Releases](https://github.com/ad-astra26/titan-public/releases)**
   — the canonical source. Each `vX.Y.Z` tag has its own page with
   rendered notes + attached prebuilt binaries + SHA256SUMS.
2. **GitHub Release tag pages** — direct URL pattern:
   `github.com/ad-astra26/titan-public/releases/tag/vX.Y.Z`. Same
   content as #1 but addressable per release.
3. **`CHANGELOG.md`** (in-repo, root) — curated mirror, versioned
   alongside the code. The release workflow reads this file to
   compose the GitHub Release body.

If any disagree, GitHub Releases wins for accuracy of "what's actually
shipped" (the binaries on the release page are what users download).
CHANGELOG.md wins for editorial intent (what we said the release was
about).

---

## How releases are versioned

**Product semver, decoupled from SPEC version** (per RFP_Titan_setup_release
locked decision #4):

- **PATCH** (`vX.Y.Z` → `vX.Y.Z+1`) — critical bugfix; no API changes;
  no DNA changes; no SPEC contract changes
- **MINOR** (`vX.Y` → `vX.Y+1`) — one tested major feature ships;
  backwards compatible at the persisted-state level
- **MAJOR** (`vX` → `vX+1`) — persisted-state shape changed; migration
  required; tagged with explicit `MIGRATIONS.md` prose

Each release records which SPEC version it embodies. The SPEC version
bumps weekly (often multiple times); product version bumps only on
deliberate public releases.

---

## Cadence

- **MINOR**: ~weekly for tested major features
- **PATCH**: ad-hoc for critical bugfixes
- **MAJOR**: rare — significant persisted-state shape changes

This is deliberately NOT a daily-ship cadence. Each public release is
a stable, tested, well-documented artifact. Dev work happens
continuously; releases happen consciously.

---

## What's in each release

Every release attaches the same set of files:

| Asset | What it is |
|-------|------------|
| `titan-inner-body-rs` | musl-static x86_64-linux binary |
| `titan-inner-mind-rs` | musl-static x86_64-linux binary |
| `titan-inner-spirit-rs` | musl-static x86_64-linux binary |
| `titan-kernel-rs` | musl-static x86_64-linux binary |
| `titan-outer-body-rs` | musl-static x86_64-linux binary |
| `titan-outer-mind-rs` | musl-static x86_64-linux binary |
| `titan-outer-spirit-rs` | musl-static x86_64-linux binary |
| `titan-trinity-rs` | musl-static x86_64-linux binary |
| `titan-unified-spirit-rs` | musl-static x86_64-linux binary |
| `SHA256SUMS` | SHA256 hash for each binary, plus the file itself |

Plus the GitHub Release body (rendered from CHANGELOG.md) and the
tag's annotated message.

### How to verify a release

```bash
# Download all 10 files
curl -fsSL -O https://github.com/ad-astra26/titan-public/releases/download/vX.Y.Z/SHA256SUMS
for b in titan-{inner,outer}-{body,mind,spirit}-rs titan-kernel-rs titan-trinity-rs titan-unified-spirit-rs; do
  curl -fsSL -O "https://github.com/ad-astra26/titan-public/releases/download/vX.Y.Z/$b"
done

# Verify checksums
sha256sum -c SHA256SUMS
```

A correctly-built release passes `sha256sum -c` cleanly. Each binary
is a static x86_64-linux-musl ELF built from public source by
GitHub's own runners — no third-party blobs, no Anthropic dependency.

### How releases are built (reproducible)

The workflow lives at `.github/workflows/release.yml` (also visible
on the public repo). Trigger: `on: push: tags: ['v*']`. Build
environment: `ubuntu-latest` GitHub-hosted runner. Toolchain: Rust
stable + `musl-tools`. Build command: `cargo build --release --bins
--target x86_64-unknown-linux-musl`.

You can verify the build is reproducible by running the same
toolchain on your own machine against the same tag and comparing
SHA256SUMS.

---

## Release process (for maintainers)

The professional flow, per the W6 scripts:

### Step 1: prep the CHANGELOG entry (on dev, titan-v6)

```bash
bash scripts/prep_release.sh v0.0.2
```

Drafts a CHANGELOG.md section from `git log <prev-tag>..HEAD` grouped
by conventional-commit prefix (feat/fix/docs/refactor/etc.). Inserts
above `[Unreleased]`. Opens `$EDITOR` for curation — tighten prose,
group logically, drop noise.

### Step 2: commit + sync to public

```bash
git add CHANGELOG.md
git commit -m "chore(release): prep v0.0.2"
bash scripts/git_publish.sh
```

This pushes to private origin (titan-dev) and syncs to public
(titan-public). The release-prep commit lands on public/main.

### Step 3: cut the release

```bash
bash scripts/cut_release.sh v0.0.2
```

This clones the public repo to a temp dir, tags `v0.0.2` at public's
HEAD with annotated message = CHANGELOG section, pushes the tag. The
tag push fires `.github/workflows/release.yml`, which builds the 9
binaries, computes SHA256SUMS, extracts the CHANGELOG section, and
creates the GitHub Release.

### Step 4: verify

Watch
[github.com/ad-astra26/titan-public/actions](https://github.com/ad-astra26/titan-public/actions).
When the workflow completes, the Release appears at
[`releases/tag/v0.0.2`](https://github.com/ad-astra26/titan-public/releases).

---

## Per-release entry template

Each CHANGELOG section follows this shape:

```markdown
## v0.0.2 — 2026-06-01

### Highlights
- One or two paragraphs describing what's significant about this
  release. Why is the user reading these notes excited / cautious?

### Added
- New feature A
- New feature B

### Fixed
- Bug X (closes BUG-SLUG)
- Bug Y

### Changed
- Behavior Z now ...
- API endpoint /v6/foo renamed to /v6/bar (no breaking change; alias
  preserved for two MINOR cycles)

### Removed
- Deprecated foo (see migration notes)

### **Breaking changes** (only if any)
- Detailed prose explaining what changed + how to migrate + a link
  to a longer migration doc if needed

### SPEC version embodied
`titan-docs/specs/SPEC_titan_architecture.md vX.YZ.W` (internal).

### Verifying the binaries
```bash
sha256sum -c SHA256SUMS
```
```

The `prep_release.sh` script drafts this shape for you; you tighten
the prose.

---

## Pre-1.0 history

The public release line restarts at **v0.0.1**. Existing pre-v0.0.1
git tags (`v1.2.1`, `v1.3.0`, `v2.0.0-sovereign`,
`titan-v6.pre-filter-repo-20260428`, etc.) are dev-history markers
from the pre-public-release era and do NOT appear in the public
release list.

Each public release cumulates work that was already running in dev for
some time. v0.0.1 specifically embodies ~12 months of cognitive
architecture work that predates the public-release tooling itself.

---

## Subscribing to releases

- **GitHub:** click "Watch" → "Releases only" on
  [titan-public](https://github.com/ad-astra26/titan-public)
- **RSS:** `https://github.com/ad-astra26/titan-public/releases.atom`
- **Email:** if you're a GitHub user with notifications enabled

We do not maintain a separate mailing list. GitHub Releases is the
canonical announce channel.

---

→ [Upgrading](../operating/upgrading.md) — how to apply a release
→ [GitHub Releases](https://github.com/ad-astra26/titan-public/releases)

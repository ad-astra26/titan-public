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

## v0.0.1 — TBD

_The first tagged public release. Placeholder until the actual cut happens;
`prep_release.sh v0.0.1` will move the relevant items from "[Unreleased]"
into this section + replace TBD with the date._

### Highlights

- First public-release tooling end-to-end: setup-release RFP → docs scaffold
  → Releases pipeline.

### What's in this release

- **README + `docs/`** — front door + 20 user-facing docs (5 content-complete,
  the rest well-structured outlines that fill in subsequent releases).
- **Release pipeline** — push a `v*` tag on the public repo and a GitHub
  Release lands automatically with 9 musl daemons + SHA256SUMS.
- **Public sync** — daily commit-preserving sync from dev to public, with
  gitleaks + grep gates HALTing on any finding.

### What's *not* yet in this release (deliberately deferred)

- `setup_titan` one-liner installer (W1) — comes in a subsequent release.
- Most `docs/` are scaffolds; full content fills land per-release as content
  is written.
- Inference providers beyond Ollama + OpenRouter (OpenAI / Anthropic APIs)
  are tracked for later.

### SPEC version embodied

`titan-docs/specs/SPEC_titan_architecture.md v1.59.0` (internal).

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

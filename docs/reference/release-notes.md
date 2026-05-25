# Release notes

> Pointer to authoritative sources and a per-release summary template.

> 📝 **Status: outline (W5 scaffold).** Per-release entries land at each
> tagged release (W6).

---

## Where to read release notes

The canonical sources, in order of preference:

1. **[GitHub Releases](https://github.com/ad-astra26/titan-public/releases)**
   — the authoritative source. Auto-generated from the SPEC Changelog and
   commit log at each tag. Includes attached prebuilt Rust binaries.
2. **GitHub Release tag page** — every `vX.Y.Z` tag has its own page with
   the rendered notes (e.g.
   `github.com/ad-astra26/titan-public/releases/tag/vX.Y.Z`).
3. **`CHANGELOG.md` (in-repo)** — when it lands in v3.0 it will mirror
   the same content for offline reading; until then, the GitHub Releases
   page is canonical.

## How releases work

- **Versioning** — product semver, decoupled from SPEC version. PATCH
  for critical bugfixes; MINOR for tested major features. Each release
  records which SPEC version it embodies.
- **Cadence** — ~weekly MINOR releases for tested features, ad-hoc
  PATCH for critical bugs.
- **Notes** — auto-generated; the SPEC Changelog and commit log are
  the source of truth.

See [Upgrading](../operating/upgrading.md) for how to apply a release.

## Per-release entry template

[ outline: each release entry includes:
- Date + tag
- One-sentence headline
- New features (bullet list)
- Fixed bugs (bullet list with `BUG-...-SLUG` references)
- Breaking changes (bold, with migration notes if any)
- SPEC version embodied
- Pre-built binary checksums
- Special acknowledgments
]

## Pre-1.0 history

[ outline: a brief retro will appear here once we cut v3.0 — the
versions before the public-release era were development-tagged and
won't appear in this list. The v3.0 release notes will tie the public
launch to the work that preceded it. ]

---

→ [Upgrading](../operating/upgrading.md)
→ [GitHub Releases](https://github.com/ad-astra26/titan-public/releases)

# Upgrading

> `setup_titan --upgrade` semantics, version compatibility, what changes
> across releases.

> 📝 **Status: outline (W5 scaffold).** `--upgrade` ships with W1 in
> v3.x. This page locks in the contract.

---

## What this covers

How to take a running Titan from version *N* to version *N+1* safely,
what guarantees an upgrade provides (and doesn't), and how to read the
release notes for breaking changes.

---

## The upgrade contract

[ outline: idempotent; never clobbers identity; never silently changes
DNA parameters; preserves all on-chain anchors; resumable if
interrupted ]

## Two flavors of release

| Type | What changes | Example |
|------|--------------|---------|
| **PATCH** (`vX.Y.Z` → `vX.Y.Z+1`) | Critical bugfix; no API changes; no DNA changes; no SPEC contract changes | Safety hotfix |
| **MINOR** (`vX.Y` → `vX.Y+1`) | One tested major feature ships; may add new SPEC contracts; backwards compatible at the persisted-state level | New CGN consumer, new worker, new endpoint |

We do **not** ship MAJOR (`vX` → `vX+1`) releases lightly. A MAJOR
release indicates the persisted-state shape changed and migration is
required. We document migration paths in `MIGRATIONS.md` and the
release notes when they occur.

## Cadence

~weekly for tested MINOR features, ad-hoc PATCH for critical bugs.
Not daily.

## How to upgrade

```bash
# When --upgrade is in (v3.x):
setup_titan --upgrade

# Manually, until then:
cd <your-titan-checkout>
git fetch origin
git checkout vX.Y.Z       # the release tag
bash scripts/t1_manage.sh restart    # or t2/t3 depending on which Titan
```

## What `--upgrade` does

[ outline: stop systemd unit → fetch the tag → checkout → run any
migrations (idempotent) → restart → wait for health → report status ]

## What `--upgrade` will refuse to do

[ outline: skip a MAJOR release boundary without explicit confirmation;
upgrade mid-meditation without `--force`; upgrade with uncommitted
changes in the repo ]

## Reading release notes

Every release has notes auto-generated from the SPEC Changelog and the
commit log since the last tag. They live on the
[GitHub Releases page](https://github.com/ad-astra26/titan-public/releases)
— that page is the authoritative source. An in-repo `CHANGELOG.md` mirror
lands in v3.0.

We call out **breaking changes** in bold. If you see a bold call-out
in a release note for the version you're upgrading to, read the
referenced section before running `--upgrade`.

---

→ [Release notes](../reference/release-notes.md)
→ [Backup and recovery](backup-recovery.md)
→ [Troubleshooting](troubleshooting.md)

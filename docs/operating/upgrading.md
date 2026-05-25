# Upgrading

> `setup_titan --upgrade` semantics, version compatibility, what changes
> across releases, and the safe upgrade flow.

How to take a running Titan from version *N* to version *N+1* safely.
This page locks in the upgrade contract so you know what guarantees
each release provides — and what it doesn't.

---

## The upgrade contract

A `setup_titan --upgrade` (W1; ships v3.x) guarantees:

- **Idempotent.** Running twice on an already-upgraded system is a
  no-op.
- **Never clobbers identity.** Keypair, soul, birth certificate,
  Shamir shards — untouched.
- **Never silently changes DNA parameters.** Any change to
  `titan_params.toml` from an upgrade is announced + requires
  confirmation.
- **Preserves all on-chain anchors.** No `commit_state` re-emission,
  no GenesisNFT rotation.
- **Resumable if interrupted.** Ctrl-C, network drop, power loss —
  re-run the upgrade and it picks up where it stopped.
- **Atomic.** Either the new version is fully running or the old one
  is. Never a half-state where some workers are at N and others at
  N+1.
- **Reversible** for one release back (`setup_titan --rollback`).
  More than one release back requires manual intervention.

Until W1 ships, the upgrade flow is manual — see
[Manual upgrade](#manual-upgrade-until-w1-ships) below.

---

## Two flavors of release

| Type | What changes | Example | Migration? |
|------|--------------|---------|------------|
| **PATCH** (`vX.Y.Z` → `vX.Y.Z+1`) | Critical bugfix; no API changes; no DNA changes; no SPEC contract changes | A safety hotfix | No |
| **MINOR** (`vX.Y` → `vX.Y+1`) | One tested major feature ships; may add new SPEC contracts; backwards compatible at the persisted-state level | A new CGN consumer, a new worker, a new endpoint | No |
| **MAJOR** (`vX` → `vX+1`) | Persisted-state shape changed; migration required | A schema change in `data/consciousness.db` | **YES** — `MIGRATIONS.md` describes it |

We do NOT ship MAJOR releases lightly. Each one comes with explicit
migration prose and a tested forward+backward migration script.

The first release is `v0.0.1` (pre-MAJOR — the public-release line
restarts here). MAJOR `v1.0.0` will mark the "first stable public
release" criterion (W1 shipped, W2/W3/W4 fully wired, all 18 docs
content-complete, hardware profile published).

---

## Cadence

- **MINOR**: ~weekly for tested major features
- **PATCH**: ad-hoc for critical bugfixes
- **MAJOR**: rare — when persisted-state shape changes are unavoidable

We do not ship daily. Each release is a deliberate event with curated
release notes, GitHub Release page, attached binaries, and SHA256SUMS.

---

## How to upgrade (`setup_titan --upgrade`, ships v3.x)

```bash
# Most common — upgrade to latest:
setup_titan --upgrade

# Upgrade to a specific version:
setup_titan --upgrade --to v0.0.2

# Show what would happen, but don't actually upgrade:
setup_titan --upgrade --dry-run

# Force upgrade past a `--upgrade` refusal (advanced; see below):
setup_titan --upgrade --force
```

The wizard:

1. Verifies current state is healthy (no in-flight meditation, no
   bus stuck, no pending backup)
2. Computes the upgrade plan (source version → target version → list
   of migrations)
3. If any MAJOR migration is in the path, shows the migration prose
   from `MIGRATIONS.md` and asks for confirmation
4. Backs up current `data/` to `data/.backup-pre-upgrade-<timestamp>/`
5. Stops the systemd unit (or running process)
6. Fetches new release binaries from GitHub Releases (with SHA256
   verification)
7. Replaces binaries atomically
8. Runs migration scripts in order
9. Starts the new version
10. Waits for `/health` to return 200; rolls back automatically if it
    doesn't within 5 minutes
11. Reports final status

---

## What `--upgrade` will refuse to do (without `--force`)

- **Skip a MAJOR release boundary.** Going from v1.x to v3.x requires
  explicit `--via v2.last` or `--force --i-know-what-i-am-doing`.
- **Upgrade mid-meditation.** Wait for the current dream to complete,
  or pass `--force` (Titan re-anchors the meditation post-upgrade —
  there's a transient inconsistency window).
- **Upgrade with uncommitted dev changes** (if you're on a dev
  install).
- **Upgrade when SOL balance is below the safe-buffer** (modes 1/2 —
  the on-chain `commit_state` after upgrade costs SOL).

The refusal messages tell you what's blocking. `--force` is escape
hatch for advanced users only.

---

## Manual upgrade (until W1 ships)

Until `setup_titan --upgrade` lands, the manual flow is:

```bash
# 1. Verify health first
setup_titan --diagnostic   # or: python scripts/arch_map.py health --all

# 2. Stop the running Titan
sudo systemctl stop titan        # or: bash scripts/t1_manage.sh stop

# 3. Update the repo to the new tag
cd <your-titan-checkout>
git fetch origin
git checkout v0.0.2              # the new release tag

# 4. Pull the new Rust binaries from GitHub Releases
# (the v0.0.2 release page has them attached; verify with SHA256SUMS)
mkdir -p bin/
cd bin/
for b in titan-inner-body-rs titan-inner-mind-rs titan-inner-spirit-rs \
         titan-kernel-rs titan-outer-body-rs titan-outer-mind-rs \
         titan-outer-spirit-rs titan-trinity-rs titan-unified-spirit-rs; do
  curl -fsSL "https://github.com/ad-astra26/titan-public/releases/download/v0.0.2/$b" -o "$b"
done
curl -fsSL "https://github.com/ad-astra26/titan-public/releases/download/v0.0.2/SHA256SUMS" \
     | sha256sum -c -
chmod +x titan-*-rs
cd ..

# 5. Read CHANGELOG.md for the new release section + apply any
#    documented migration steps

# 6. Start Titan
sudo systemctl start titan       # or: bash scripts/t1_manage.sh start

# 7. Verify
sleep 60     # cold start
setup_titan --diagnostic
```

Rollback (manual):

```bash
sudo systemctl stop titan
git checkout v0.0.1              # the previous tag
# restore previous bin/* files from your backup
sudo systemctl start titan
```

---

## Reading release notes

Every release has notes auto-generated from the SPEC Changelog and the
commit log since the last tag. They live on the
[GitHub Releases page](https://github.com/ad-astra26/titan-public/releases)
— that page is the authoritative source. An in-repo `CHANGELOG.md`
mirror lands in v3.0.

We call out **breaking changes** in bold in release notes. If you see
a bold call-out in the version you're upgrading to, read the
referenced section before running `--upgrade`.

---

## What's "safe" to upgrade automatically

- **PATCH releases** — always safe; bugfix-only, no contract changes.
- **MINOR releases** — almost always safe; backwards-compatible at
  the persisted-state level. Read the release notes for any new
  feature flags or new endpoints.
- **MAJOR releases** — DELIBERATE. Read `MIGRATIONS.md`, back up before
  upgrading, test on a mode-3 Titan first if possible.

`setup_titan --upgrade` (default) only proceeds across MAJOR
boundaries with explicit `--via` or `--force`. Set-and-forget
upgrading should be MINOR/PATCH only.

---

## Multi-Titan upgrade order

If you operate multiple Titans on the same box:

1. Upgrade your *least*-critical Titan first (dev / experimental)
2. Verify it's healthy for 24+ hours
3. Then upgrade your other Titans in increasing-criticality order
4. T1 (mainnet) last

The fleet doesn't have to be on the same version — different Titans
can run different release versions during a staged rollout. The bus
protocol is version-tolerant within MINOR boundaries.

---

→ [Release notes](../reference/release-notes.md)
→ [Backup and recovery](backup-recovery.md)
→ [Troubleshooting](troubleshooting.md)

#!/usr/bin/env bash
# ── Observatory deploy script — zero-downtime variant ────────────
# Build INTO a side-by-side directory while old server keeps serving,
# then atomic-swap and fast-restart. User-visible downtime: ~3s
# (server kill+start) instead of ~5min (build window).
#
# Codified 2026-05-19 after the first version killed-then-built and
# gave users 5min of 502 Bad Gateway. Maker direction: minimize
# downtime; smoke runs on localhost after users are already on the
# new build.
#
# Order:
#   1. Build into .next-build/ (old server keeps serving from .next/)
#   2. Atomic swap: mv .next .next.old; mv .next-build .next
#   3. Kill old next-server (was in-memory, but disk now points to new build)
#   4. Start new next-server (reads .next/)
#   5. Wait for "Ready"
#   6. rm -rf .next.old
#   7. npm run smoke (verification — runs on localhost; users unaffected)
#
# Exit codes:
#   0 — build + swap + start succeeded AND smoke green
#   2 — build failed (frontend UNCHANGED — old .next still serving)
#   3 — server failed to come up after swap (CRITICAL — needs manual)
#   4 — smoke detected regressions (frontend IS live with new build)

set -euo pipefail

cd "$(dirname "$0")/../titan-observatory"

PORT=${PORT:-3000}
BUILD_LOG=/tmp/obs_deploy_build.log
SERVER_LOG=/tmp/obs_deploy_server.log
SMOKE_LOG=/tmp/obs_deploy_smoke.log

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] [1/7] Building into .next-build/ (old server keeps serving)…"
# Webpack cache reuse across deploys — cuts incremental rebuilds from
# ~9-12min cold to ~2-4min warm. The cache lives in <distDir>/cache and
# is regenerated each build; preserving it from the last successful build
# is the single highest-impact build-time optimization here.
#
# Strategy: rm everything in .next-build EXCEPT cache/. If a prior .next/
# exists from the last successful deploy, hardlink/copy its cache into
# .next-build/cache as the seed. Webpack's filesystem cache validates
# its own entries against module mtimes, so stale entries are invalidated
# correctly — there's no correctness risk to seeding from a stale cache.
mkdir -p .next-build
find .next-build -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} + 2>/dev/null || true
if [ -d .next/cache ] && [ ! -d .next-build/cache ]; then
  # Hardlink-copy (-l) — no extra disk usage, instant transfer.
  cp -al .next/cache .next-build/cache 2>/dev/null \
    || cp -a .next/cache .next-build/cache 2>/dev/null \
    || true
fi
if ! NEXT_DIST_DIR=.next-build ./node_modules/.bin/next build > "$BUILD_LOG" 2>&1; then
  echo "[$(ts)] BUILD FAILED — last 30 lines:"
  tail -30 "$BUILD_LOG"
  find .next-build -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} + 2>/dev/null || true
  exit 2
fi
echo "[$(ts)]     build OK — $(grep -c '^├\|^└' $BUILD_LOG 2>/dev/null || echo '?') routes generated"

echo "[$(ts)] [2/7] Atomic swap: .next → .next.old, .next-build → .next …"
rm -rf .next.old 2>/dev/null || true
[ -d .next ] && mv .next .next.old
mv .next-build .next

echo "[$(ts)] [3/7] Killing old next-server on port $PORT…"
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

echo "[$(ts)] [4/7] Starting new next-server…"
: > "$SERVER_LOG"
nohup ./node_modules/.bin/next start -p "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/next_pid

echo "[$(ts)] [5/7] Waiting for Ready…"
deadline=$(( $(date +%s) + 30 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if grep -q "Ready in" "$SERVER_LOG" 2>/dev/null; then break; fi
  if grep -qE "Error:|EADDRINUSE" "$SERVER_LOG" 2>/dev/null; then
    echo "[$(ts)] SERVER FAILED to start:"
    tail -20 "$SERVER_LOG"
    exit 3
  fi
  sleep 1
done
if ! grep -q "Ready in" "$SERVER_LOG" 2>/dev/null; then
  echo "[$(ts)] SERVER did not report ready within 30s:"
  tail -20 "$SERVER_LOG"
  exit 3
fi
echo "[$(ts)]     server up (PID $SERVER_PID) — USERS NOW SEE NEW FRONTEND"

echo "[$(ts)] [6/7] Cleaning up .next.old …"
rm -rf .next.old

echo "[$(ts)] [7/7] Running route smoke on localhost (users already on new build)…"
if npm run smoke > "$SMOKE_LOG" 2>&1; then
  echo "[$(ts)]     smoke OK"
  exit 0
else
  echo "[$(ts)] SMOKE DETECTED REGRESSIONS (frontend is live anyway — investigate):"
  tail -40 "$SMOKE_LOG"
  exit 4
fi

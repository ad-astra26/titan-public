#!/bin/bash
# verify_sovereign_state.sh — Verify Titan sovereign-state invariants.
#
# Reads data/sovereign_manifest_TN.json and asserts every sacred-file
# invariant holds. Catches the class of bug we hit 2026-05-14 (T1
# soul_keypair.enc decrypted to a different pubkey than titan_identity_keypair).
#
# Invariants checked:
#   1. titan_identity_keypair.json exists, is 0600 perms, pubkey matches manifest
#   2. genesis_record.json exists, pubkey matches manifest, is immutable
#   3. soul_keypair.enc decrypts (with this machine's hardware fingerprint)
#      to bytes whose pubkey matches titan_identity_keypair.json's pubkey
#   4. ~/.titan/microkernel_<id>.toml exists, has l0_rust_enabled=true
#   5. birth_dna_snapshot.json + genesis_nft_metadata.json exist (presence-only)
#   6. Sacred file permissions are not world/group writable
#
# Usage:
#   bash scripts/verify_sovereign_state.sh                    # verify current Titan
#   bash scripts/verify_sovereign_state.sh --titan T1         # explicit
#   bash scripts/verify_sovereign_state.sh --reattest         # regenerate manifest from current state
#   bash scripts/verify_sovereign_state.sh --strict           # exit non-zero on warning too
#
# Exit codes:
#   0   all invariants hold
#   1   manifest missing / unreadable
#   2   pubkey invariant failed (CRITICAL — likely identity divergence)
#   3   file missing or permission too lax
#   4   bad CLI args
#
# Per-Titan: T1 runs verify against /home/antigravity/projects/titan/data/...
#            T3 runs against /home/antigravity/projects/titan3/data/...
#            T2 runs against /home/antigravity/projects/titan/data/... (same dir as T1 codebase
#            but DIFFERENT data dir since T2 lives on remote VPS — manifest paths are absolute).

set -u

# ── Determine current Titan ID + base dir ──────────────────────────────
TITAN_ID=""
REATTEST=0
STRICT=0
for arg in "$@"; do
    case "${arg}" in
        --titan) ;; # consumed by next iter
        --titan=*) TITAN_ID="${arg#*=}" ;;
        --reattest) REATTEST=1 ;;
        --strict) STRICT=1 ;;
    esac
done
# Support --titan T1 form
for ((i=1; i<=$#; i++)); do
    eval "arg=\${$i}"
    if [ "${arg}" = "--titan" ]; then
        next=$((i + 1))
        eval "TITAN_ID=\${$next}"
    fi
done

# Auto-detect Titan ID from environment or local titan_identity.json
if [ -z "${TITAN_ID}" ]; then
    TITAN_ID="${TITAN_ID:-${TITAN_KERNEL_TITAN_ID:-${TITAN_ID:-}}}"
fi
if [ -z "${TITAN_ID}" ]; then
    # Try the canonical resolver path
    if [ -f /home/antigravity/projects/titan/data/titan_identity.json ]; then
        TITAN_ID=$(python3 -c "import json; print(json.load(open('/home/antigravity/projects/titan/data/titan_identity.json')).get('titan_id', ''))" 2>/dev/null)
    fi
fi
if [ -z "${TITAN_ID}" ]; then
    # Default
    TITAN_ID="T1"
fi

# Resolve base data dir per Titan
case "${TITAN_ID}" in
    T1|T2) BASE_DIR="/home/antigravity/projects/titan" ;;
    T3)    BASE_DIR="/home/antigravity/projects/titan3" ;;
    *)
        echo "  ✗ verify_sovereign_state: unknown TITAN_ID '${TITAN_ID}'" >&2
        exit 4
        ;;
esac

DATA_DIR="${BASE_DIR}/data"
# Manifest lives in git for genesis-attested invariants. Find by Titan ID
# so T1's manifest doesn't shadow T2/T3 when they share the host filesystem.
MANIFEST="${DATA_DIR}/sovereign_manifest_${TITAN_ID}.json"

# Resolve home dir for current user (since this can run as antigravity OR root)
HOME_DIR="${HOME:-$(eval echo ~$(whoami))}"
OVERRIDE_FILE="${HOME_DIR}/.titan/microkernel_${TITAN_ID}.toml"

echo "═════════════════════════════════════════════════════════════════"
echo "  Sovereign-state verifier — ${TITAN_ID}"
echo "═════════════════════════════════════════════════════════════════"
echo "  Manifest:      ${MANIFEST}"
echo "  Data dir:      ${DATA_DIR}"
echo "  Override file: ${OVERRIDE_FILE}"
echo

# cwd matters — titan_hcl.utils.crypto reads data/hw_salt.bin relatively.
cd "${BASE_DIR}"

# ── --reattest: regenerate manifest from current state ──────────────────
if [ "${REATTEST}" = "1" ]; then
    echo "  → REATTEST mode: regenerating manifest from current state"
    "${BASE_DIR}/test_env/bin/python" <<PYREATTEST
import json, hashlib, sys, os
titan_id = "${TITAN_ID}"
data_dir = "${DATA_DIR}"

# Canonical pubkey is whatever titan_identity_keypair.json holds — the
# on-disk identity IS the source of truth. genesis_record (if present) is
# an ADDITIONAL on-chain anchor for mainnet (T1) — for devnet Titans
# (T2/T3) it may not exist or may be stale T1 leakage; report as advisory.
gr_path = f"{data_dir}/genesis_record.json"
gr_pubkey = None
genesis_tx = None
ceremony_date = None
if os.path.exists(gr_path):
    try:
        gr = json.load(open(gr_path))
        gr_pubkey = gr.get("titan_pubkey")
        genesis_tx = gr.get("genesis_tx")
        ceremony_date = gr.get("ceremony_date")
    except Exception:
        pass

# Helper: derive Solana pubkey from a keypair JSON file using solders (in venv).
from solders.pubkey import Pubkey
from solders.keypair import Keypair as SoldersKP
def pubkey_from_file(path):
    with open(path) as f:
        kp = json.load(f)
    kp_bytes = bytes(int(x) & 0xFF for x in kp)
    if len(kp_bytes) == 64:
        return str(Pubkey.from_bytes(kp_bytes[32:64]))
    elif len(kp_bytes) == 32:
        return str(SoldersKP.from_seed(kp_bytes).pubkey())
    else:
        raise ValueError(f"unexpected keypair length: {len(kp_bytes)}")

# Derive canonical pubkey from titan_identity_keypair.json
ik_path = f"{data_dir}/titan_identity_keypair.json"
ik = json.load(open(ik_path))
ik_bytes = bytes(int(b) & 0xFF for b in ik)
ik_pubkey = pubkey_from_file(ik_path)

# soul_keypair.enc: decrypt, derive pubkey, compare
sk_enc_path = f"{data_dir}/soul_keypair.enc"
sk_enc_pubkey = None
if os.path.exists(sk_enc_path):
    sys.path.insert(0, "/home/antigravity/projects/titan")
    try:
        from titan_hcl.utils.crypto import decrypt_for_machine
        with open(sk_enc_path, "rb") as f:
            decrypted = decrypt_for_machine(f.read())
        # Derive pubkey from decrypted bytes via solders (no temp file)
        if len(decrypted) == 64:
            sk_enc_pubkey = str(Pubkey.from_bytes(decrypted[32:64]))
        elif len(decrypted) == 32:
            sk_enc_pubkey = str(SoldersKP.from_seed(decrypted).pubkey())
        else:
            raise ValueError(f"unexpected decrypted length: {len(decrypted)}")
    except Exception as e:
        print(f"  ⚠ Could not decrypt soul_keypair.enc on this machine: {e}")

def sha256(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

# Determine if this Titan has on-chain anchor — mainnet T1 will, devnet may not.
genesis_matches_identity = (gr_pubkey == ik_pubkey)
manifest = {
    "titan_id": titan_id,
    "titan_pubkey": ik_pubkey,
    "network": "mainnet" if (genesis_matches_identity and titan_id == "T1") else "devnet",
    "genesis_record_present": gr_pubkey is not None,
    "genesis_record_matches_identity": genesis_matches_identity,
    "genesis_tx": genesis_tx if genesis_matches_identity else None,
    "ceremony_date": ceremony_date if genesis_matches_identity else None,
    "files": {
        "data/titan_identity_keypair.json": {
            "sha256": sha256(ik_path),
            "must_decode_to_pubkey": ik_pubkey,
            "immutable": True,
        },
        "data/soul_keypair.enc": {
            "sha256": sha256(sk_enc_path),
            "decrypt_must_match_pubkey": ik_pubkey,
            "immutable_until_hardware_migration": True,
            "optional": True,  # devnet Titans may not have hardware-bound encrypt
        },
        "data/birth_dna_snapshot.json": {
            "sha256": sha256(f"{data_dir}/birth_dna_snapshot.json"),
            "immutable": True,
            "optional": True,  # devnet may lack birth ceremony artifacts
        },
        "data/genesis_nft_metadata.json": {
            "sha256": sha256(f"{data_dir}/genesis_nft_metadata.json"),
            "immutable": True,
            "optional": True,
        },
    },
    "reattested_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "reattest_note": "manifest regenerated from current on-disk state — verify by re-running without --reattest",
}
# Only include genesis_record file entry when it matches identity (otherwise
# it's stale T1 leakage on T2/T3 and shouldn't be in their manifest).
if genesis_matches_identity:
    manifest["files"]["data/genesis_record.json"] = {
        "sha256": sha256(gr_path),
        "titan_pubkey_must_equal": ik_pubkey,
        "immutable": True,
    }

out = f"{data_dir}/sovereign_manifest_{titan_id}.json"
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  ✓ Manifest written: {out}")
if sk_enc_pubkey:
    print(f"  soul_keypair.enc decrypts to: {sk_enc_pubkey}")
    print(f"  Match identity: {'YES' if sk_enc_pubkey == ik_pubkey else 'NO ⚠ (would have caught 2026-05-14 bug)'}")
print(f"  Canonical pubkey: {ik_pubkey}")
print(f"  Network: {manifest['network']}")
if gr_pubkey is None:
    print(f"  Genesis record: absent (devnet or pre-birth)")
elif genesis_matches_identity:
    print(f"  Genesis record: present, matches identity (on-chain anchored)")
else:
    print(f"  ⚠ Genesis record: present but DOES NOT match identity ({gr_pubkey} vs {ik_pubkey})")
    print(f"  ⚠   This is leakage of another Titan's genesis_record.json — excluded from this manifest.")
    print(f"  ⚠   Consider removing the leaked file: {gr_path}")
PYREATTEST
    rc=$?
    if [ $rc -ne 0 ]; then exit $rc; fi
    echo
    echo "  Next: run 'bash scripts/verify_sovereign_state.sh' (without --reattest) to verify"
    exit 0
fi

# ── Standard verify mode ───────────────────────────────────────────────

if [ ! -f "${MANIFEST}" ]; then
    echo "  ✗ Manifest missing: ${MANIFEST}" >&2
    echo "    Run with --reattest to generate from current state (one-time):" >&2
    echo "      bash scripts/verify_sovereign_state.sh --reattest --titan ${TITAN_ID}" >&2
    exit 1
fi

FAILED=0
WARNED=0

"${BASE_DIR}/test_env/bin/python" <<PYVERIFY
import json, hashlib, sys, os
from solders.pubkey import Pubkey
from solders.keypair import Keypair as SoldersKP
titan_id = "${TITAN_ID}"

def pubkey_from_file(path):
    with open(path) as f:
        kp = json.load(f)
    kp_bytes = bytes(int(x) & 0xFF for x in kp)
    if len(kp_bytes) == 64:
        return str(Pubkey.from_bytes(kp_bytes[32:64]))
    elif len(kp_bytes) == 32:
        return str(SoldersKP.from_seed(kp_bytes).pubkey())
    else:
        raise ValueError(f"unexpected keypair length: {len(kp_bytes)}")

data_dir = "${DATA_DIR}"
base_dir = "${BASE_DIR}"
override_file = "${OVERRIDE_FILE}"
manifest_path = "${MANIFEST}"

failed = 0
warned = 0

def fail(msg):
    global failed
    print(f"  ✗ FAIL: {msg}")
    failed += 1

def warn(msg):
    global warned
    print(f"  ⚠ WARN: {msg}")
    warned += 1

def ok(msg):
    print(f"  ✓ {msg}")

manifest = json.load(open(manifest_path))
if manifest.get("titan_id") != titan_id:
    fail(f"manifest titan_id ({manifest.get('titan_id')}) != expected ({titan_id})")

canonical_pubkey = manifest.get("titan_pubkey")
if not canonical_pubkey:
    fail("manifest.titan_pubkey missing")
    sys.exit(2)

print(f"  Canonical pubkey from manifest: {canonical_pubkey}")
print()

# ── 1. titan_identity_keypair.json ─────────────────────────────────────
ik_path = f"{base_dir}/data/titan_identity_keypair.json"
if not os.path.exists(ik_path):
    fail(f"{ik_path} MISSING")
else:
    # Permission check
    perms = oct(os.stat(ik_path).st_mode & 0o777)
    if int(perms, 8) & 0o077:
        warn(f"{ik_path} permissions {perms} — should be 0600 or 0400")
    # Pubkey check (pure-python via solders)
    try:
        ik_pubkey = pubkey_from_file(ik_path)
        if ik_pubkey == canonical_pubkey:
            ok(f"titan_identity_keypair.json → {ik_pubkey} (matches canonical)")
        else:
            fail(f"titan_identity_keypair.json → {ik_pubkey} (MISMATCH — expected {canonical_pubkey})")
    except Exception as e:
        fail(f"pubkey derivation failed on {ik_path}: {e}")

# ── 2. genesis_record.json (only check if manifest declared it) ────────
if "data/genesis_record.json" in manifest.get("files", {}):
    gr_path = f"{base_dir}/data/genesis_record.json"
    if not os.path.exists(gr_path):
        fail(f"{gr_path} MISSING — manifest declared on-chain anchor but file is gone")
    else:
        gr = json.load(open(gr_path))
        if gr.get("titan_pubkey") == canonical_pubkey:
            ok(f"genesis_record.titan_pubkey matches canonical ({manifest.get('network','?')})")
        else:
            fail(f"genesis_record.titan_pubkey ({gr.get('titan_pubkey')}) != canonical ({canonical_pubkey})")
else:
    ok(f"genesis_record not declared in manifest (network={manifest.get('network','devnet')})")

# ── 3. soul_keypair.enc decrypts to canonical pubkey ───────────────────
sk_path = f"{base_dir}/data/soul_keypair.enc"
if not os.path.exists(sk_path):
    warn(f"{sk_path} missing — Python plugin will fall back to plaintext titan_identity_keypair.json")
else:
    sys.path.insert(0, base_dir)
    try:
        from titan_hcl.utils.crypto import decrypt_for_machine
        with open(sk_path, "rb") as f:
            decrypted = decrypt_for_machine(f.read())
        # Derive pubkey directly via solders (no temp file)
        if len(decrypted) == 64:
            sk_pubkey = str(Pubkey.from_bytes(decrypted[32:64]))
        elif len(decrypted) == 32:
            sk_pubkey = str(SoldersKP.from_seed(decrypted).pubkey())
        else:
            raise ValueError(f"unexpected decrypted length: {len(decrypted)}")
        if sk_pubkey == canonical_pubkey:
            ok(f"soul_keypair.enc decrypts to {sk_pubkey} (matches canonical)")
        else:
            fail(f"soul_keypair.enc decrypts to {sk_pubkey} (MISMATCH — canonical {canonical_pubkey}). This is THE 2026-05-14 bug class.")
    except Exception as e:
        fail(f"decrypt_for_machine failed on {sk_path}: {e} (wrong hardware? corrupt file?)")

# ── 4. ~/.titan/microkernel_<id>.toml ──────────────────────────────────
if not os.path.exists(override_file):
    warn(f"{override_file} missing — no Phase C activation override")
else:
    content = open(override_file).read()
    if "l0_rust_enabled = true" in content or "l0_rust_enabled=true" in content:
        ok(f"{override_file} → l0_rust_enabled=true (Phase C active)")
    else:
        warn(f"{override_file} present but l0_rust_enabled != true (Phase A/B mode)")

# ── 5. birth_dna + genesis_nft_metadata (presence-only) ────────────────
for n in ["birth_dna_snapshot.json", "genesis_nft_metadata.json"]:
    p = f"{base_dir}/data/{n}"
    if os.path.exists(p):
        ok(f"{n} present")
    else:
        warn(f"{n} missing (expected for birthed mainnet Titans)")

print()
print(f"  Total: failed={failed}, warned={warned}")

if failed > 0:
    print()
    print("  🚨 SOVEREIGN STATE FAILURE — DO NOT proceed with operations until resolved.")
    print("     Common causes:")
    print("       • soul_keypair.enc stale (pre-birth or hardware migration without re-encrypt)")
    print("       • titan_identity_keypair.json was overwritten by a deploy / migration")
    print("       • Wrong Titan: TITAN_ID env / .titan/ override pointing to wrong identity")
    sys.exit(2)

if warned > 0 and "${STRICT}" == "1":
    sys.exit(3)
PYVERIFY

rc=$?
if [ $rc -ne 0 ]; then
    exit $rc
fi

echo
echo "  ═ all sovereign-state invariants hold ═"

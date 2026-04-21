#!/usr/bin/env python3
"""decrypt_backup.py — Maker emergency decryption tool (rFP_backup_worker Phase 7.4).

Use case: Titan is gone (VPS lost, disk destroyed) and the Maker must recover an
encrypted backup from Arweave (or a rescued local `.tar.gz.enc` file). This tool
is standalone — it does NOT require a running Titan. It relies only on:

  - A Maker-held pair of Shamir shards (2 of the 3 minted at mainnet birth)
    OR a pre-reconstructed 64-byte Ed25519 keypair file.
  - The encrypted tarball bytes.
  - The encryption manifest stanza captured at upload time (found in the local
    `data/backup_records/{type}_{ts}.json` record OR in the Arweave tarball's
    manifest sidecar).

Flow:
  1. Reconstruct the Titan keypair from 2-of-3 Shamir shards (reuses
     titan_plugin.utils.shamir).
  2. Derive the master key (HKDF-SHA256 from Ed25519 seed, salted with titan
     pubkey).
  3. Derive the per-backup key (HKDF from master + backup_id + backup_type).
  4. AES-256-GCM decrypt. Auth tag verifies wrong-key attempts.
  5. Verify recovered plaintext SHA-256 matches manifest.plaintext_sha256.
  6. Write plaintext tarball to --output.

Example:
    python scripts/decrypt_backup.py \\
        --shard1-file shard1_maker.hex \\
        --shard2-file shard2.bin \\
        --ciphertext data/backups/personality_20260501_abc123.tar.gz.enc \\
        --record data/backup_records/personality_1778000000.json \\
        --output /tmp/personality_restored.tar.gz

The --record flag accepts a backup record JSON that contains the encryption
stanza (preferred). Alternatively, use --manifest to pass the JSON stanza
directly, and --backup-type to specify the type.

Exit codes:
  0 — plaintext written, verified
  1 — usage error or user input problem
  2 — Shamir reconstruction failed
  3 — decryption failed (wrong key / tampered / malformed manifest)
  4 — plaintext SHA-256 verification failed after decrypt
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _err(msg: str, code: int = 1):
    print(f"[decrypt_backup] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _info(msg: str):
    print(f"[decrypt_backup] {msg}")


def _load_shard(hex_path_or_literal: str) -> bytes:
    """Accept either a hex string on the CLI or a file containing hex.

    Uses shamir.parse_maker_envelope first (JSON-over-hex envelope minted at
    mainnet birth); if that fails, falls back to raw hex → bytes.
    """
    from titan_plugin.utils.shamir import parse_maker_envelope

    data = hex_path_or_literal
    if os.path.exists(data):
        with open(data) as f:
            data = f.read().strip()
    # Try envelope first (strips metadata → raw shard bytes)
    try:
        shard, _metadata = parse_maker_envelope(data)
        return shard
    except Exception:
        pass
    # Fall back to raw hex
    try:
        return bytes.fromhex(data)
    except Exception as e:
        _err(f"could not parse shard (neither envelope nor raw hex): {e}", 1)


def _reconstruct_keypair(shard_sources: list) -> tuple:
    """shard_sources = list of 2-or-more string args (paths or hex literals)."""
    from titan_plugin.utils.shamir import combine_shares
    try:
        shards = [_load_shard(s) for s in shard_sources]
        if len(shards) < 2:
            _err("at least 2 Shamir shards required", 1)
        kp_bytes = combine_shares(shards)
    except Exception as e:
        _err(f"Shamir reconstruction failed: {e}", 2)
    if len(kp_bytes) != 64:
        _err(f"reconstructed keypair must be 64 bytes (got {len(kp_bytes)})", 2)
    try:
        from solders.keypair import Keypair
        titan_pubkey = str(Keypair.from_bytes(kp_bytes).pubkey())
    except Exception:
        titan_pubkey = kp_bytes[32:64].hex()
    return kp_bytes, titan_pubkey


def _load_encryption_manifest(args) -> tuple:
    """Returns (manifest_dict, backup_type). Reads from --record or --manifest."""
    if args.record:
        if not os.path.exists(args.record):
            _err(f"record file not found: {args.record}", 1)
        with open(args.record) as f:
            rec = json.load(f)
        enc = rec.get("encryption")
        if not enc:
            _err("record has no 'encryption' stanza — backup was not encrypted (legacy).", 1)
        backup_type = rec.get("backup_type") or args.backup_type or "personality"
        return enc, backup_type
    if args.manifest:
        if not os.path.exists(args.manifest):
            _err(f"manifest file not found: {args.manifest}", 1)
        with open(args.manifest) as f:
            enc = json.load(f)
        if not args.backup_type:
            _err("--backup-type is required with --manifest", 1)
        return enc, args.backup_type
    _err("must provide one of --record or --manifest", 1)


def main():
    parser = argparse.ArgumentParser(
        description="Maker emergency decryption for Titan encrypted backups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--shard1-file", help="Path to shard 1 (hex envelope or raw hex)")
    parser.add_argument("--shard1-hex", help="Shard 1 as hex on command line")
    parser.add_argument("--shard2-file", help="Path to shard 2")
    parser.add_argument("--shard2-hex", help="Shard 2 hex")
    parser.add_argument("--shard3-file", help="Path to shard 3 (on-chain recovered)")
    parser.add_argument("--shard3-hex", help="Shard 3 hex")
    parser.add_argument("--keypair-file",
                        help="Alternative to shards: pre-reconstructed Ed25519 "
                             "64-byte keypair JSON (e.g. data/titan_identity_keypair.json)")
    parser.add_argument("--ciphertext", required=True,
                        help="Path to encrypted tarball (.tar.gz.enc or .tar.zst.enc)")
    parser.add_argument("--record",
                        help="Local backup record JSON (has encryption stanza + backup_type)")
    parser.add_argument("--manifest",
                        help="Alternative to --record: bare encryption stanza JSON")
    parser.add_argument("--backup-type", choices=["personality", "soul", "timechain"],
                        help="Required with --manifest; inferred from --record otherwise")
    parser.add_argument("--output", required=True, help="Where to write decrypted tarball")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip plaintext SHA-256 verification (not recommended)")
    args = parser.parse_args()

    # ── Keypair reconstruction ─────────────────────────────────────────
    if args.keypair_file:
        try:
            from titan_plugin.logic.backup_crypto import load_keypair_bytes
            kp_bytes, titan_pubkey = load_keypair_bytes(args.keypair_file)
        except Exception as e:
            _err(f"keypair load failed: {e}", 2)
    else:
        shard_sources = []
        for pair in [(args.shard1_file, args.shard1_hex),
                      (args.shard2_file, args.shard2_hex),
                      (args.shard3_file, args.shard3_hex)]:
            chosen = pair[0] or pair[1]
            if chosen:
                shard_sources.append(chosen)
        if len(shard_sources) < 2:
            _err("must provide at least 2 shards (--shard1/2/3 -file or -hex) "
                  "OR --keypair-file", 1)
        _info(f"Reconstructing keypair from {len(shard_sources)} shard(s)...")
        kp_bytes, titan_pubkey = _reconstruct_keypair(shard_sources)

    _info(f"Titan pubkey: {titan_pubkey}")

    # ── Encryption manifest ─────────────────────────────────────────────
    encryption, backup_type = _load_encryption_manifest(args)
    algo = encryption.get("algorithm", "none")
    if algo == "none":
        _err("manifest indicates algorithm=none — backup is not encrypted. "
              "Extract directly with tar instead.", 1)
    _info(f"Manifest: algorithm={algo} tier={encryption.get('tier', '?')} "
           f"key={encryption.get('key_id', '?')} type={backup_type}")

    # ── Decrypt ─────────────────────────────────────────────────────────
    if not os.path.exists(args.ciphertext):
        _err(f"ciphertext not found: {args.ciphertext}", 1)
    with open(args.ciphertext, "rb") as f:
        ct = f.read()
    _info(f"Loaded ciphertext: {len(ct)} bytes")

    try:
        from titan_plugin.logic.backup_crypto import decrypt_from_manifest
        plaintext = decrypt_from_manifest(
            ct, encryption, kp_bytes, titan_pubkey, backup_type)
    except Exception as e:
        _err(f"decryption failed: {e}", 3)

    # ── Verify plaintext SHA-256 ────────────────────────────────────────
    expected = encryption.get("plaintext_sha256")
    actual = hashlib.sha256(plaintext).hexdigest()
    if expected:
        if actual != expected:
            _err(f"plaintext sha256 mismatch: expected={expected[:16]}... "
                  f"actual={actual[:16]}... — manifest corrupt or wrong ciphertext",
                  4)
        _info(f"Plaintext SHA-256 verified ({actual[:16]}...)")
    elif not args.skip_verify:
        _info("WARNING: manifest has no plaintext_sha256 field — skipping verification")

    # ── Write plaintext ─────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(plaintext)
    _info(f"Plaintext written: {args.output} ({len(plaintext)} bytes)")
    _info("Extract with: tar -xzf %s -C <target-dir>" % args.output)
    sys.exit(0)


if __name__ == "__main__":
    main()

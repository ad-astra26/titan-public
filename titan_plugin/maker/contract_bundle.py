"""contract_bundle — disk-only helpers for the Phase C contract bundle.

Computes the deterministic bundle hash from contract JSON files and reads
the on-disk verification state, WITHOUT depending on ContractStore (which
lives in the timechain_worker subprocess). This lets TitanMaker (running
in the main process) autoseed the R8 proposal at boot without IPC.

The hashing algorithm here MUST stay byte-identical to ContractStore's
load_meta_cognitive_contracts() bundle hash — they read the same files
and must produce the same hash, otherwise R8 verification will mismatch.
"""
import hashlib
import json
import os
from typing import Optional


def _default_contracts_dir() -> str:
    """Resolve the canonical contracts directory inside the package."""
    here = os.path.dirname(os.path.abspath(__file__))
    # titan_plugin/maker/contract_bundle.py → titan_plugin/contracts/meta_cognitive/
    return os.path.normpath(
        os.path.join(here, "..", "contracts", "meta_cognitive"))


def compute_bundle_hash_and_names(
    contracts_dir: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Compute the deterministic SHA-256 bundle hash + contract id list.

    Mirrors the bundle hashing inside ContractStore.load_meta_cognitive_contracts()
    so that TitanMaker (main process) can compute the SAME hash as ContractStore
    (timechain_worker subprocess) using only the on-disk JSON files.

    Returns:
        (bundle_hash_hex, contract_names_list)
        ("", []) if the directory doesn't exist or has no JSON contract files.
    """
    if contracts_dir is None:
        contracts_dir = _default_contracts_dir()
    if not os.path.isdir(contracts_dir):
        return "", []
    json_files = sorted(
        f for f in os.listdir(contracts_dir)
        if f.endswith(".json") and not f.startswith(".")
    )
    if not json_files:
        return "", []
    bundle_hasher = hashlib.sha256()
    names: list[str] = []
    for fname in json_files:
        fpath = os.path.join(contracts_dir, fname)
        try:
            with open(fpath, "rb") as f:
                raw = f.read()
            bundle_hasher.update(fname.encode())
            bundle_hasher.update(b"\x00")
            bundle_hasher.update(raw)
            bundle_hasher.update(b"\x00")
            d = json.loads(raw.decode())
            cname = d.get("contract_id", fname.replace(".json", ""))
            names.append(cname)
        except Exception:
            names.append(fname.replace(".json", ""))
    return bundle_hasher.hexdigest(), names


def is_bundle_verified_on_disk(
    contracts_dir: Optional[str] = None,
) -> bool:
    """Check if .bundle_signature.json exists AND its hash matches the live bundle.

    Returns True if R8 ceremony has been completed and the signature is
    still valid for the current bundle. Returns False if no signature exists
    OR if it's stale (bundle changed since signing).
    """
    if contracts_dir is None:
        contracts_dir = _default_contracts_dir()
    sig_path = os.path.join(contracts_dir, ".bundle_signature.json")
    if not os.path.exists(sig_path):
        return False
    try:
        with open(sig_path) as f:
            sig_meta = json.load(f)
        live_hash, _ = compute_bundle_hash_and_names(contracts_dir)
        return sig_meta.get("bundle_hash") == live_hash
    except Exception:
        return False

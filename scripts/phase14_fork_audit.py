#!/usr/bin/env python3
"""Phase 14 (§3K.3.C) — READ-ONLY fork-topology + conversation-routing audit.

Inspects a TimeChain's `index.db` and reports, per chain:
  - the fork_registry topology (id, name, type, tip_height, file path);
  - which of the 6 primary forks are name-resolvable;
  - whether the `conversation` primary exists, and if id 5 is held by a
    non-conversation fork (the T2/T3 divergence);
  - a heuristic count of conversation-like blocks (verified-output / chat tags)
    currently sitting in WHICHEVER fork id 5 resolves to — to size any
    misrouting before deciding on an additive re-tag.

NO writes. NO Titan load. Safe to run against live mainnet data.

Usage:
    python scripts/phase14_fork_audit.py [--data-dir data/timechain]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

PRIMARY_NAMES = ["main", "declarative", "procedural", "episodic", "meta",
                 "conversation"]
# Heuristic markers of an outer-chat / verified-output TX (arch §7).
CONVO_TAGS = ("verified_output", "chat", "conversation", "user:", "telegram")
# Canonical id→name map (mirrors timechain.FORK_NAMES — inlined so this audit
# stays a zero-dependency read-only tool runnable from any cwd).
_CANONICAL_FORK_NAMES = {
    0: "main", 1: "declarative", 2: "procedural", 3: "episodic",
    4: "meta", 5: "conversation", 100: "system",
}


def _chain_file(data_dir: Path, fork_id: int, fork_name: str | None) -> str:
    # Mirrors TimeChain._get_chain_file_path: primaries → chain_<name>.bin
    # keyed off the canonical id→name map; else sidechains/sc_NNNN.bin.
    name = _CANONICAL_FORK_NAMES.get(fork_id)
    if name:
        p = data_dir / f"chain_{name}.bin"
    else:
        p = data_dir / "sidechains" / f"sc_{fork_id:04d}.bin"
    return f"{p.name} ({'exists' if p.exists() else 'MISSING'})"


def audit(data_dir: Path) -> int:
    db = data_dir / "index.db"
    if not db.exists():
        print(f"[audit] no index.db at {db}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT fork_id, fork_name, fork_type, tip_height, topic "
            "FROM fork_registry ORDER BY fork_id"
        ).fetchall()
        name_to_id = {r["fork_name"]: r["fork_id"] for r in rows}
        id_to_name = {r["fork_id"]: r["fork_name"] for r in rows}

        print(f"\n=== Fork topology — {data_dir} ===")
        print(f"{'id':>5} {'name':<22} {'type':<16} {'tip':>8}  file")
        for r in rows:
            print(f"{r['fork_id']:>5} {r['fork_name']:<22} {r['fork_type']:<16} "
                  f"{r['tip_height']:>8}  {_chain_file(data_dir, r['fork_id'], r['fork_name'])}")

        print("\n--- Primary fork name-resolvability (INV-Syn-26) ---")
        all_ok = True
        for name in PRIMARY_NAMES:
            fid = name_to_id.get(name)
            ok = fid is not None
            all_ok = all_ok and ok
            print(f"  {name:<14} -> {'id ' + str(fid) if ok else 'MISSING'}"
                  f"{'  [reserved canonical 5]' if name == 'conversation' else ''}")

        # The divergence check.
        occupant_of_5 = id_to_name.get(5)
        print("\n--- id-5 occupancy ---")
        print(f"  id 5 = {occupant_of_5!r}")
        conv_id = name_to_id.get("conversation")
        if conv_id == 5:
            print("  ✅ conversation IS the canonical fork 5 (T1-correct topology).")
        elif conv_id is not None:
            print(f"  ⚠ conversation RELOCATED to id {conv_id}; id 5 held by "
                  f"{occupant_of_5!r}.")
        else:
            print(f"  🔴 conversation fork ABSENT; id 5 held by {occupant_of_5!r} "
                  "→ reseed will materialize it at the next free id.")

        # Size conversation-like content sitting in whatever fork 5 is.
        if occupant_of_5 is not None:
            total5 = conn.execute(
                "SELECT COUNT(*) FROM block_index WHERE fork_id = 5"
            ).fetchone()[0]
            like = 0
            for (tags,) in conn.execute(
                "SELECT tags FROM block_index WHERE fork_id = 5 AND tags != ''"
            ).fetchall():
                t = (tags or "").lower()
                if any(m in t for m in CONVO_TAGS):
                    like += 1
            print(f"\n--- conversation-like content in fork 5 ({occupant_of_5}) ---")
            print(f"  total blocks in fork 5: {total5}")
            print(f"  conversation-like (tag heuristic {CONVO_TAGS}): {like}")
            if conv_id != 5 and like > 0:
                print(f"  ⚠ {like} conversation-like blocks are commingled in fork 5 — "
                      "candidates for an ADDITIVE cross-ref (never delete/move).")
        return 0 if all_ok else 1
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/timechain")
    args = ap.parse_args()
    sys.exit(audit(Path(args.data_dir)))


if __name__ == "__main__":
    main()

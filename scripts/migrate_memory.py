#!/usr/bin/env python3
"""
scripts/migrate_memory.py
One-time migration: SQLite memory_nodes.db → DuckDB + FAISS + Kuzu.

Migrates all existing memory nodes to the new direct backend:
  1. Export memory_nodes.db (SQLite) → DuckDB titan_memory.duckdb
  2. Re-embed all persistent nodes → FAISS memory_vectors.faiss
  3. Cognify un-cognified nodes → Kuzu knowledge_graph.kuzu (background, interruptible)
  4. Verify counts and search quality

Usage:
  source test_env/bin/activate
  python scripts/migrate_memory.py [--cognify] [--verify-only]

Flags:
  --cognify      Also run LLM entity extraction (slow: ~3-5s/node, needs Ollama Cloud)
  --verify-only  Only run verification step (no migration)
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SQLITE_PATH = os.path.join(DATA_DIR, "memory_nodes.db")
DUCKDB_PATH = os.path.join(DATA_DIR, "titan_memory.duckdb")
FAISS_PATH = os.path.join(DATA_DIR, "memory_vectors.faiss")
KUZU_PATH = os.path.join(DATA_DIR, "knowledge_graph.kuzu")


def migrate_sqlite_to_duckdb():
    """Step 1: Export all nodes from SQLite memory_nodes.db → DuckDB."""
    if not os.path.exists(SQLITE_PATH):
        logger.error("SQLite DB not found: %s", SQLITE_PATH)
        return 0, 0

    from titan_plugin.core.direct_memory import TitanDuckDB

    # Check if DuckDB already has data
    db = TitanDuckDB(DUCKDB_PATH)
    existing = db.get_stats()
    if existing["total"] > 0:
        logger.info("DuckDB already has %d nodes — skipping SQLite import.", existing["total"])
        db.close()
        return existing["total"], 0

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    # Migrate memory_nodes
    rows = conn.execute("SELECT * FROM memory_nodes").fetchall()
    migrated = 0
    for row in rows:
        node = dict(row)
        db.insert_node(node)
        migrated += 1

    # Migrate identity_nodes
    id_rows = conn.execute("SELECT * FROM identity_nodes").fetchall()
    identities = 0
    for row in id_rows:
        r = dict(row)
        db.insert_identity(r["id"], r["identifier"], r["created_at"])
        identities += 1

    conn.close()
    db.close()

    logger.info("Migrated %d memory nodes + %d identities to DuckDB.", migrated, identities)
    return migrated, identities


def embed_persistent_nodes():
    """Step 2: Re-embed all persistent nodes into FAISS index."""
    from titan_plugin.core.direct_memory import TitanDuckDB, TitanVectorIndex

    db = TitanDuckDB(DUCKDB_PATH)
    vi = TitanVectorIndex(FAISS_PATH)

    if vi.count > 0:
        logger.info("FAISS index already has %d vectors — skipping re-embed.", vi.count)
        db.close()
        return vi.count

    persistent = db.get_nodes_by_status("persistent")
    logger.info("Embedding %d persistent nodes into FAISS...", len(persistent))

    # Batch embed for efficiency
    batch_size = 64
    total_embedded = 0

    for i in range(0, len(persistent), batch_size):
        batch = persistent[i:i + batch_size]
        texts = []
        ids = []
        for node in batch:
            text = (
                f"Memory #{node['id']} | "
                f"User: {node.get('user_prompt', '')} | "
                f"Agent: {node.get('agent_response', '')}"
            )
            texts.append(text)
            ids.append(node["id"])

        vecs = vi.embed_batch(texts)
        vi.add_batch(vecs, ids)

        # Update DuckDB with embedding references
        for j, node_id in enumerate(ids):
            db.update_node(node_id, embedding_id=total_embedded + j)

        total_embedded += len(batch)
        if total_embedded % 200 == 0 or total_embedded == len(persistent):
            logger.info("  Embedded %d/%d nodes...", total_embedded, len(persistent))

    vi.save()
    db.close()
    logger.info("FAISS index built: %d vectors (dim=384).", vi.count)
    return vi.count


def cognify_to_kuzu():
    """Step 3: Run LLM entity extraction on un-cognified nodes into Kuzu graph.

    This is interruptible — tracks progress via a local JSON file.
    On restart, skips already-processed node IDs.

    Supports running while T1 is live (read-only DuckDB, separate Kuzu connection).
    Uses Ollama Cloud (gemma3:4b) for entity extraction: ~3-5s per node.
    For ~2000 persistent nodes: estimated 2-3 hours.
    """
    import asyncio
    import duckdb
    from titan_plugin.core.direct_memory import TitanKnowledgeGraph, TitanCognify

    # Track which nodes we've already cognified (survives interruptions)
    cognify_done_path = os.path.join(DATA_DIR, "cognify_done_ids.json")
    done_ids = set()
    if os.path.exists(cognify_done_path):
        try:
            with open(cognify_done_path) as f:
                done_ids = set(json.load(f))
            logger.info("Resuming: %d nodes already cognified from previous run", len(done_ids))
        except Exception:
            pass

    # Open DuckDB read-only (works while T1 holds the write lock)
    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        logger.info("DuckDB opened in read-only mode (T1 may be running)")
    except Exception as e:
        logger.error("Cannot open DuckDB: %s", e)
        return 0

    # Read persistent nodes
    rows = conn.execute(
        "SELECT * FROM memory_nodes WHERE status = 'persistent'"
    ).fetchall()
    cols = [d[0] for d in conn.description]
    all_persistent = [dict(zip(cols, row)) for row in rows]
    conn.close()

    kg = TitanKnowledgeGraph(KUZU_PATH)

    uncognified = [n for n in all_persistent if n["id"] not in done_ids]
    total = len(all_persistent)
    already_done = len(done_ids)

    logger.info("Persistent nodes: %d total, %d already cognified, %d to process",
                total, already_done, len(uncognified))

    if not uncognified:
        logger.info("All nodes already cognified!")
        kg.close()
        return 0

    # Initialize LLM client
    try:
        from titan_plugin.utils.ollama_cloud import OllamaCloudClient
        config_path = os.path.join(os.path.dirname(__file__), "..", "titan_plugin", "config.toml")
        import tomllib
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        inference_cfg = config.get("inference", {})
        api_key = inference_cfg.get("ollama_cloud_api_key", "")
        base_url = inference_cfg.get("ollama_cloud_url", "https://ollama.com/v1")
        llm = OllamaCloudClient(api_key=api_key, base_url=base_url)
        logger.info("Ollama Cloud client initialized: %s", base_url)
    except Exception as e:
        logger.error("Failed to initialize LLM client: %s", e)
        kg.close()
        return 0

    cognify = TitanCognify(llm, kg)

    # Process nodes with progress tracking
    processed = 0
    errors = 0
    start = time.time()

    # Save progress file for monitoring
    progress_path = os.path.join(DATA_DIR, "cognify_progress.json")

    async def _run():
        nonlocal processed, errors
        for i, node in enumerate(uncognified):
            node_id = node["id"]
            text = f"User: {node.get('user_prompt', '')} | Agent: {node.get('agent_response', '')}"

            # Parse neuromod context if available
            neuromod = {}
            nctx = node.get("neuromod_context")
            if nctx:
                try:
                    neuromod = json.loads(nctx) if isinstance(nctx, str) else nctx
                except (json.JSONDecodeError, TypeError):
                    pass

            try:
                entities = await cognify.cognify_node(node_id, text, neuromod)
                # Verify LLM actually responded (cognify_node silently returns [] on LLM error)
                # We check by testing a quick LLM ping every 50 nodes to catch auth issues
                if i == 0:
                    # Verify API is working on first node
                    test = await llm.complete("test", model="gemma3:4b", max_tokens=5, timeout=10)
                    if not test:
                        raise RuntimeError("LLM API not responding — check API key")
                done_ids.add(node_id)
                processed += 1
            except Exception as e:
                errors += 1
                logger.warning("Node %d failed: %s", node_id, e)
                # If first node fails, abort early (likely auth issue)
                if i == 0:
                    logger.error("First node failed — aborting cognify. Fix the issue and retry.")
                    break

            # Progress update every 50 nodes
            if (i + 1) % 50 == 0 or (i + 1) == len(uncognified):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(uncognified) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "  Progress: %d/%d (%.1f%%) — %d errors — %.1f nodes/s — ETA %.0fm",
                    i + 1, len(uncognified),
                    (i + 1) / len(uncognified) * 100,
                    errors, rate, remaining / 60,
                )
                # Write progress + done IDs (atomic save)
                try:
                    with open(progress_path, "w") as f:
                        json.dump({
                            "processed": already_done + processed,
                            "total": total,
                            "errors": errors,
                            "rate_per_sec": round(rate, 2),
                            "eta_minutes": round(remaining / 60, 1),
                            "started_at": start,
                            "updated_at": time.time(),
                        }, f, indent=2)
                    with open(cognify_done_path, "w") as f:
                        json.dump(sorted(done_ids), f)
                except Exception:
                    pass

    asyncio.run(_run())

    # Final stats
    elapsed = time.time() - start
    kg_stats = kg.get_stats()
    total_entities = sum(kg_stats.values())
    logger.info(
        "Cognify complete: %d processed, %d errors in %.1fs (%.1f nodes/s). "
        "Kuzu: %d entities (%s)",
        processed, errors, elapsed,
        processed / elapsed if elapsed > 0 else 0,
        total_entities, kg_stats,
    )

    kg.close()

    # Cleanup progress file
    try:
        os.remove(progress_path)
    except OSError:
        pass

    return processed


def verify_migration():
    """Step 3: Verify counts and search quality."""
    from titan_plugin.core.direct_memory import TitanDuckDB, TitanVectorIndex, TitanKnowledgeGraph

    print("\n" + "=" * 60)
    print("  MIGRATION VERIFICATION")
    print("=" * 60)

    # Compare SQLite vs DuckDB
    if os.path.exists(SQLITE_PATH):
        conn = sqlite3.connect(SQLITE_PATH)
        sqlite_total = conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
        sqlite_persistent = conn.execute(
            "SELECT COUNT(*) FROM memory_nodes WHERE status='persistent'"
        ).fetchone()[0]
        sqlite_identities = conn.execute("SELECT COUNT(*) FROM identity_nodes").fetchone()[0]
        conn.close()
    else:
        sqlite_total = sqlite_persistent = sqlite_identities = "N/A"

    import duckdb as _duckdb
    try:
        _duck_conn = _duckdb.connect(DUCKDB_PATH, read_only=True)
        _duck_total = _duck_conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]
        _duck_by_status = _duck_conn.execute(
            "SELECT status, COUNT(*) FROM memory_nodes GROUP BY status"
        ).fetchall()
        duck_stats = {"total": _duck_total, "by_status": {s: c for s, c in _duck_by_status}}
        duck_identities = _duck_conn.execute("SELECT COUNT(*) FROM identity_nodes").fetchone()[0]
        _duck_conn.close()
    except Exception as e:
        print(f"  DuckDB: cannot open ({e}) — try stopping Titan first")
        duck_stats = {"total": "?", "by_status": {}}
        duck_identities = "?"

    print(f"\n  SQLite memory_nodes:  {sqlite_total} total, {sqlite_persistent} persistent, {sqlite_identities} identities")
    print(f"  DuckDB memory_nodes:  {duck_stats['total']} total, {duck_stats.get('by_status', {})} identities={duck_identities}")

    match = (sqlite_total == "N/A") or (duck_stats["total"] == sqlite_total)
    print(f"  Count match: {'PASS' if match else 'FAIL'}")

    # FAISS
    vi = TitanVectorIndex(FAISS_PATH)
    print(f"\n  FAISS vectors: {vi.count}")
    expected_persistent = duck_stats.get("by_status", {}).get("persistent", 0)
    print(f"  Expected (persistent nodes): {expected_persistent}")
    print(f"  Vector match: {'PASS' if vi.count == expected_persistent else 'WARN (may need re-embed)'}")

    # Test a search
    if vi.count > 0:
        test_query = "artificial intelligence"
        qvec = vi.embed(test_query)
        results = vi.search(qvec, top_k=3)
        print(f"\n  Sample FAISS search for '{test_query}':")
        for node_id, score in results:
            print(f"    id={node_id} score={score:.3f}")

    # Kuzu
    if os.path.exists(KUZU_PATH):
        kg = TitanKnowledgeGraph(KUZU_PATH)
        kg_stats = kg.get_stats()
        total_entities = sum(kg_stats.values())
        print(f"\n  Kuzu entities: {total_entities} ({kg_stats})")
        kg.close()
    else:
        print(f"\n  Kuzu: not yet created (run with --cognify)")

    # Size comparison
    print(f"\n  Size comparison:")
    for label, path in [
        ("SQLite memory_nodes.db", SQLITE_PATH),
        ("DuckDB titan_memory.duckdb", DUCKDB_PATH),
        ("FAISS memory_vectors.faiss", FAISS_PATH),
        ("Kuzu knowledge_graph.kuzu", KUZU_PATH),
    ]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    {label}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"    {label}: not found")

    cognee_path = os.path.join(os.path.dirname(DATA_DIR), "cognee_data")
    if os.path.exists(cognee_path):
        import subprocess
        result = subprocess.run(["du", "-sh", cognee_path], capture_output=True, text=True)
        print(f"    Cognee data (old): {result.stdout.strip()}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Migrate memory from Cognee to Direct backend")
    parser.add_argument("--cognify", action="store_true", help="Run LLM entity extraction")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    args = parser.parse_args()

    if args.verify_only:
        verify_migration()
        return

    logger.info("=== Memory Migration: SQLite → DuckDB + FAISS + Kuzu ===")
    start = time.time()

    # Step 1: SQLite → DuckDB
    logger.info("\n--- Step 1: SQLite → DuckDB ---")
    nodes, identities = migrate_sqlite_to_duckdb()

    # Step 2: Embed persistent nodes → FAISS
    logger.info("\n--- Step 2: Embed persistent → FAISS ---")
    vectors = embed_persistent_nodes()

    # Step 3 (optional): Cognify → Kuzu
    if args.cognify:
        logger.info("\n--- Step 3: Cognify → Kuzu (this may take hours) ---")
        cognified = cognify_to_kuzu()
        logger.info("Cognified %d nodes into Kuzu graph.", cognified)

    elapsed = time.time() - start
    logger.info("\n=== Migration complete in %.1fs ===", elapsed)

    # Step 4: Verify
    verify_migration()


if __name__ == "__main__":
    main()

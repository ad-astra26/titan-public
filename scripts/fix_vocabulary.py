#!/usr/bin/env python3
"""Phase 1a: Vocabulary cleanup migration script.

Fixes misclassified words, removes leaked slot tags and markdown artifacts.
Safe to run on live Titans (SQLite WAL mode).

Usage:
    source test_env/bin/activate
    python scripts/fix_vocabulary.py [--db data/inner_memory.db] [--dry-run]
"""
import argparse
import re
import shutil
import sqlite3
import sys
import time

# ── Reclassification map (word -> correct type) ────────────────────────
RECLASSIFY = {
    "you": "pronoun",
    "more": "adverb",
    "okay": "interjection",
    "new": "adjective",
    "please": "interjection",
    "ideas": "noun",
    "sad": "adjective",
    "maybe": "adverb",
    "good": "adjective",
    "kind": "adjective",
    "the": "determiner",
    "makes": "verb",
    "feels": "verb",
    "like": "verb",
    "things": "noun",
    "need": "verb",
    "people": "noun",
    "because": "conjunction",
    "also": "adverb",
    "together": "adverb",
    "still": "adverb",
}

# Patterns to DELETE entirely
SLOT_TAG_PATTERN = re.compile(r'^(adj|verb|noun|adv)\d*$', re.IGNORECASE)
MARKDOWN_PATTERN = re.compile(r'^\*\*.*\*\*$')
MARKDOWN_PARTIAL = re.compile(r'^\*\*\w+$|^\w+\*\*$')  # **word or word**


def main():
    parser = argparse.ArgumentParser(description="Vocabulary cleanup migration")
    parser.add_argument("--db", default="data/inner_memory.db", help="DB path")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify")
    args = parser.parse_args()

    db_path = args.db
    dry_run = args.dry_run

    # Backup
    if not dry_run:
        backup = f"{db_path}.bak_{int(time.time())}"
        shutil.copy2(db_path, backup)
        print(f"Backup: {backup}")

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")

    # ── Step 1: Reclassify mistyped words ──────────────────────────────
    print("\n=== RECLASSIFICATION ===")
    reclassified = 0
    for word, correct_type in RECLASSIFY.items():
        row = conn.execute(
            "SELECT word, word_type FROM vocabulary WHERE word = ?", (word,)
        ).fetchone()
        if row and row[1] != correct_type:
            print(f"  {word}: {row[1]} -> {correct_type}")
            if not dry_run:
                conn.execute(
                    "UPDATE vocabulary SET word_type = ? WHERE word = ?",
                    (correct_type, word))
            reclassified += 1
        elif row and row[1] == correct_type:
            print(f"  {word}: already correct ({correct_type})")

    # ── Step 2: Delete leaked slot tags and markdown artifacts ─────────
    print("\n=== DELETION (garbage entries) ===")
    all_words = conn.execute("SELECT word FROM vocabulary").fetchall()
    deleted = 0
    for (word,) in all_words:
        reason = None
        if SLOT_TAG_PATTERN.match(word):
            reason = "leaked slot tag"
        elif MARKDOWN_PATTERN.match(word):
            reason = "markdown artifact"
        elif MARKDOWN_PARTIAL.match(word):
            reason = "partial markdown"
        elif len(word) < 2 and word.lower() not in ("i", "a"):
            reason = "single char"

        if reason:
            print(f"  DELETE '{word}' ({reason})")
            if not dry_run:
                conn.execute("DELETE FROM vocabulary WHERE word = ?", (word,))
            deleted += 1

    # ── Step 3: Check for other suspicious entries ─────────────────────
    print("\n=== SUSPICIOUS (review manually) ===")
    suspicious = conn.execute(
        "SELECT word, word_type, confidence, times_produced "
        "FROM vocabulary WHERE word_type = 'verb' "
        "ORDER BY times_produced DESC LIMIT 30"
    ).fetchall()
    for word, wtype, conf, prod in suspicious:
        # Flag verbs that probably aren't verbs
        if word in RECLASSIFY:
            continue  # Already handled
        if word.endswith(("ness", "ment", "tion", "sion", "ity")):
            print(f"  SUSPECT noun-as-verb: '{word}' (type={wtype}, conf={conf:.2f}, prod={prod})")
        elif word in {"also", "very", "never", "always", "often", "still", "together",
                      "inside", "outside", "here", "there", "now", "then", "maybe"}:
            print(f"  SUSPECT adverb-as-verb: '{word}' (type={wtype}, conf={conf:.2f}, prod={prod})")

    if not dry_run:
        conn.commit()

    # ── Summary ────────────────────────────────────────────────────────
    total = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
    conn.close()

    print(f"\n=== SUMMARY {'(DRY RUN)' if dry_run else ''} ===")
    print(f"  Reclassified: {reclassified}")
    print(f"  Deleted: {deleted}")
    print(f"  Total vocabulary after: {total}")


if __name__ == "__main__":
    main()

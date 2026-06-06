#!/usr/bin/env python3
"""
Merge and deduplicate Mina corpus files.
Reads parallel_corpus.jsonl and corpus_enriched.jsonl (if exists),
removes duplicates based on French text (case-insensitive),
and saves the merged result.
"""

import json
import os
from pathlib import Path

# File paths
BASE_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
CORPUS_DIR = BASE_DIR / "data" / "corpus"
PARALLEL_FILE = CORPUS_DIR / "parallel_corpus.jsonl"
ENRICHED_FILE = CORPUS_DIR / "corpus_enriched.jsonl"
OUTPUT_FILE = CORPUS_DIR / "corpus_merged.jsonl"

def normalize_entry(entry):
    """Normalize entry to have consistent field names."""
    # Handle both formats
    fr_text = entry.get("french") or entry.get("fr", "")
    mina_text = entry.get("mina", "")

    # Build normalized entry
    normalized = {
        "fr": fr_text,
        "mina": mina_text,
        "domain": entry.get("domain", ""),
    }

    # Preserve original fields if they exist
    if "audio" in entry:
        normalized["audio"] = entry["audio"]
    if "timestamp" in entry:
        normalized["timestamp"] = entry["timestamp"]

    return normalized

def load_jsonl(file_path):
    """Load JSONL file and return list of entries."""
    entries = []
    if not os.path.exists(file_path):
        print(f"  [INFO] File not found: {file_path}")
        return entries

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Invalid JSON line: {e}")
    return entries

def save_jsonl(file_path, entries):
    """Save entries to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def deduplicate(entries):
    """Remove duplicates based on case-insensitive French text."""
    seen = {}  # lowercase french -> entry (preserves first occurrence)

    for entry in entries:
        fr_lower = entry.get("fr", "").lower().strip()
        if fr_lower and fr_lower not in seen:
            seen[fr_lower] = entry

    return list(seen.values())

def main():
    print("=" * 60)
    print("Mina Corpus Merge & Deduplication")
    print("=" * 60)

    all_entries = []

    # Load parallel_corpus.jsonl
    print(f"\n1. Loading: {PARALLEL_FILE}")
    parallel_entries = load_jsonl(PARALLEL_FILE)
    print(f"   Found {len(parallel_entries)} entries")

    for entry in parallel_entries:
        normalized = normalize_entry(entry)
        if normalized["fr"]:
            all_entries.append(normalized)

    # Load corpus_enriched.jsonl if exists
    print(f"\n2. Loading: {ENRICHED_FILE}")
    if os.path.exists(ENRICHED_FILE):
        enriched_entries = load_jsonl(ENRICHED_FILE)
        print(f"   Found {len(enriched_entries)} entries")

        for entry in enriched_entries:
            normalized = normalize_entry(entry)
            if normalized["fr"]:
                all_entries.append(normalized)
    else:
        print("   [SKIP] File does not exist")

    # Deduplicate
    print(f"\n3. Deduplicating...")
    total_before = len(all_entries)
    print(f"   Entries before deduplication: {total_before}")

    deduped_entries = deduplicate(all_entries)
    total_after = len(deduped_entries)
    duplicates_removed = total_before - total_after

    print(f"   Entries after deduplication: {total_after}")
    print(f"   Duplicates removed: {duplicates_removed}")

    # Save merged corpus
    print(f"\n4. Saving to: {OUTPUT_FILE}")
    save_jsonl(OUTPUT_FILE, deduped_entries)
    print(f"   Saved {len(deduped_entries)} entries")

    # Print summary
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"  Total lines before deduplication: {total_before}")
    print(f"  Total lines after deduplication:  {total_after}")
    print(f"  Duplicates removed:               {duplicates_removed}")
    print(f"  Output file:                     {OUTPUT_FILE}")
    print("=" * 60)

    # Show sample of merged corpus
    print("\nSample entries from merged corpus:")
    for i, entry in enumerate(deduped_entries[:3]):
        print(f"  [{i+1}] FR: {entry['fr'][:50]}...")
        print(f"      MINA: {entry['mina'][:50]}...")
        if entry.get('domain'):
            print(f"      DOMAIN: {entry['domain']}")
        print()

if __name__ == "__main__":
    main()
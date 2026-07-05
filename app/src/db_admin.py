"""
DocMind — Database Admin Script
---------------------------------
Personal utility for manual control over ChromaDB.
Completely standalone — no backend needed, runs directly against ChromaDB Cloud.
Works in development and against production (reads from .env).

Commands:
    python -m app.src.db_admin                          # show DB stats
    python -m app.src.db_admin --wipe                   # wipe entire collection
    python -m app.src.db_admin --wipe-session <id>      # wipe one session
    python -m app.src.db_admin --sessions               # inspect all sessions
    python -m app.src.db_admin --clean-orphans <ids>    # clean orphaned chunks
"""

import argparse
import os
import sys
from collections import defaultdict

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME = "docmind-research"

if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
    print("❌  Missing ChromaDB config in .env — check CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE.")
    sys.exit(1)

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _build_session_map(collection) -> dict:
    """Build a map of session_id → {papers, chunk_count} from ChromaDB metadata."""
    results  = collection.get(include=["metadatas"])
    sessions = defaultdict(lambda: {"papers": set(), "chunk_count": 0})

    for meta in results["metadatas"]:
        sid   = meta.get("session_id", "unknown")
        title = meta.get("title", "unknown")
        sessions[sid]["papers"].add(title)
        sessions[sid]["chunk_count"] += 1

    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATS
# ─────────────────────────────────────────────────────────────────────────────

def show_stats():
    """Show high-level DB stats."""
    collection = get_collection()
    total      = collection.count()

    print(f"\n📊  Collection : {COLLECTION_NAME}")
    print(f"    Total chunks: {total}")

    if total == 0:
        print("    Database is empty.\n")
        return

    sessions = _build_session_map(collection)
    print(f"    Sessions    : {len(sessions)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SESSION INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────

def inspect_sessions():
    """List all sessions with their papers and chunk counts."""
    collection = get_collection()
    total      = collection.count()

    if total == 0:
        print("\n✅  Database is empty — no active sessions.\n")
        return

    sessions = _build_session_map(collection)

    print(f"\n🔍  Active sessions in ChromaDB ({len(sessions)} total)\n")
    print(f"  {'SESSION ID':<40} {'CHUNKS':>6}  PAPERS")
    print(f"  {'-'*40} {'-'*6}  {'-'*50}")

    for sid, data in sessions.items():
        papers      = sorted(data["papers"])
        chunk_count = data["chunk_count"]
        print(f"\n  {sid:<40} {chunk_count:>6}")
        for paper in papers:
            print(f"    └─ {paper[:75]}")

    print(f"\n  Total chunks: {total}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. WIPE ENTIRE COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def wipe_all():
    """Wipe every chunk from the collection."""
    collection = get_collection()
    total      = collection.count()

    if total == 0:
        print("\n✅  Database is already empty.\n")
        return

    show_stats()
    confirm = input(f"⚠️   This will delete ALL {total} chunks across all sessions.\n    Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("    Aborted.\n")
        return

    all_ids = collection.get(include=[])["ids"]
    collection.delete(ids=all_ids)
    print(f"\n✅  Wiped {total} chunks from ChromaDB.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. WIPE ONE SESSION
# ─────────────────────────────────────────────────────────────────────────────

def wipe_session(session_id: str):
    """Wipe all chunks belonging to a specific session."""
    collection = get_collection()

    results = collection.get(
        where={"session_id": session_id},
        include=["metadatas"],
    )
    ids    = results["ids"]
    papers = {m.get("title", "unknown") for m in results["metadatas"]}

    if not ids:
        print(f"\n❌  No chunks found for session: {session_id}\n")
        return

    print(f"\n🗑️   Session   : {session_id}")
    print(f"    Chunks    : {len(ids)}")
    print(f"    Papers    :")
    for p in sorted(papers):
        print(f"      └─ {p[:75]}")

    confirm = input(f"\n    Delete {len(ids)} chunks? Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("    Aborted.\n")
        return

    collection.delete(ids=ids)
    print(f"\n✅  Deleted {len(ids)} chunks for session {session_id}.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLEAN ORPHANED CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def clean_orphans(active_session_ids: list[str]):
    """
    Delete chunks whose session_id is NOT in the provided list of active sessions.
    Use this when the server has restarted and in-memory sessions are gone
    but ChromaDB still holds their chunks.

    Pass the session IDs you want to KEEP. Everything else gets deleted.

    Example:
        python -m app.src.db_admin --clean-orphans abc123 def456
    """
    collection = get_collection()
    total      = collection.count()

    if total == 0:
        print("\n✅  Database is empty — nothing to clean.\n")
        return

    sessions = _build_session_map(collection)
    orphan_sessions = {
        sid: data for sid, data in sessions.items()
        if sid not in active_session_ids
    }

    if not orphan_sessions:
        print("\n✅  No orphaned sessions found.\n")
        return

    orphan_chunk_count = sum(d["chunk_count"] for d in orphan_sessions.values())

    print(f"\n🧹  Orphaned sessions found: {len(orphan_sessions)}")
    print(f"    Orphaned chunks        : {orphan_chunk_count}\n")

    for sid, data in orphan_sessions.items():
        print(f"  Session: {sid}")
        for p in sorted(data["papers"]):
            print(f"    └─ {p[:75]}")
        print()

    confirm = input(f"    Delete {orphan_chunk_count} orphaned chunks? Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("    Aborted.\n")
        return

    # Fetch and delete orphan IDs
    deleted = 0
    for sid in orphan_sessions:
        results = collection.get(
            where={"session_id": sid},
            include=[],
        )
        ids = results["ids"]
        if ids:
            collection.delete(ids=ids)
            deleted += len(ids)

    print(f"\n✅  Deleted {deleted} orphaned chunks.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DocMind DB Admin — manual ChromaDB control.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Wipe the entire collection (all sessions).",
    )
    parser.add_argument(
        "--wipe-session",
        metavar="SESSION_ID",
        help="Wipe chunks for a specific session ID.",
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="Inspect all sessions currently in ChromaDB.",
    )
    parser.add_argument(
        "--clean-orphans",
        nargs="+",
        metavar="SESSION_ID",
        help="Delete chunks for sessions NOT in this list (orphan cleanup).",
    )

    args = parser.parse_args()

    if args.wipe:
        wipe_all()
    elif args.wipe_session:
        wipe_session(args.wipe_session)
    elif args.sessions:
        inspect_sessions()
    elif args.clean_orphans:
        clean_orphans(active_session_ids=args.clean_orphans)
    else:
        show_stats()
        print("  Run with --help to see all commands.\n")
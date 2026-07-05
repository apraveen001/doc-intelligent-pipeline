"""
DocMind — Database Control Script
-----------------------------------
A personal utility to inspect and wipe the ChromaDB collection.
Run this locally anytime — works in dev and against production.

Usage:
    python -m app.src.check_db           # show DB stats
    python -m app.src.check_db --wipe    # wipe entire collection
    python -m app.src.check_db --session <session_id>  # wipe one session
"""

import argparse
import os
import sys

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME = "docmind-research"

if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
    print("❌ Missing ChromaDB config in .env")
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


def show_stats():
    """Show current DB state."""
    collection = get_collection()
    total = collection.count()
    print(f"\n📊 Collection: {COLLECTION_NAME}")
    print(f"   Total chunks: {total}")

    if total == 0:
        print("   Database is empty.\n")
        return

    # Show unique sessions
    results = collection.get(include=["metadatas"])
    sessions = {}
    for meta in results["metadatas"]:
        sid = meta.get("session_id", "unknown")
        title = meta.get("title", "unknown")
        if sid not in sessions:
            sessions[sid] = set()
        sessions[sid].add(title)

    print(f"   Active sessions: {len(sessions)}\n")
    for sid, titles in sessions.items():
        print(f"   Session: {sid[:8]}...")
        for t in titles:
            print(f"     └─ {t[:70]}")
    print()


def wipe_all():
    """Wipe the entire collection."""
    collection = get_collection()
    total = collection.count()

    if total == 0:
        print("✅ Database is already empty.")
        return

    confirm = input(f"⚠️  This will delete ALL {total} chunks. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    all_ids = collection.get(include=[])["ids"]
    collection.delete(ids=all_ids)
    print(f"✅ Wiped {total} chunks from ChromaDB.")


def wipe_session(session_id: str):
    """Wipe chunks for a specific session."""
    collection = get_collection()

    results = collection.get(
        where={"session_id": session_id},
        include=[],
    )
    ids = results["ids"]

    if not ids:
        print(f"❌ No chunks found for session: {session_id}")
        return

    confirm = input(f"⚠️  Delete {len(ids)} chunks for session {session_id[:8]}...? Type 'yes': ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    collection.delete(ids=ids)
    print(f"✅ Deleted {len(ids)} chunks for session {session_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocMind DB control script.")
    parser.add_argument("--wipe",    action="store_true", help="Wipe entire collection.")
    parser.add_argument("--session", type=str,            help="Wipe a specific session by ID.")
    args = parser.parse_args()

    if args.wipe:
        show_stats()
        wipe_all()
    elif args.session:
        show_stats()
        wipe_session(args.session)
    else:
        show_stats()
"""
Build the diet-guideline vector store from data/guidelines/ (idempotent).

    python scripts/build_rag.py            # add new/changed documents
    python scripts/build_rag.py --rebuild  # wipe data/chroma_db and re-index everything
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_stores import CHROMA_DIR, GUIDELINE_DIR, ingest_document, search  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    files = sorted(f for f in os.listdir(GUIDELINE_DIR) if f.endswith((".md", ".txt", ".pdf")))
    total = sum(ingest_document(os.path.join(GUIDELINE_DIR, f)) for f in files)
    print(f"{len(files)} documents, {total} new chunks -> {CHROMA_DIR}")
    for q in ("How much sodium per day for high blood pressure?", "Which foods are high in purines?"):
        hits = search(q, k=2)
        print(f"\nQ: {q}\n" + "\n".join(f"  [{h.metadata['source']}] {h.page_content[:110]!r}" for h in hits))


if __name__ == "__main__":
    main()

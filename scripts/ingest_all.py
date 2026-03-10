"""Master ingest pipeline: indexes both PDF documents and FRMR JSON into ChromaDB.

Run this to build a combined corpus from:
  - docs/*.pdf         (your FedRAMP PDF documents)
  - FRMR GitHub JSON   (FedRAMP machine-readable requirements & definitions)

Usage (from project root):
    python scripts/ingest_all.py

To ingest only one source, run the individual scripts:
    python scripts/ingest.py          # PDFs only
    python scripts/ingest_json.py     # FRMR JSON only
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Add scripts/ to path so sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from ingest import ingest_pdfs
from ingest_json import ingest_frmr


def main() -> None:
    print("=" * 60)
    print("  FedRAMP RAG — Full Ingest (PDFs + FRMR JSON)")
    print("=" * 60)

    # ── Phase 1: PDFs ─────────────────────────────────────────────────────────
    # drop_existing=True rebuilds the collection from scratch.
    print("\n── Phase 1: PDF documents ──────────────────────────────────")
    pdf_count = ingest_pdfs(drop_existing=True)

    if pdf_count == 0:
        print("[all] No PDFs were indexed. Continuing with FRMR JSON only.")

    # ── Phase 2: FRMR JSON ────────────────────────────────────────────────────
    # drop_existing=False appends to the collection Phase 1 created.
    print("\n── Phase 2: FRMR machine-readable requirements ─────────────")
    json_count = ingest_frmr(drop_existing=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Ingest complete.")
    print(f"  PDF pages indexed   : {pdf_count}")
    print(f"  FRMR entries indexed: {json_count}")
    print(f"  Total               : {pdf_count + json_count}")
    print("=" * 60)
    print("\nStart the app with:  streamlit run app/main.py")


if __name__ == "__main__":
    main()

"""
rag_preprocess.py — PDF to chunked text preprocessor for RAG

Extracts PDFs page-by-page and groups them into small, labeled text chunks
before uploading to Google File Search. Each chunk header embeds the document
name and page range, so the model can always cite accurately.

Usage:
    python rag_preprocess.py [pdf_dir] [output_dir]

    pdf_dir     — directory containing your PDFs (default: same as rag_setup.py)
    output_dir  — where to write chunked .txt files (default: ./rag_chunks)
"""

import os
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("PyMuPDF not installed. Run: pip install pymupdf")

# ---------------------------------------------------------------------------
# Configuration — keep in sync with rag_setup.py
# ---------------------------------------------------------------------------

PDF_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "raw_pdfs")
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "rag_chunks"

# Pages per chunk for large documents. Smaller = more precise retrieval,
# but more files to upload. 5 is recommended for dense reference tables
# (error guides). Narrative docs use a larger size automatically.
CHUNK_SIZE_DENSE = 5   # for docs > 200 pages (error guides, large references)
CHUNK_SIZE_MEDIUM = 10  # for docs 51–200 pages
CHUNK_SIZE_SMALL = 0    # 0 = upload as single file (docs <= 50 pages)

# Overlap pages repeated at each chunk boundary so entries that straddle a
# seam appear fully in at least one chunk.  Only applied to chunked docs.
CHUNK_OVERLAP = 1

FILES_CONFIG = {
    "Aruba_8325_IGSG_en_us.pdf":                                          "Aruba_8325_Installation_and_Getting_Started_Guide",
    "Aruba_8325H_IGSG_en_us.pdf":                                         "Aruba_8325H_Installation_and_Getting_Started_Guide",
    "EC-10108_Install_Guide.pdf":                                         "EdgeConnect_EC-10108_Install_Guide",
    "EC-10108_StartUp_Guide.pdf":                                         "EdgeConnect_EC-10108_StartUp_Guide",
    "error-event-message-guide.pdf":                                      "Error_and_Event_Message_Reference_Guide",
    "HPE Aruba Networking CX 8325 Switch Series-a00059009enw.pdf":        "HPE_Aruba_CX_8325_Switch_Series_Specs",
    "HPE Aruba Networking EdgeConnect SD-WAN QuickSpecs-a50004289enw.pdf":"HPE_Aruba_EdgeConnect_SD-WAN_QuickSpecs",
    "ngfw-administration.pdf":                                            "NGFW_Administration_Guide",
    "Orch_UserGuide_R960.pdf":                                            "Orchestrator_User_Guide_R960",
    "pa-1400-hardware-reference.pdf":                                     "PA-1400_Hardware_Reference_Guide",
    "pa-1400-series.pdf":                                                 "PA-1400_Series_Guide",
    "pan-os-admin.pdf":                                                   "PAN-OS_Administration_Guide",
    "xr5610-om-en-us.pdf":                                                "XR5610_Operations_Manual",
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def chunk_size_for(total_pages: int) -> int:
    """Return the chunk size appropriate for a document of this length."""
    if total_pages <= 50:
        return 0  # single file
    elif total_pages <= 200:
        return CHUNK_SIZE_MEDIUM
    else:
        return CHUNK_SIZE_DENSE


def extract_page_text(page) -> str:
    """Extract text from a PDF page, falling back to raw dict blocks."""
    text = page.get_text("text")
    if text.strip():
        return text
    # Fallback: try extracting blocks individually (handles some scanned PDFs)
    blocks = page.get_text("blocks")
    return "\n".join(b[4] for b in blocks if b[4].strip())


def process_pdf(pdf_path: str, display_name: str, output_dir: str):
    """Convert one PDF into labeled text chunk files."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ❌ Could not open {os.path.basename(pdf_path)}: {e}")
        return 0

    total_pages = len(doc)
    size = chunk_size_for(total_pages)
    files_written = 0

    if size == 0:
        # Small document — single output file
        lines = [f"Document: {display_name}",
                 f"Total pages: {total_pages}",
                 "=" * 60, ""]
        for i in range(total_pages):
            page_text = extract_page_text(doc[i]).strip()
            if page_text:
                lines.append(f"[Page {i + 1}]")
                lines.append(page_text)
                lines.append("")
        out_path = os.path.join(output_dir, f"{display_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        files_written = 1
    else:
        # Large document — chunked files with overlap so entries near
        # chunk boundaries appear fully in at least one chunk.
        for start in range(0, total_pages, size):
            end = min(start + size - 1, total_pages - 1)
            start_page = start + 1   # 1-indexed for humans (chunk label)
            end_page = end + 1

            # Extend window backwards by CHUNK_OVERLAP pages for all chunks
            # after the first; the label still reflects the primary range.
            read_start = max(0, start - CHUNK_OVERLAP)

            lines = [f"Document: {display_name}",
                     f"Pages: {start_page}–{end_page} of {total_pages}",
                     "=" * 60, ""]
            has_content = False
            for i in range(read_start, end + 1):
                page_text = extract_page_text(doc[i]).strip()
                if page_text:
                    lines.append(f"[Page {i + 1}]")
                    lines.append(page_text)
                    lines.append("")
                    has_content = True

            if has_content:
                chunk_name = f"{display_name}_p{start_page:04d}-{end_page:04d}.txt"
                out_path = os.path.join(output_dir, chunk_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                files_written += 1

    doc.close()
    return files_written


def preprocess_all(pdf_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"PDF source:       {pdf_dir}\n")

    total_files = 0
    for filename, display_name in FILES_CONFIG.items():
        pdf_path = os.path.join(pdf_dir, filename)
        if not os.path.exists(pdf_path):
            print(f"  ⚠️  Missing: {filename}")
            continue

        doc = fitz.open(pdf_path)
        pages = len(doc)
        doc.close()
        size = chunk_size_for(pages)
        chunk_label = f"{pages // size} chunks of {size} pages" if size else f"1 file ({pages} pages)"
        print(f"  Processing: {filename}  ({pages} pages → {chunk_label})")

        n = process_pdf(pdf_path, display_name, output_dir)
        print(f"    → {n} file(s) written")
        total_files += n

    print(f"\n{'=' * 60}")
    print(f"✅ Done. {total_files} total chunk files written to '{output_dir}/'")
    print(f"\nNext step: run rag_setup.py pointing it at '{output_dir}/' instead of the PDF directory.")
    print(f"  python rag_setup.py {output_dir}")


if __name__ == "__main__":
    preprocess_all(PDF_DIR, OUTPUT_DIR)

"""
loader.py — extracts raw text from PDFs, page by page.

Tries the fast path first: pypdf's text-layer extraction (works for
"born-digital" PDFs with a real text layer). If a page comes back empty,
falls back to OCR (pytesseract + pdf2image) — this handles scanned/
photographed PDFs, which is common for older Islamic texts distributed as
page-image scans rather than digitally typeset books.

OCR is much slower than text extraction (seconds per page, not
milliseconds), so it only runs on pages that actually need it.

REQUIRED for OCR to work (free, one-time setup):
  1. Install Tesseract OCR engine (the actual OCR software):
       Windows: https://github.com/UB-Mannheim/tesseract/wiki  (installer)
       Mac:     brew install tesseract
       Linux:   sudo apt install tesseract-ocr
  2. Install Poppler (needed by pdf2image to rasterize PDF pages):
       Windows: https://github.com/oschwartz10612/poppler-windows/releases
                (unzip, then add its "Library/bin" folder to your PATH)
       Mac:     brew install poppler
       Linux:   sudo apt install poppler-utils
  3. pip install pytesseract pdf2image pillow  (already in requirements.txt)

If Tesseract/Poppler aren't installed, OCR calls will raise a clear error
telling you what's missing — the pypdf fast path still works fine for any
non-scanned PDFs in the meantime.
"""

from pathlib import Path
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class PageText:
    filename: str
    page_number: int  # 1-indexed
    text: str


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """OCR a single page. Imports are local so the whole project doesn't
    hard-require OCR libraries/binaries if you never hit this path."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(
        str(pdf_path), first_page=page_number, last_page=page_number, dpi=300
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0]).strip()


def load_pdf_pages(pdf_path: Path, use_ocr_fallback: bool = True) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages = []
    empty_pages = 0
    ocr_used = 0
    total_pages = len(reader.pages)

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text and use_ocr_fallback:
            try:
                text = _ocr_page(pdf_path, i)
                if text:
                    ocr_used += 1
            except Exception as e:
                # Missing Tesseract/Poppler, or a corrupt page — don't crash
                # the whole ingestion run over one page, just warn once per file.
                if empty_pages == 0:
                    print(
                        f"[OCR ERROR] {pdf_path.name} page {i}: {e} "
                        f"— check Tesseract/Poppler are installed (see loader.py docstring)."
                    )

        if not text:
            empty_pages += 1

        pages.append(PageText(filename=pdf_path.name, page_number=i, text=text))

        if i % 50 == 0:
            print(f"  ...{pdf_path.name}: processed {i}/{total_pages} pages ({ocr_used} via OCR so far)")

    if ocr_used:
        print(f"[OCR] {pdf_path.name}: {ocr_used}/{total_pages} pages required OCR.")

    if empty_pages > total_pages * 0.5:
        print(
            f"[WARNING] {pdf_path.name}: {empty_pages}/{total_pages} pages still "
            f"empty after OCR fallback. Check Tesseract/Poppler are installed, "
            f"or the scan quality may be too poor to OCR."
        )

    return pages


def load_all_pdfs(raw_pdf_dir: Path) -> dict[str, list[PageText]]:
    """Loads every PDF in the directory. Returns {filename: [PageText, ...]}."""
    result = {}
    pdf_files = sorted(raw_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDFs found in {raw_pdf_dir}. Add your files there.")
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        result[pdf_path.name] = load_pdf_pages(pdf_path)
    return result

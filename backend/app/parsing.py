"""Document parsing — extract UTF-8 text from uploaded bytes, per file type.

This is the CONTROL-PLANE (torch-free) text-extraction stage between raw upload and
onboarding. The onboard/train path (routers/jobs.py) must hand the ML plane extracted
TEXT, never raw PDF/DOCX bytes — so every uploaded document runs through extract_text()
and the result is what gets persisted + read back for retrieval and onboarding.

Deliberately pure-python (pypdf / python-docx / beautifulsoup4) so it stays importable
in the torch-free control plane: no OCR, no torch, no native ML deps. A scanned/image-only
PDF yields no text (returns ok=False), which is the honest signal — the fix is OCR upstream,
not here.

Contract: extract_text(filename, data) -> (text, ok, error).
  ok=True  -> `text` is the extracted UTF-8 body ("" is allowed only for genuinely empty
              text files; an image-only PDF that yields nothing is ok=False).
  ok=False -> `error` is a SHORT human-readable reason (unsupported type, encrypted,
              corrupt, decode failure). `text` is "" and the caller marks the doc failed.
"""
from __future__ import annotations

import io
import logging
import zipfile

logger = logging.getLogger(__name__)

# Decompression-bomb guards (anti-DoS): a small DOCX (a zip) can inflate to gigabytes and a PDF can
# declare an enormous page count, either of which would exhaust memory/CPU during extraction. Refuse
# up front as a clean parse FAILURE (never an exception the caller has to catch). Generous ceilings —
# a legitimate document is orders of magnitude under these.
_MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # 200 MB of inflated zip members
_MAX_DOCX_MEMBERS = 2_000                          # a real .docx has tens of parts, not thousands
_MAX_PDF_PAGES = 2_000

# Extensions we can extract text from. The upload validation (routers/corpora.py) and this
# module share this set as the single source of truth for "supported document type".
SUPPORTED_EXTS: frozenset[str] = frozenset(
    {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
)


class _ParseRejected(Exception):
    """A document we deliberately refuse to parse (e.g. a decompression bomb). The message is a short,
    safe reason; extract_text's normalizer turns it into (ok=False, error) like any other failure."""


def _ext(filename: str) -> str:
    """Lowercased extension including the dot ('' when the name has none)."""
    name = filename.rsplit("/", 1)[-1]
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""


def _extract_txt(data: bytes) -> str:
    """Plain text / markdown: decode as UTF-8. Fall back through common encodings so a
    Windows-authored (cp1252) or Latin-1 file doesn't fail the whole upload; last resort
    is a lossy UTF-8 decode (never raises) so we always return SOMETHING for a text file."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    """PDF via pypdf. Raises on an encrypted PDF we can't open with the empty password, or
    on a corrupt file — the caller turns that into ok=False + a short error. An image-only
    (scanned) PDF extracts no text; the caller treats empty output as a failure."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Try the empty password (many "encrypted" PDFs are just permission-flagged);
            # a real password-protected file raises and becomes a clean parse failure.
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001 — normalize to our error contract
                raise PdfReadError("encrypted PDF (password required)") from exc
        # Refuse an absurd page count BEFORE iterating pages (each extract_text() is real work).
        n_pages = len(reader.pages)
        if n_pages > _MAX_PDF_PAGES:
            raise _ParseRejected(f"PDF has {n_pages} pages (limit {_MAX_PDF_PAGES})")
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except _ParseRejected:
        raise
    except PdfReadError:
        raise
    except Exception as exc:  # noqa: BLE001 — pypdf raises assorted low-level errors on corrupt input
        raise PdfReadError(f"unreadable PDF: {type(exc).__name__}") from exc


def _extract_docx(data: bytes) -> str:
    """DOCX via python-docx. Pulls paragraph text plus table cell text (so a doc that puts
    its content in tables isn't onboarded empty). A non-DOCX (e.g. legacy .doc) or corrupt
    file raises PackageNotFoundError, which the caller turns into a clean parse failure."""
    from docx import Document as DocxDocument

    # A DOCX is a zip. Inspect the central directory FIRST (no inflation happens here) and refuse a
    # decompression bomb — too many members or a total uncompressed size that would blow up memory —
    # before python-docx unzips anything.
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > _MAX_DOCX_MEMBERS:
                raise _ParseRejected(f"DOCX has {len(infos)} members (limit {_MAX_DOCX_MEMBERS})")
            total = sum(i.file_size for i in infos)  # declared uncompressed sizes
            if total > _MAX_DOCX_UNCOMPRESSED_BYTES:
                raise _ParseRejected(
                    f"DOCX inflates to {total} bytes (limit {_MAX_DOCX_UNCOMPRESSED_BYTES})")
    except zipfile.BadZipFile as exc:
        # Not a valid zip (e.g. a legacy .doc renamed) — a clean parse failure, not a crash.
        raise _ParseRejected("not a valid DOCX (bad zip)") from exc

    doc = DocxDocument(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text:
                    parts.append(cell.text)
    return "\n".join(parts)


def _extract_html(data: bytes) -> str:
    """HTML via BeautifulSoup: strip tags, drop script/style, keep visible text. Uses the
    stdlib html.parser (no lxml dependency) so it stays pure-python."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_extract_txt(data), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def extract_text(filename: str, data: bytes) -> tuple[str, bool, str]:
    """Extract UTF-8 text from `data` based on `filename`'s extension.

    Returns (text, ok, error). Unsupported extension, an encrypted/corrupt file, or an
    extraction that yields no text (e.g. an image-only PDF) -> ok=False with a short error.
    Never raises: any low-level extractor error is caught and normalized to the error string
    so one bad file can't crash an upload of many.
    """
    ext = _ext(filename)
    if ext not in SUPPORTED_EXTS:
        return "", False, f"unsupported file type '{ext or filename}'"

    try:
        if ext in (".txt", ".md"):
            text = _extract_txt(data)
        elif ext == ".pdf":
            text = _extract_pdf(data)
        elif ext == ".docx":
            text = _extract_docx(data)
        else:  # .html / .htm
            text = _extract_html(data)
    except Exception as exc:  # noqa: BLE001 — normalize every extractor failure to the contract
        logger.info("parse failed for %s: %s", filename, exc)
        return "", False, f"could not extract text ({type(exc).__name__}): {str(exc)[:120]}"

    # Binary formats (PDF/DOCX/HTML) that extract to nothing are a FAILURE — an image-only
    # scan or a broken file the extractor opened but found no text in. Plain .txt/.md are
    # allowed to be legitimately empty (an empty note is still a valid, if useless, document).
    if not text.strip() and ext not in (".txt", ".md"):
        return "", False, "no extractable text (document may be image-only or empty)"
    return text, True, ""

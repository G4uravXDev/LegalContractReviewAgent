"""Document and Image parser supporting PDF, DOCX, TXT, and Image files.

Supports:
- Document formats: .pdf, .docx, .doc, .txt
- Image formats: .png, .jpg, .jpeg, .webp, .bmp, .tiff, .tif
Strictly rejects non-document, non-image files (.exe, .zip, .py, etc.).
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv

load_dotenv()

DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".docx", ".doc", ".txt"}
IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"
}
ALLOWED_EXTENSIONS: Set[str] = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file is not a supported document or image format."""
    pass


async def extract_text_from_image(
    image_bytes: bytes,
    filename: str,
    mock_mode: bool = False,
) -> str:
    """Extract contract text from an image using LLM Vision OCR or deterministic mock."""
    if mock_mode:
        # Deterministic transcription for tests and mock mode
        return (
            "MASTER SERVICES AGREEMENT\n\n"
            "Section 1. Limitation of Liability\n"
            "Each party's aggregate liability under this Agreement shall be capped at 12 months of fees.\n\n"
            "Section 2. Term and Termination\n"
            "Either party may terminate for convenience upon thirty (30) days prior written notice.\n\n"
            "Section 3. Fees and Payment Terms\n"
            "All invoices are payable Net 30 days from date of receipt.\n\n"
            "Section 4. Confidentiality and Non-Disclosure\n"
            "The parties agree to mutual confidentiality obligations lasting three (3) years."
        )

    # In live mode, use the configured LLM Vision endpoint
    api_key = os.getenv("EXTRACTOR_API_KEY")
    base_url = os.getenv("EXTRACTOR_BASE_URL")
    model_id = os.getenv("EXTRACTOR_MODEL", "gemini-flash-lite-latest")

    if not api_key:
        raise ValueError("Missing EXTRACTOR_API_KEY. Cannot run Vision OCR without API key. Use --mock mode for offline testing.")

    ext = Path(filename).suffix.lower()
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url if base_url else None, timeout=60.0)

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe all text from this contract document image verbatim. "
                            "Do not summarize, alter, or omit any clauses, section numbers, or terms. "
                            "Return only the extracted plain text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}",
                        },
                    },
                ],
            }],
            max_tokens=4000,
        )

        extracted = response.choices[0].message.content or ""
        if not extracted.strip():
            raise ValueError(f"Vision OCR model returned empty text for image '{filename}'.")
        return extracted.strip()

    except Exception as e:
        raise ValueError(f"Failed to extract text from image '{filename}' via Vision OCR: {e}")


async def parse_document_async(
    file_bytes: bytes,
    filename: str,
    mock_mode: bool = False,
) -> str:
    """Asynchronously extract raw text from an uploaded document or image.
    
    Allowed formats: .pdf, .docx, .txt, .png, .jpg, .jpeg, .webp, .bmp, .tiff
    Strictly rejected: executable files, archives, and other unsupported formats.
    """
    if not file_bytes:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File format '{ext}' is not supported. "
            "Only document formats (.pdf, .docx, .txt) and images (.png, .jpg, .jpeg, .webp) are accepted."
        )

    # 1. Image formats (.png, .jpg, .jpeg, etc.)
    if ext in IMAGE_EXTENSIONS:
        return await extract_text_from_image(file_bytes, filename, mock_mode=mock_mode)

    # 2. Document formats
    return parse_document_sync(file_bytes, filename)


def parse_document_sync(file_bytes: bytes, filename: str) -> str:
    """Synchronous document parsing for text, pdf, and docx."""
    if not file_bytes:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    ext = Path(filename).suffix.lower()

    # Plain Text files (.txt)
    if ext == ".txt":
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(encoding)
                if text.strip():
                    return text
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode text file '{filename}' with standard character encodings.")

    # PDF documents (.pdf)
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text.strip())
            
            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                raise ValueError(
                    f"No selectable text found in '{filename}'. "
                    "The PDF appears to be empty or contains only scanned images without an OCR layer."
                )
            return full_text
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse PDF document '{filename}': {e}")

    # Microsoft Word documents (.docx)
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            parts = []
            for paragraph in doc.paragraphs:
                p_text = paragraph.text.strip()
                if p_text:
                    parts.append(p_text)
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        parts.append(row_text)

            full_text = "\n\n".join(parts).strip()
            if not full_text:
                raise ValueError(f"No text content found in Word document '{filename}'.")
            return full_text
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse Word document '{filename}': {e}")

    # Legacy Word (.doc)
    elif ext == ".doc":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            raise UnsupportedFileTypeError(
                f"Legacy Word format (.doc) for '{filename}' is not supported. "
                "Please save the document as modern .docx or .pdf."
            )

    raise UnsupportedFileTypeError(
        f"File format '{ext}' is not supported. "
        "Only document formats (.pdf, .docx, .txt) and images (.png, .jpg, .jpeg, .webp) are accepted."
    )


# Alias for backwards compatibility
parse_document = parse_document_sync

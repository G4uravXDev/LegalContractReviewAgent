"""Utilities package."""

from utils.document_parser import (
    parse_document,
    parse_document_sync,
    parse_document_async,
    extract_text_from_image,
    UnsupportedFileTypeError,
    ALLOWED_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
)

__all__ = [
    "parse_document",
    "parse_document_sync",
    "parse_document_async",
    "extract_text_from_image",
    "UnsupportedFileTypeError",
    "ALLOWED_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "IMAGE_EXTENSIONS",
]

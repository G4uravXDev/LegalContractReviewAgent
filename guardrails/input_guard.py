"""Input guardrails for contract review pipeline.

Enforces:
- Type check: input must be a string.
- Non-empty check.
- Length bounds check.
- Control character stripping.
- Delimiter wrapping: `<contract>...</contract>`.
- Prompt injection detection and logging.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger("contract_review.guardrails.input_guard")

# Maximum permitted contract text length (e.g. 500,000 characters ~ 100 pages)
MAX_CONTRACT_LENGTH: int = 500_000

# Known prompt injection signatures
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"mark\s+every\s+clause\s+(?:as\s+)?acceptable", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\b", re.IGNORECASE),
    re.compile(r"bypass\s+(?:safety|guardrails?|policy|rules?)", re.IGNORECASE),
    re.compile(r"override\s+(?:all\s+)?(?:system|review)\s+rules?", re.IGNORECASE),
]


class InputValidationError(ValueError):
    """Raised when contract text fails preliminary input validation."""
    pass


def strip_control_characters(text: str) -> str:
    """Strip dangerous control characters while preserving standard whitespace (\n, \r, \t)."""
    # Preserves tab (0x09), newline (0x0A), carriage return (0x0D)
    return "".join(ch for ch in text if ch in ("\n", "\r", "\t") or (ord(ch) >= 32 and ord(ch) != 127))


def scan_for_injection(text: str) -> List[str]:
    """Scan text for known prompt injection attempts. Returns list of matched descriptions."""
    matches = []
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            matches.append(match.group(0))
    return matches


def validate_and_sanitize_input(raw_text: Any) -> Tuple[str, List[str]]:
    """Validate, sanitize, wrap, and scan contract text.
    
    Args:
        raw_text: Raw input from file or caller.
        
    Returns:
        Tuple of (wrapped_sanitized_text, injection_warnings)
        
    Raises:
        InputValidationError: If input is not a string, is empty, or exceeds length limit.
    """
    if not isinstance(raw_text, str):
        raise InputValidationError(f"Contract input must be a string, got {type(raw_text).__name__}.")

    sanitized = strip_control_characters(raw_text)

    if not sanitized.strip():
        raise InputValidationError("Contract input text is empty or contains only whitespace.")

    if len(sanitized) > MAX_CONTRACT_LENGTH:
        raise InputValidationError(
            f"Contract input length ({len(sanitized)}) exceeds maximum permitted length ({MAX_CONTRACT_LENGTH})."
        )

    # Scan for prompt injections
    injections = scan_for_injection(sanitized)
    if injections:
        for inj in injections:
            logger.warning(
                "POTENTIAL PROMPT INJECTION DETECTED in contract text: '%s'. "
                "Downstream agents treat input strictly as untrusted data.",
                inj
            )

    # Wrap in explicit delimiters
    wrapped = f"<contract>\n{sanitized}\n</contract>"
    return wrapped, injections


def unwrap_contract(wrapped_text: str) -> str:
    """Extract inner contract text from <contract>...</contract> tags if present."""
    match = re.search(r"<contract>\s*([\s\S]*?)\s*</contract>", wrapped_text)
    if match:
        return match.group(1)
    return wrapped_text

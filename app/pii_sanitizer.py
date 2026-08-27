"""Aggressive regex PII redaction applied before any LLM transmission."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)

IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
)

IPV6_RE = re.compile(
    r"(?<![A-Za-z0-9:])(?:"
    r"(?:[A-Fa-f0-9]{1,4}:){1,7}[A-Fa-f0-9]{1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,7}:"
    r"|::(?:[A-Fa-f0-9]{1,4}:){0,7}[A-Fa-f0-9]{1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}(?::[A-Fa-f0-9]{1,4}){0,5}"
    r"|::"
    r")(?![A-Za-z0-9:])"
)

# Cover NANP, spaced international, and common dotted/dashed local formats.
PHONE_RE = re.compile(
    r"""
    (?<!\w)
    (?:
        \+\d{1,3}[\s.-]*
        (?:\(?\d{1,4}\)?[\s.-]*){1,3}
        \d{2,4}
      |
        \(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}
      |
        \d{3}[\s.-]\d{3}[\s.-]\d{4}
      |
        \d{3}[\s.-]\d{4}
    )
    (?!\w)
    """,
    re.VERBOSE,
)

_PIPELINE: tuple[tuple[re.Pattern[str], str], ...] = (
    (EMAIL_RE, "[EMAIL]"),
    (IPV6_RE, "[IP]"),
    (IPV4_RE, "[IP]"),
    (PHONE_RE, "[PHONE]"),
)


@dataclass(frozen=True)
class SanitizationResult:
    text: str
    replacements: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return any(count > 0 for count in self.replacements.values())


def sanitize_text(text: str | None) -> str:
    """Replace emails, phone numbers, and IP addresses with typed tokens."""
    return sanitize_with_stats(text).text


def sanitize_with_stats(text: str | None) -> SanitizationResult:
    if not text:
        return SanitizationResult(text="" if text is None else text)

    counts: dict[str, int] = {"[EMAIL]": 0, "[PHONE]": 0, "[IP]": 0}
    sanitized = text
    for pattern, token in _PIPELINE:
        sanitized, n = pattern.subn(token, sanitized)
        counts[token] += n
    return SanitizationResult(text=sanitized, replacements=counts)


def sanitize_ticket_fields(subject: str, body: str) -> tuple[str, str]:
    """Redact the ticket fields that are forwarded to the LLM."""
    return sanitize_text(subject), sanitize_text(body)


if __name__ == "__main__":
    sample = (
        "Contact jane.doe@initech.com or +1 (415) 555-0199. "
        "Source IP 10.12.8.44 / 2001:db8:85a3::8a2e:370:7334."
    )
    result = sanitize_with_stats(sample)
    assert "[EMAIL]" in result.text
    assert "[PHONE]" in result.text
    assert result.text.count("[IP]") == 2
    assert "jane.doe" not in result.text
    print("pii_sanitizer ok:", result.text)

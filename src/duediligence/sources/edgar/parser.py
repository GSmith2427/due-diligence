"""10-K HTML → clean text.

SEC filings are HTML, often with inline XBRL tags, navigation chrome,
embedded tables, and other rendering artifacts. This module produces a
readable plain-text rendering suitable for chunking and embedding.

Goals:
    * Preserve textual content and natural paragraph structure.
    * Remove script/style/nav elements and rendering chrome.
    * Render tables as tab-separated rows (compact, retains structure).
    * Collapse whitespace runs without destroying paragraph boundaries.

Non-goals (deliberate):
    * Recovering inline XBRL facts. That belongs in a separate XBRL parser
      we'll add when we need structured financials.
    * Perfect typographic fidelity. We're producing input for an embedder,
      not for human reading.
"""

from __future__ import annotations

import re
from typing import Final

from bs4 import BeautifulSoup, NavigableString, Tag

# Tags that contribute no semantic content to a filing's text.
_DROP_TAGS: Final[frozenset[str]] = frozenset(
    {"script", "style", "head", "meta", "link", "title", "noscript"}
)

# Tags that should be treated as block-level for paragraph separation.
_BLOCK_TAGS: Final[frozenset[str]] = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "tr",
    }
)


def html_to_text(html: bytes | str) -> str:
    """Convert filing HTML to clean plain text.

    The implementation is intentionally simple: load the document, drop
    non-content tags, walk the tree turning blocks into newline-separated
    text and tables into TSV-like rows, then collapse runs of whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in _DROP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Replace table elements with a TSV representation before generic text
    # extraction, so cell boundaries are preserved.
    for table in soup.find_all("table"):
        table.replace_with(NavigableString(_render_table(table)))

    parts: list[str] = []
    _collect_text(soup, parts)
    text = "".join(parts)

    # Collapse runs of whitespace within lines, but preserve newlines.
    text = re.sub(r"[ \f\v]+", " ", text)
    # Cap blank-line runs at one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _collect_text(node: Tag | NavigableString, out: list[str]) -> None:
    if isinstance(node, NavigableString):
        out.append(str(node))
        return
    for child in node.children:
        if isinstance(child, NavigableString):
            out.append(str(child))
        elif isinstance(child, Tag):
            is_block = child.name in _BLOCK_TAGS
            if is_block:
                out.append("\n")
            _collect_text(child, out)
            if is_block:
                out.append("\n")


def _render_table(table: Tag) -> str:
    """Render an HTML table as TSV-ish rows.

    Cells are joined by tabs, rows by newlines. Whitespace inside cells is
    collapsed to a single space so each row stays on one line.
    """
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
            for cell in row.find_all(["td", "th"])
        ]
        if any(cells):
            lines.append("\t".join(cells))
    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"

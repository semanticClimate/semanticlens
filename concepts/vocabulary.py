"""
Builds a vocabulary from the IPCC AR6 glossary

Extracts terms, definitions, aliases, and working-group details
from glossary HTML and saves them with source information
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ipcc_ar6_vocabulary.json"

DEFAULT_SOURCE = (
    "https://raw.githubusercontent.com/TIBHannover/ClimateKG-Data-Bench/"
    "main/data/glossary-ar6.html"
)
OFFICIAL_GLOSSARY_URL = "https://apps.ipcc.ch/glossary/searchlatest.php"
WORKING_GROUP_CODES = {"WGI": "WG1", "WGII": "WG2", "WGIII": "WG3"}
SCHEMA_VERSION = "1.0"
MAX_HTML_BYTES = 5_000_000

_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})
_ACRONYM = re.compile(r"^(?:[A-Z][A-Z0-9+./-]{1,14}|[A-Z]{2,14}s)$")
_END_PAREN = re.compile(r"\(([^()]*)\)\s*$")


def normalize_term(text: str) -> str:
    """
    Normalize a term by standardizing Unicode characters, dashes, and spaces
    """
    if not isinstance(text, str):
        raise TypeError("Term must be a string")
    value = unicodedata.normalize("NFKC", text).translate(_DASHES)
    return " ".join(value.casefold().split())


def _visible_text(element: Any) -> str:
    """
    Extract clean, readable text from an HTML element
    """
    return " ".join(element.get_text(" ", strip=True).split()) if element else ""


def _aliases(label: str, alternate_heading: str) -> tuple[str, ...]:
    """
    Find alternate names and acronyms for a glossary term without duplicates
    """
    options: list[str] = []
    if alternate_heading and normalize_term(alternate_heading) != normalize_term(label):
        options.append(alternate_heading)
    match = _END_PAREN.search(alternate_heading)
    if match:
        candidate = match.group(1).strip()
        if _ACRONYM.fullmatch(candidate):
            options.append(candidate)

    unique: dict[str, str] = {}
    for alias in options:
        key = normalize_term(alias)
        if key and key != normalize_term(label):
            unique.setdefault(key, alias)
    return tuple(unique.values())


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    """
    Store a glossary term with its definition, aliases, and source details
    """

    term_id: str
    label: str
    definition: str | None
    aliases: tuple[str, ...]
    working_groups: tuple[str, ...]
    source_series: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the glossary term into a dictionary
        """
        return asdict(self)


class Vocabulary:
    """
    Store glossary terms and provide lookups by term ID or spelling
    """

    def __init__(self, terms: list[GlossaryTerm]) -> None:
        self.terms = tuple(
            sorted(terms, key=lambda t: (normalize_term(t.label), t.term_id))
        )
        by_id: dict[str, GlossaryTerm] = {}
        index: dict[str, set[str]] = {}
        for term in self.terms:
            if term.term_id in by_id:
                raise ValueError(f"Duplicate vocabulary term ID: {term.term_id}")
            by_id[term.term_id] = term
            for spelling in (term.label, *term.aliases):
                key = normalize_term(spelling)
                if key:
                    index.setdefault(key, set()).add(term.term_id)
        self._by_id = by_id
        self._index = {key: tuple(sorted(ids)) for key, ids in index.items()}

    def get(self, term_id: str) -> GlossaryTerm | None:
        """
        Find a glossary term by its ID
        """
        return self._by_id.get(term_id)

    def lookup(self, spelling: str) -> tuple[GlossaryTerm, ...]:
        """
        Find all glossary terms that match a given spelling
        """
        return tuple(
            self._by_id[i] for i in self._index.get(normalize_term(spelling), ())
        )

    @property
    def ambiguous_labels(self) -> dict[str, list[str]]:
        """
        Return spellings that match more than one glossary term
        """
        return {
            key: list(ids) for key, ids in sorted(self._index.items()) if len(ids) > 1
        }


def parse_glossary_html(html: str) -> Vocabulary:
    """
    Extract glossary terms and their details from structured HTML

    Check for missing entries and repeated labels before building the vocabulary
    """
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.select(".term-list .term")
    if not elements:
        raise ValueError(
            "No glossary terms found; expected .term-list .term HTML elements"
        )

    terms: list[GlossaryTerm] = []
    seen_labels: set[str] = set()
    for index, node in enumerate(elements, start=1):
        label_node = node.select_one(".term-name")
        definition_node = node.select_one(".term-def")
        if label_node is None or definition_node is None:
            raise ValueError(
                f"Glossary entry {index} lacks a label or definition element"
            )
        label = _visible_text(label_node)
        if not label:
            raise ValueError(f"Glossary entry {index} has an empty label")

        definition = _visible_text(definition_node) or None
        key = normalize_term(label)
        if key in seen_labels:
            raise ValueError(f"Repeated normalized headword: {label!r}")
        seen_labels.add(key)

        alternate = _visible_text(node.select_one(".term-aka"))
        alternate = re.sub(r"^Also known as:\s*", "", alternate, flags=re.IGNORECASE)
        series = []
        for badge in node.select(".series-badge"):
            code = _visible_text(badge).split(" ", 1)[0].upper()
            if code and code not in series:
                series.append(code)
        groups = tuple(wg for code, wg in WORKING_GROUP_CODES.items() if code in series)
        term_id = "ipcc-ar6-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        terms.append(
            GlossaryTerm(
                term_id=term_id,
                label=label,
                definition=definition,
                aliases=_aliases(label, alternate),
                working_groups=groups,
                source_series=tuple(series),
            )
        )
    return Vocabulary(terms)


def _read_html(source: str | Path) -> tuple[str, bytes]:
    """
    Read glossary HTML from an HTTPS URL or local file

    Check the content size and return the decoded text and original bytes
    """
    source = str(source)
    if source.startswith("https://"):
        with requests.get(
            source,
            timeout=(10, 45),
            stream=True,
            headers={"User-Agent": "IPCC-Visual-Knowledge-Graph/0.1"},
        ) as response:
            response.raise_for_status()
            if urlsplit(response.url).scheme != "https":
                raise ValueError("Refusing a redirect to non-HTTPS content")
            chunks = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                chunks.extend(chunk)
                if len(chunks) > MAX_HTML_BYTES:
                    raise ValueError("Glossary HTML exceeds configured size limit")
            data = bytes(chunks)
    elif source.startswith(("http://", "ftp://")):
        raise ValueError("Remote source must use HTTPS")
    else:
        data = Path(source).read_bytes()
        if len(data) > MAX_HTML_BYTES:
            raise ValueError("Glossary HTML exceeds configured size limit")
    return data.decode("utf-8-sig"), data


def build_vocabulary(
    source: str | Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    *,
    min_terms: int = 500,
) -> dict[str, Any]:
    """
    Build and save a vocabulary from the glossary HTML

    Check the number of terms and include source details and summary counts
    """
    html, source_bytes = _read_html(source)
    vocabulary = parse_glossary_html(html)
    if len(vocabulary.terms) < min_terms:
        raise ValueError(
            f"Only {len(vocabulary.terms)} terms parsed (minimum {min_terms}); "
            "check source completeness before saving"
        )

    summary = {
        "total_terms": len(vocabulary.terms),
        "terms_by_working_group": {
            wg: sum(wg in term.working_groups for term in vocabulary.terms)
            for wg in ("WG1", "WG2", "WG3")
        },
        "unassigned_to_ar6_wg": sum(not t.working_groups for t in vocabulary.terms),
        "missing_definitions": sum(t.definition is None for t in vocabulary.terms),
        "ambiguous_spellings": len(vocabulary.ambiguous_labels),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "vocabulary_id": "ipcc-ar6-glossary",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "derived_export": str(source),
            "official_glossary": OFFICIAL_GLOSSARY_URL,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "note": "Third-party HTML export derived from IPCC glossary; review before publication.",
        },
        "summary": summary,
        "ambiguous_spellings": vocabulary.ambiguous_labels,
        "terms": [term.to_dict() for term in vocabulary.terms],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pending = output.with_name(output.name + ".tmp")
    try:
        with pending.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return summary


def load_vocabulary(path: Path = DEFAULT_OUTPUT) -> Vocabulary:
    """
    Load glossary terms from a saved vocabulary JSON file
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported vocabulary schema_version")
    rows = payload.get("terms")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Vocabulary is missing its terms")
    try:
        terms = [
            GlossaryTerm(
                term_id=item["term_id"],
                label=item["label"],
                definition=item["definition"],
                aliases=tuple(item["aliases"]),
                working_groups=tuple(item["working_groups"]),
                source_series=tuple(item["source_series"]),
            )
            for item in rows
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed vocabulary term record") from exc
    return Vocabulary(terms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a provenance-aware IPCC AR6 glossary vocabulary"
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Structured glossary HTML URL or local path",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-terms", type=int, default=500)
    args = parser.parse_args()
    if args.min_terms < 1:
        parser.error("--min-terms must be at least 1")
    summary = build_vocabulary(args.source, args.output, min_terms=args.min_terms)
    print(
        f"Vocabulary: {summary['total_terms']} terms, "
        f"{summary['missing_definitions']} without definitions, "
        f"{summary['unassigned_to_ar6_wg']} without WG1/2/3 attribution, "
        f"{summary['ambiguous_spellings']} ambiguous spellings"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

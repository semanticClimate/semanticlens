"""
Clean validated IPCC AR6 figure records without modifying raw data

- Validate records using the Phase 2 FigureRecord schema
- Remove known IPCC footer images from figure image candidates
- Remove duplicate image URLs while preserving their order
- Flag missing captions and images without adding replacements
- Detect duplicate figure records and report invalid input

Does not scrape, download images, modify scientific captions, read or write files or create the SQLite database
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import urlsplit

from pydantic import ValidationError

from corpus.schemas import FigureRecord


_DECORATIVE_FILENAMES = frozenset(
    {"footer-left.png", "footer-right.png", "footer-center.png", "footer-centre.png"}
)


class CleanFigure(FigureRecord):
    """
    A source backed figure with cleaned image references and explicit status
    """

    caption_status: Literal["available", "missing"]
    image_status: Literal["referenced", "missing"]


@dataclass(frozen=True, slots=True)
class CleaningIssue:
    """
    A rejected or repeated raw record, identified without copying captions
    """

    index: int
    code: Literal["invalid_record", "duplicate_record"]
    figure_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """
    Cleaned figures and information needed by the later quality report
    """

    figures: tuple[CleanFigure, ...]
    issues: tuple[CleaningIssue, ...]
    input_count: int
    removed_decorative_urls: int
    removed_duplicate_image_urls: int

    @property
    def summary(self) -> dict[str, int]:
        """
        Small JSON ready summary; no persistence or report formatting here
        """
        return {
            "input_records": self.input_count,
            "cleaned_figures": len(self.figures),
            "invalid_records": sum(i.code == "invalid_record" for i in self.issues),
            "duplicate_records": sum(i.code == "duplicate_record" for i in self.issues),
            "missing_captions": sum(f.caption_status == "missing" for f in self.figures),
            "missing_images": sum(f.image_status == "missing" for f in self.figures),
            "removed_decorative_urls": self.removed_decorative_urls,
            "removed_duplicate_image_urls": self.removed_duplicate_image_urls,
        }


def _is_decorative_image(url: str) -> bool:
    """
    Recognize known IPCC site furniture not general static hosted images
    """
    path = urlsplit(url).path.lower()
    return "/static/" in path and path.rsplit("/", 1)[-1] in _DECORATIVE_FILENAMES


def _clean_image_urls(urls: list[str]) -> tuple[list[str], int, int]:
    """
    Keep real candidates in original order & count each type of removal
    """
    kept: list[str] = []
    seen: set[str] = set()
    decorative_removed = 0
    duplicates_removed = 0

    for url in urls:
        if _is_decorative_image(url):
            decorative_removed += 1
        elif url in seen:
            duplicates_removed += 1
        else:
            kept.append(url)
            seen.add(url)

    return kept, decorative_removed, duplicates_removed


def _clean_validated(record: FigureRecord) -> tuple[CleanFigure, int, int]:
    urls, decorative_removed, duplicates_removed = _clean_image_urls(record.image_urls)

    primary_image = record.image_url if record.image_url in urls else (urls[0] if urls else None)

    cleaned = CleanFigure.model_validate(
        {
            **record.model_dump(),
            "image_url": primary_image,
            "image_urls": urls,
            "caption_status": "available" if record.has_caption else "missing",
            "image_status": "referenced" if urls else "missing",
        }
    )
    return cleaned, decorative_removed, duplicates_removed


def clean_figure(record: FigureRecord | Mapping[str, Any]) -> CleanFigure:
    """
    Clean one figure & raise ValidationError for malformed records
    """
    validated = FigureRecord.model_validate(record)
    return _clean_validated(validated)[0]


def clean_figures(
    records: Iterable[FigureRecord | Mapping[str, Any]],
) -> CleaningResult:
    """
    Clean a corpus and deduplicate by provenance derived figure ID

    Exact duplicates are kept once and reported. Conflicting duplicate IDs
    raise ValueError rather than silently choosing potentially wrong evidence

    Invalid inputs are excluded from processed output and reported; original
    raw JSON is never changed
    """
    figures: dict[str, CleanFigure] = {}
    issues: list[CleaningIssue] = []
    total = 0
    decorative_removed = 0
    duplicates_removed = 0

    for index, raw in enumerate(records):
        total += 1
        try:
            validated = FigureRecord.model_validate(raw)
        except ValidationError as exc:
            figure_id = raw.get("figure_id") if isinstance(raw, Mapping) else None
            first = exc.errors(include_input=False, include_url=False)[0]
            issues.append(
                CleaningIssue(
                    index=index,
                    code="invalid_record",
                    figure_id=figure_id if isinstance(figure_id, str) else None,
                    detail=f"{'.'.join(map(str, first['loc']))}: {first['msg']}",
                )
            )
            continue

        cleaned, removed, repeated = _clean_validated(validated)
        decorative_removed += removed
        duplicates_removed += repeated
        existing = figures.get(cleaned.figure_id)
        if existing is not None:
            
            without_timestamp = {"collected_at"}
            if existing.model_dump(exclude=without_timestamp) != cleaned.model_dump(
                exclude=without_timestamp
            ):
                raise ValueError(
                    f"Conflicting records for figure_id {cleaned.figure_id!r}; "
                    "resolve this in the raw corpus before building the database"
                )
            issues.append(
                CleaningIssue(
                    index=index,
                    code="duplicate_record",
                    figure_id=cleaned.figure_id,
                    detail="Repeated figure with identical data (apart from collection time)",
                )
            )
            continue
        figures[cleaned.figure_id] = cleaned

    return CleaningResult(
        figures=tuple(figures.values()),
        issues=tuple(issues),
        input_count=total,
        removed_decorative_urls=decorative_removed,
        removed_duplicate_image_urls=duplicates_removed,
    )

"""
Build the validated IPCC AR6 figure corpus from three raw WG JSON files

- Load and clean records using corpus.schemas and corpus.clean
- Store the cleaned data in SQLite
- Generate a data quality report
- Keep the original input JSON files unchanged
- Do not scrape IPCC, generate captions or check image URL availability

Usage:
    python -m corpus.build_corpus

Requires:
    corpus.schemas, corpus.clean and Pydantic 2.x
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from corpus.clean import CleanFigure, CleaningResult, clean_figures


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKING_GROUPS = ("WG1", "WG2", "WG3")
DATABASE_SCHEMA_VERSION = 1


_CREATE_TABLES = """
CREATE TABLE figures (
    figure_id            TEXT PRIMARY KEY,
    schema_version       TEXT NOT NULL,
    working_group        TEXT NOT NULL CHECK (working_group IN ('WG1', 'WG2', 'WG3')),
    figure_label         TEXT,
    title                TEXT,
    chapter              TEXT,
    section              TEXT NOT NULL,
    caption              TEXT,
    caption_html         TEXT,
    image_url            TEXT,
    source_url           TEXT NOT NULL UNIQUE,
    source_collection    TEXT NOT NULL,
    copyright_policy_url TEXT NOT NULL,
    collected_at         TEXT NOT NULL,
    caption_status       TEXT NOT NULL CHECK (caption_status IN ('available', 'missing')),
    image_status         TEXT NOT NULL CHECK (image_status IN ('referenced', 'missing'))
);

CREATE TABLE figure_images (
    figure_id    TEXT NOT NULL REFERENCES figures(figure_id) ON DELETE CASCADE,
    position     INTEGER NOT NULL CHECK (position >= 0),
    image_url    TEXT NOT NULL,
    PRIMARY KEY (figure_id, position)
);

CREATE INDEX idx_figures_wg_section ON figures(working_group, section);
CREATE INDEX idx_figures_caption_status ON figures(caption_status);
"""

_INSERT_FIGURE = """
INSERT INTO figures (
    figure_id, schema_version, working_group, figure_label, title,
    chapter, section, caption, caption_html, image_url, source_url,
    source_collection, copyright_policy_url, collected_at,
    caption_status, image_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _read_raw_figures(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Load *all* three WG exports & never silently omit a missing input file
    """
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for wg in WORKING_GROUPS:
        path = raw_dir / f"{wg.lower()}.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Required corpus input not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a non-empty JSON array in {path}")

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path} at index {index}")
            if row.get("working_group") != wg:
                raise ValueError(
                    f"Working Group mismatch in {path} at index {index}: "
                    f"expected {wg}, found {row.get('working_group')!r}"
                )

        records.extend(rows)
        counts[wg] = len(rows)

    return records, counts


def _figure_row(figure: CleanFigure) -> tuple[str | None, ...]:
    return (
        figure.figure_id,
        figure.schema_version,
        figure.working_group,
        figure.figure_label,
        figure.title,
        figure.chapter,
        figure.section,
        figure.caption,
        figure.caption_html,
        figure.image_url,
        figure.source_url,
        figure.source_collection,
        figure.copyright_policy_url,
        figure.collected_at,
        figure.caption_status,
        figure.image_status,
    )


def _write_database(path: Path, figures: tuple[CleanFigure, ...]) -> None:
    """
    Create a self contained relational database in a new temporary file
    """
    ordered = sorted(figures, key=lambda item: (item.working_group, item.figure_id))

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise RuntimeError("SQLite foreign-key enforcement is unavailable")
        connection.executescript(_CREATE_TABLES)
        connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")

        with connection:  
            connection.executemany(_INSERT_FIGURE, (_figure_row(f) for f in ordered))
            connection.executemany(
                "INSERT INTO figure_images (figure_id, position, image_url) "
                "VALUES (?, ?, ?)",
                (
                    (f.figure_id, position, url)
                    for f in ordered
                    for position, url in enumerate(f.image_urls)
                ),
            )

        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Database foreign-key check failed: {violations[:3]}")
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result}")
        written = connection.execute("SELECT COUNT(*) FROM figures").fetchone()[0]
        if written != len(figures):
            raise RuntimeError(f"Expected {len(figures)} figures in SQLite, found {written}")


def _quality_report(
    result: CleaningResult, input_counts: dict[str, int]
) -> dict[str, Any]:
    """
    Create source traceable, JSON serializable figures and issue counts
    """
    figures_by_wg = Counter(f.working_group for f in result.figures)
    missing_captions = Counter(
        f.working_group for f in result.figures if f.caption_status == "missing"
    )
    missing_images = Counter(
        f.working_group for f in result.figures if f.image_status == "missing"
    )
    sections = {
        wg: dict(sorted(Counter(
            f.section for f in result.figures if f.working_group == wg
        ).items()))
        for wg in WORKING_GROUPS
    }

    return {
        "report_type": "processed_corpus_quality",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_schema_version": "1.0",
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "source_files": {f"{wg.lower()}.json": input_counts[wg] for wg in WORKING_GROUPS},
        "summary": result.summary,
        "figures_by_working_group": {
            wg: {
                "total": figures_by_wg[wg],
                "missing_captions": missing_captions[wg],
                "missing_images": missing_images[wg],
                "sections": sections[wg],
            }
            for wg in WORKING_GROUPS
        },
        "figure_image_references": sum(len(f.image_urls) for f in result.figures),
        "issues": [
            {
                "index": issue.index,
                "code": issue.code,
                "figure_id": issue.figure_id,
                "detail": issue.detail,
            }
            for issue in result.issues
        ],
        "notes": [
            "Missing captions and image references are flagged, not invented or removed.",
            "Image status indicates a candidate URL, not a tested working download.",
            "This report covers collected JSON records; it does not prove full IPCC site coverage.",
        ],
    }


def build_corpus(
    raw_dir: Path = PROJECT_ROOT / "data" / "raw",
    database_path: Path = PROJECT_ROOT / "data" / "processed" / "corpus.db",
    report_path: Path = PROJECT_ROOT / "reports" / "corpus_quality.json",
) -> dict[str, Any]:
    """
    Validate all WG files, clean records and safely rebuild database / report
    """
    raw_dir = Path(raw_dir)
    database_path = Path(database_path)
    report_path = Path(report_path)

    records, input_counts = _read_raw_figures(raw_dir)
    result = clean_figures(records)

    invalid = [issue for issue in result.issues if issue.code == "invalid_record"]
    if invalid:
        sample = "; ".join(
            f"record {i.index} ({i.figure_id or 'unknown'}): {i.detail}"
            for i in invalid[:3]
        )
        raise ValueError(f"Cannot build corpus: {len(invalid)} invalid record(s). {sample}")
    if not result.figures:
        raise ValueError("Cannot build an empty corpus")

    report = _quality_report(result, input_counts)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".corpus-build-", dir=database_path.parent) as db_tmp:
        with tempfile.TemporaryDirectory(prefix=".corpus-report-", dir=report_path.parent) as report_tmp:
            pending_db = Path(db_tmp) / "corpus.db"
            pending_report = Path(report_tmp) / "corpus_quality.json"
            _write_database(pending_db, result.figures)
            with pending_report.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            os.replace(pending_db, database_path)
            os.replace(pending_report, report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the IPCC figure SQLite corpus from WG1/WG2/WG3 raw JSON files."
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--database", type=Path, default=PROJECT_ROOT / "data" / "processed" / "corpus.db"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "corpus_quality.json"
    )
    args = parser.parse_args()
    report = build_corpus(args.raw_dir, args.database, args.report)
    summary = report["summary"]
    print(
        f"Corpus built: {summary['cleaned_figures']} figures; "
        f"{summary['missing_captions']} missing captions; "
        f"{summary['missing_images']} missing image references; "
        f"{summary['removed_decorative_urls']} decorative URLs removed."
    )
    print(f"SQLite: {args.database}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

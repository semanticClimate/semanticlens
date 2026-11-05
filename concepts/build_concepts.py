"""Build an evidence-backed IPCC AR6 figure-to-glossary-concept index.

Reads the *processed* Step 2 corpus and Step 3 vocabulary, then runs the
existing lexical extractor and conservative normalizer on every available
caption. Creates a separate, queryable SQLite database and quality report.
Nothing here infers visual content, scientific causality, or ontology edges.

Run from the repository root (or any working directory):
    python -m concepts.build_concepts

Inputs are never modified. Both outputs are constructed and verified in
temporary files before being individually replaced; replacing two files is
not a single atomic transaction. No downloads or additional dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from concepts.extract import ConceptExtractor
from concepts.normalize import ConceptNormalizer
from concepts.quality import MentionQuality
from concepts.vocabulary import Vocabulary, load_vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.db"
DEFAULT_VOCABULARY = PROJECT_ROOT / "data" / "processed" / "ipcc_ar6_vocabulary.json"
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "processed" / "concepts.db"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "concept_quality.json"
WORKING_GROUPS = ("WG1", "WG2", "WG3")
DATABASE_SCHEMA_VERSION = 2
METHOD = "glossary_lexical_quality_v2"


@dataclass(frozen=True, slots=True)
class _Figure:
    figure_id: str
    working_group: str
    source_url: str
    caption: str | None
    caption_status: str


_CREATE_SCHEMA = """
CREATE TABLE concepts (
    term_id TEXT PRIMARY KEY,
    canonical_label TEXT NOT NULL,
    definition TEXT
);
CREATE TABLE concept_aliases (
    term_id TEXT NOT NULL REFERENCES concepts(term_id),
    alias TEXT NOT NULL,
    PRIMARY KEY (term_id, alias)
);
CREATE TABLE concept_working_groups (
    term_id TEXT NOT NULL REFERENCES concepts(term_id),
    working_group TEXT NOT NULL CHECK(working_group IN ('WG1','WG2','WG3')),
    PRIMARY KEY (term_id, working_group)
);
CREATE TABLE concept_source_series (
    term_id TEXT NOT NULL REFERENCES concepts(term_id),
    series TEXT NOT NULL,
    PRIMARY KEY (term_id, series)
);
CREATE TABLE figures (
    figure_id TEXT PRIMARY KEY,
    working_group TEXT NOT NULL CHECK(working_group IN ('WG1','WG2','WG3')),
    source_url TEXT NOT NULL UNIQUE,
    caption_status TEXT NOT NULL CHECK(caption_status IN ('available','missing')),
    caption_sha256 TEXT
);
CREATE TABLE mentions (
    mention_id INTEGER PRIMARY KEY,
    figure_id TEXT NOT NULL REFERENCES figures(figure_id),
    start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
    matched_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    match_kind TEXT NOT NULL CHECK(match_kind IN ('label','alias','mixed')),
    resolution TEXT NOT NULL CHECK(resolution IN ('lexical','explicit_expansion','ambiguous','chemical_formula','excluded')),
    quality_rule TEXT CHECK(quality_rule IN ('co2_formula','co2_equivalent_qualifier','citation_source_label')),
    term_id TEXT REFERENCES concepts(term_id),
    CHECK ((resolution IN ('ambiguous','excluded') AND term_id IS NULL) OR
           (resolution NOT IN ('ambiguous','excluded') AND term_id IS NOT NULL)),
    UNIQUE (figure_id, start_offset, end_offset)
);
CREATE TABLE mention_candidates (
    mention_id INTEGER NOT NULL REFERENCES mentions(mention_id) ON DELETE CASCADE,
    term_id TEXT NOT NULL REFERENCES concepts(term_id),
    PRIMARY KEY (mention_id, term_id)
);
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX idx_mentions_term ON mentions(term_id);
CREATE INDEX idx_mentions_figure ON mentions(figure_id);
CREATE INDEX idx_figures_wg ON figures(working_group);
CREATE INDEX idx_candidates_term ON mention_candidates(term_id);
-- A resolved figure-to-concept link exists only if a supported caption
-- mention was found. One figure can mention the same concept many times.
CREATE VIEW figure_concepts (figure_id, term_id, working_group) AS
SELECT DISTINCT m.figure_id, m.term_id, f.working_group
FROM mentions AS m
JOIN figures AS f ON f.figure_id = m.figure_id
WHERE m.term_id IS NOT NULL;
"""


def _sha256(path: Path) -> str:
    """Fingerprint source bytes for reproducibility and stale-output checks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_vocabulary(path: Path) -> tuple[Vocabulary, dict[str, Any], str]:
    """Use the existing vocabulary loader; retain its source provenance."""
    if not path.is_file():
        raise FileNotFoundError(f"Vocabulary JSON not found: {path}")
    checksum_before = _sha256(path)
    vocabulary = load_vocabulary(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _sha256(path) != checksum_before:
        raise ValueError("Vocabulary changed while the build was reading it")
    if not isinstance(payload, dict) or payload.get("vocabulary_id") != "ipcc-ar6-glossary":
        raise ValueError("Expected the IPCC AR6 glossary vocabulary JSON")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not all(
        isinstance(provenance.get(key), str) and provenance[key]
        for key in ("derived_export", "official_glossary", "source_sha256")
    ):
        raise ValueError("Vocabulary is missing source provenance")
    if payload.get("summary", {}).get("total_terms") != len(vocabulary.terms):
        raise ValueError("Vocabulary summary does not match its terms")
    return vocabulary, provenance, checksum_before


def _read_figures(path: Path) -> tuple[_Figure, ...]:
    """Read the validated Step 2 figures in a read-only SQLite connection."""
    if not path.is_file():
        raise FileNotFoundError(f"Processed figure corpus not found: {path}")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Source figure corpus failed SQLite quick_check")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != 1:
            raise ValueError(f"Expected Step 2 corpus database schema 1, found {version}")
        try:
            rows = db.execute(
                "SELECT figure_id, working_group, source_url, caption, caption_status "
                "FROM figures ORDER BY figure_id"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise ValueError("Source corpus is missing the Step 2 figures schema") from exc

    if not rows:
        raise ValueError("Source figure corpus contains no figures")
    figures: list[_Figure] = []
    for figure_id, wg, source_url, caption, status in rows:
        if not isinstance(figure_id, str) or not figure_id.strip():
            raise ValueError("Source corpus has an invalid figure_id")
        if wg not in WORKING_GROUPS:
            raise ValueError(f"Invalid working group for {figure_id}")
        if not isinstance(source_url, str) or not source_url.startswith(
            f"https://www.ipcc.ch/report/ar6/{wg.lower()}/figures/"
        ):
            raise ValueError(f"Invalid source provenance for {figure_id}")
        expected = "available" if isinstance(caption, str) and caption.strip() else "missing"
        if status != expected:
            raise ValueError(f"Inconsistent caption status for {figure_id}")
        figures.append(_Figure(figure_id, wg, source_url, caption, status))
    return tuple(figures)


def _populate_database(
    path: Path,
    vocabulary: Vocabulary,
    figures: tuple[_Figure, ...],
    metadata: dict[str, str],
) -> dict[str, Any]:
    """Create every source-backed concept, mention, and candidate association."""
    extractor = ConceptExtractor(vocabulary)
    normalizer = ConceptNormalizer(vocabulary)
    quality = MentionQuality(vocabulary)
    rule_counts: Counter[str] = Counter()
    summary: Counter[str] = Counter()
    by_wg: dict[str, Counter[str]] = {wg: Counter() for wg in WORKING_GROUPS}
    wg_concepts: dict[str, set[str]] = {wg: set() for wg in WORKING_GROUPS}
    ambiguity: Counter[str] = Counter()
    concept_frequency: Counter[str] = Counter()  # unique figure-concept pairs
    used_ids: set[str] = set()

    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        if db.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise RuntimeError("SQLite foreign-key enforcement is unavailable")
        db.executescript(_CREATE_SCHEMA)
        db.execute(f"PRAGMA user_version={DATABASE_SCHEMA_VERSION}")

        with db:
            db.executemany(
                "INSERT INTO concepts(term_id,canonical_label,definition) VALUES (?,?,?)",
                ((t.term_id, t.label, t.definition) for t in vocabulary.terms),
            )
            db.executemany(
                "INSERT INTO concept_aliases(term_id,alias) VALUES (?,?)",
                ((t.term_id, a) for t in vocabulary.terms for a in t.aliases),
            )
            db.executemany(
                "INSERT INTO concept_working_groups(term_id,working_group) VALUES (?,?)",
                ((t.term_id, wg) for t in vocabulary.terms for wg in t.working_groups),
            )
            db.executemany(
                "INSERT INTO concept_source_series(term_id,series) VALUES (?,?)",
                ((t.term_id, s) for t in vocabulary.terms for s in t.source_series),
            )
            db.executemany(
                "INSERT INTO metadata(key,value) VALUES (?,?)", sorted(metadata.items())
            )

            for figure in figures:
                wg = figure.working_group
                summary["total_figures"] += 1
                by_wg[wg]["total_figures"] += 1
                if figure.caption_status == "missing":
                    summary["missing_captions"] += 1
                    by_wg[wg]["missing_captions"] += 1
                else:
                    summary["available_captions"] += 1
                    by_wg[wg]["available_captions"] += 1

                caption_hash = (hashlib.sha256(figure.caption.encode("utf-8")).hexdigest()
                                if figure.caption_status == "available" else None)
                db.execute(
                    "INSERT INTO figures VALUES (?,?,?,?,?)",
                    (figure.figure_id, wg, figure.source_url, figure.caption_status, caption_hash),
                )

                mentions = quality.assess(
                    normalizer.normalize(
                        extractor.extract(figure.figure_id, figure.caption), figure.caption
                    ), figure.caption
                )
                if mentions:
                    summary["figures_with_mentions"] += 1
                    by_wg[wg]["figures_with_mentions"] += 1
                elif figure.caption_status == "available":
                    summary["captioned_figures_without_matches"] += 1
                    by_wg[wg]["captioned_figures_without_matches"] += 1

                figure_resolved: set[str] = set()
                for assessed in mentions:
                    mention = assessed.original
                    if assessed.rule is not None:
                        rule_counts[assessed.rule] += 1
                    summary["total_mentions"] += 1
                    by_wg[wg]["total_mentions"] += 1
                    if assessed.is_excluded:
                        summary["excluded_mentions"] += 1
                        by_wg[wg]["excluded_mentions"] += 1
                    elif assessed.is_ambiguous:
                        summary["ambiguous_mentions"] += 1
                        by_wg[wg]["ambiguous_mentions"] += 1
                        ambiguity[mention.normalized_text] += 1
                    else:
                        summary["resolved_mentions"] += 1
                        by_wg[wg]["resolved_mentions"] += 1
                        figure_resolved.add(assessed.term_id)
                        if assessed.resolution == "chemical_formula":
                            summary["formula_resolved_mentions"] += 1
                        if assessed.resolution == "explicit_expansion":
                            summary["explicit_expansion_mentions"] += 1
                    row = db.execute(
                        "INSERT INTO mentions(figure_id,start_offset,end_offset,"
                        "matched_text,normalized_text,match_kind,resolution,quality_rule,term_id) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (mention.figure_id, mention.start, mention.end,
                         mention.matched_text, mention.normalized_text,
                         mention.match_kind, assessed.resolution, assessed.rule, assessed.term_id),
                    )
                    db.executemany(
                        "INSERT INTO mention_candidates(mention_id,term_id) VALUES (?,?)",
                        ((row.lastrowid, term_id) for term_id in mention.candidate_term_ids),
                    )

                if figure_resolved:
                    summary["figures_with_resolved_concepts"] += 1
                    by_wg[wg]["figures_with_resolved_concepts"] += 1
                for term_id in figure_resolved:
                    concept_frequency[term_id] += 1
                    wg_concepts[wg].add(term_id)
                    used_ids.add(term_id)
                summary["figure_concept_links"] += len(figure_resolved)

        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("Built concept database has broken foreign keys")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Built concept database failed SQLite integrity_check")
        if db.execute("SELECT COUNT(*) FROM figures").fetchone()[0] != len(figures):
            raise RuntimeError("Not all figures were saved")
        if db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0] != len(vocabulary.terms):
            raise RuntimeError("Not all vocabulary concepts were saved")
        if db.execute("SELECT COUNT(*) FROM mentions").fetchone()[0] != summary["total_mentions"]:
            raise RuntimeError("Mention count mismatch")
        if db.execute("SELECT COUNT(*) FROM figure_concepts").fetchone()[0] != summary["figure_concept_links"]:
            raise RuntimeError("Figure-to-concept link count mismatch")

    summary["vocabulary_terms"] = len(vocabulary.terms)
    summary["distinct_resolved_concepts"] = len(used_ids)
    counts = (
        "total_figures", "available_captions", "missing_captions", "figures_with_mentions",
        "captioned_figures_without_matches", "figures_with_resolved_concepts",
        "total_mentions", "resolved_mentions", "ambiguous_mentions", "excluded_mentions",
        "formula_resolved_mentions", "explicit_expansion_mentions", "figure_concept_links", "vocabulary_terms",
        "distinct_resolved_concepts",
    )
    wg_counts = (
        "total_figures", "available_captions", "missing_captions", "figures_with_mentions",
        "captioned_figures_without_matches", "figures_with_resolved_concepts",
        "total_mentions", "resolved_mentions", "ambiguous_mentions", "excluded_mentions",
    )
    return {
        "summary": {key: summary[key] for key in counts},
        "by_working_group": {
            wg: {**{key: by_wg[wg][key] for key in wg_counts},
                 "distinct_resolved_concepts": len(wg_concepts[wg])}
            for wg in WORKING_GROUPS
        },
        "ambiguous_spellings_observed": dict(sorted(ambiguity.items())),
        "quality_rules_applied": dict(sorted(rule_counts.items())),
        "top_concepts_by_figures": [
            {"term_id": term_id, "label": vocabulary.get(term_id).label, "figures": count}
            for term_id, count in sorted(
                concept_frequency.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
    }


def build_concepts(
    corpus_path: Path = DEFAULT_CORPUS,
    vocabulary_path: Path = DEFAULT_VOCABULARY,
    database_path: Path = DEFAULT_DATABASE,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Build a reproducible SQLite concept index and a separate quality report.

    An invalid input or extraction error leaves existing output files intact.
    The source corpus and glossary JSON stay unchanged. Run again to rebuild
    from scratch; there are no stale/duplicate associations from prior runs.
    """
    corpus_path, vocabulary_path = Path(corpus_path), Path(vocabulary_path)
    database_path, report_path = Path(database_path), Path(report_path)
    paths = [p.resolve() for p in (corpus_path, vocabulary_path, database_path, report_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("All input and output paths must be distinct")

    vocabulary, provenance, vocabulary_sha = _read_vocabulary(vocabulary_path)
    figures = _read_figures(corpus_path)
    corpus_sha = _sha256(corpus_path)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "method": METHOD,
        "generated_at": generated_at,
        "corpus_sha256": corpus_sha,
        "vocabulary_sha256": vocabulary_sha,
        "vocabulary_source_sha256": provenance["source_sha256"],
        "vocabulary_derived_export": provenance["derived_export"],
        "vocabulary_official_glossary": provenance["official_glossary"],
    }

    database_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".concept-build-", dir=database_path.parent) as db_tmp:
        with tempfile.TemporaryDirectory(prefix=".concept-report-", dir=report_path.parent) as report_tmp:
            staged_db = Path(db_tmp) / "concepts.db"
            staged_report = Path(report_tmp) / "concept_quality.json"
            counts = _populate_database(staged_db, vocabulary, figures, metadata)
            # Detect a source changed during extraction; never publish stale links.
            if _sha256(corpus_path) != corpus_sha or _sha256(vocabulary_path) != vocabulary_sha:
                raise ValueError("Input corpus or vocabulary changed during the build")
            report = {
                "report_type": "concept_extraction_quality",
                "generated_at": generated_at,
                "method": METHOD,
                "database_schema_version": DATABASE_SCHEMA_VERSION,
                "sources": {
                    "corpus_sha256": corpus_sha,
                    "vocabulary_sha256": vocabulary_sha,
                    "vocabulary_provenance": provenance,
                },
                **counts,
                "notes": [
                    "Matches are literal glossary terms/aliases in captions, not verified visual content.",
                    "Ambiguous or excluded mentions have no selected concept and cannot create a figure_concepts link.",
                    "All original mention spans and candidates are retained for audit, including excluded citations.",
                    "CO2 formula disambiguation follows explicit IPCC notation; it is a heuristic requiring manual evaluation.",
                    "No precision or recall metric is claimed without an independently annotated gold sample.",
                    "Caption-missing figures are retained; their mentions are empty.",
                    "A figure-concept link records a caption mention, not a causal or ontology relation.",
                    "Concept extraction is vocabulary-limited; absence of a match is not absence of a topic.",
                ],
            }
            staged_report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            os.replace(staged_db, database_path)
            os.replace(staged_report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IPCC figure-to-concept caption evidence from Step 2 and Step 3."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_concepts(args.corpus, args.vocabulary, args.database, args.report)
    summary = report["summary"]
    print(
        f"Concept index built: {summary['total_figures']} figures; "
        f"{summary['available_captions']} captions; "
        f"{summary['resolved_mentions']} resolved mentions; "
        f"{summary['ambiguous_mentions']} ambiguous mentions; "
        f"{summary['figure_concept_links']} figure-concept links."
    )
    print(f"SQLite: {args.database}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

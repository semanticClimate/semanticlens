"""Reproducible, human-reviewed evaluation of caption-to-concept extraction.

Generate a balanced audit sample with `sample`, manually annotate the concepts
mentioned in each sampled caption, then run `score`. This deliberately does not
claim precision or recall from automatic rules alone.

Sample files include IPCC caption text for *local human review*. Do not commit,
redistribute, or publish them without checking IPCC reuse permissions.

Examples:
    python -m concepts.evaluate sample --size 90 --output review_sample.json
    python -m concepts.evaluate score --input review_sample.json

Annotation schema for every sampled figure:
    "reviewed": true,
    "gold_mentions": [{"start": 10, "end": 14, "term_id": "ipcc-ar6-..."}]

Annotate *every* glossary concept mention, including those the extractor missed
and wrong-ID matches. Empty gold_mentions means the reviewed caption contains
no glossary-relevant concept. This evaluates term-linked span extraction, not
visual accuracy, ontology relationships, or concepts absent from the glossary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from concepts.build_concepts import DEFAULT_CORPUS, DEFAULT_DATABASE, PROJECT_ROOT, _sha256


DEFAULT_SAMPLE = PROJECT_ROOT / "reports" / "concept_review_sample.json"
DEFAULT_EVALUATION = PROJECT_ROOT / "reports" / "concept_evaluation.json"
SCHEMA_VERSION = "1.0"


def _database(path: Path):
    """Connect to an existing, immutable concept index."""
    if not path.is_file():
        raise FileNotFoundError(f"Concept index not found: {path}")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _balanced_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Sample by WG and deliberately include ambiguity/exclusions/no matches."""
    if size < 3:
        raise ValueError("Sample size must be at least 3 (one per Working Group)")
    groups = ("WG1", "WG2", "WG3")
    pool = {wg: [r for r in rows if r["working_group"] == wg] for wg in groups}
    if any(not pool[wg] for wg in groups):
        raise ValueError("Every Working Group must contain captioned figures")
    size = min(size, len(rows))
    quotas = {wg: size // 3 + (index < size % 3) for index, wg in enumerate(groups)}
    # Allow reallocation if one WG has fewer captioned figures than its quota.
    while any(quotas[wg] > len(pool[wg]) for wg in groups):
        for wg in groups:
            if quotas[wg] > len(pool[wg]):
                deficit = quotas[wg] - len(pool[wg])
                quotas[wg] -= deficit
                for other in groups:
                    room = len(pool[other]) - quotas[other]
                    take = min(room, deficit)
                    quotas[other] += take
                    deficit -= take
                    if not deficit:
                        break
                if deficit:
                    raise ValueError("Not enough available figures")

    def order(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row['figure_id']}".encode()).hexdigest()

    chosen = []
    for wg in groups:
        available = sorted(pool[wg], key=order)
        local = []
        seen = set()
        # Oversample challenging cases; fill the rest without replacement.
        for category in ("ambiguous", "excluded", "no_matches", "generic", "other"):
            target = max(1, quotas[wg] // 4) if category != "other" else quotas[wg]
            candidates = [r for r in available if category == "other" or r["risk"] == category]
            for row in candidates:
                if len(local) >= quotas[wg] or (category != "other" and target <= 0):
                    break
                if row["figure_id"] in seen:
                    continue
                seen.add(row["figure_id"])
                local.append(row)
                target -= 1
        chosen.extend(local)
    return sorted(chosen, key=lambda row: (row["working_group"], row["figure_id"]))


def create_sample(
    corpus_path: Path = DEFAULT_CORPUS,
    database_path: Path = DEFAULT_DATABASE,
    output_path: Path = DEFAULT_SAMPLE,
    *,
    size: int = 90,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare local audit material, not gold labels or accuracy metrics."""
    corpus_path, database_path, output_path = map(Path, (corpus_path, database_path, output_path))
    if output_path.resolve() in {corpus_path.resolve(), database_path.resolve()}:
        raise ValueError("Sample output must be distinct from source files")
    if output_path.exists() and not force:
        raise FileExistsError("Review sample already exists; use --force only if you wish to overwrite annotations")
    with closing(_database(database_path)) as db:
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise ValueError("Expected the Step 3.5 concept index schema 2")
        source_sha = db.execute("SELECT value FROM metadata WHERE key='corpus_sha256'").fetchone()
        if not source_sha or source_sha[0] != _sha256(corpus_path):
            raise ValueError("Concept index was built from a different corpus; rebuild first")
        rows = [dict(r) for r in db.execute(
            "SELECT f.figure_id, f.working_group, f.source_url, "
            "SUM(CASE WHEN m.resolution='ambiguous' THEN 1 ELSE 0 END) AS ambig, "
            "SUM(CASE WHEN m.resolution='excluded' THEN 1 ELSE 0 END) AS excluded, "
            "COUNT(m.mention_id) AS mention_count, "
            "SUM(CASE WHEN c.canonical_label IN ('Models','Climate','Land','Region','Source') "
            "THEN 1 ELSE 0 END) AS generic_count "
            "FROM figures f LEFT JOIN mentions m ON f.figure_id=m.figure_id "
            "LEFT JOIN concepts c ON c.term_id=m.term_id "
            "WHERE f.caption_status='available' GROUP BY f.figure_id ORDER BY f.figure_id"
        )]
        for row in rows:
            row["risk"] = ("ambiguous" if row["ambig"] else
                           "excluded" if row["excluded"] else
                           "no_matches" if row["mention_count"] == 0 else
                           "generic" if row["generic_count"] else "other")
        selected = _balanced_sample(rows, size, seed)
        sample: dict[str, dict[str, Any]] = {}
        for row in selected:
            predictions = [dict(r) for r in db.execute(
                "SELECT m.start_offset AS start, m.end_offset AS end, m.matched_text, "
                "m.term_id, c.canonical_label, m.resolution, m.quality_rule "
                "FROM mentions m LEFT JOIN concepts c ON m.term_id=c.term_id "
                "WHERE m.figure_id=? ORDER BY m.start_offset", (row["figure_id"],)
            )]
            for prediction in predictions:
                if prediction["term_id"] is None:
                    candidates = db.execute(
                        "SELECT c.term_id,c.canonical_label FROM mention_candidates mc "
                        "JOIN concepts c ON c.term_id=mc.term_id "
                        "JOIN mentions m ON m.mention_id=mc.mention_id "
                        "WHERE m.figure_id=? AND m.start_offset=? ORDER BY c.term_id",
                        (row["figure_id"], prediction["start"]),
                    ).fetchall()
                    prediction["candidate_labels"] = [dict(c) for c in candidates]
            sample[row["figure_id"]] = {
                "figure_id": row["figure_id"],
                "working_group": row["working_group"],
                "source_url": row["source_url"],
                "risk": row["risk"],
                "predictions": predictions,
                "reviewed": False,
                "gold_mentions": [],
            }

    # Copy authoritative caption text from the Step 2 corpus for actual review.
    with closing(sqlite3.connect(corpus_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for figure_id, caption in db.execute(
            "SELECT figure_id, caption FROM figures WHERE caption_status='available'"
        ):
            if figure_id in sample:
                sample[figure_id]["caption"] = caption
    if any("caption" not in entry for entry in sample.values()):
        raise ValueError("A sampled caption is missing from source corpus")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": _sha256(corpus_path),
        "concepts_sha256": _sha256(database_path),
        "seed": seed,
        "sample_size": len(sample),
        "instructions": "Read full caption and annotate all glossary-backed mentions by character offsets; set reviewed=true only after completion. Keep this file local due to IPCC copyright.",
        "figures": list(sample.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def score_sample(
    sample_path: Path = DEFAULT_SAMPLE,
    database_path: Path = DEFAULT_DATABASE,
    output_path: Path = DEFAULT_EVALUATION,
) -> dict[str, Any]:
    """Compute exact span + concept-ID precision/recall from completed gold labels."""
    sample_path, database_path, output_path = map(Path, (sample_path, database_path, output_path))
    if output_path.resolve() in {sample_path.resolve(), database_path.resolve()}:
        raise ValueError("Evaluation output must be distinct from inputs")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("figures"), list):
        raise ValueError("Unsupported or malformed review sample")
    if payload.get("sample_size") != len(payload["figures"]) or not payload["figures"]:
        raise ValueError("Review sample size is missing or inconsistent")
    if payload.get("concepts_sha256") != _sha256(database_path):
        raise ValueError("Concept database has changed since sampling; regenerate sample")
    with closing(_database(database_path)) as db:
        existing_ids = {r[0] for r in db.execute("SELECT term_id FROM concepts")}
        confusion: Counter[str] = Counter()
        by_wg: dict[str, Counter[str]] = {wg: Counter() for wg in ("WG1", "WG2", "WG3")}
        seen_figures = set()
        for row in payload["figures"]:
            fig_id, caption, wg = row.get("figure_id"), row.get("caption"), row.get("working_group")
            if fig_id in seen_figures or not isinstance(caption, str) or wg not in by_wg:
                raise ValueError("Invalid or duplicate figure in review sample")
            seen_figures.add(fig_id)
            record = db.execute("SELECT working_group,caption_sha256,caption_status FROM figures WHERE figure_id=?", (fig_id,)).fetchone()
            if record is None or record[0] != wg or record[2] != "available":
                raise ValueError("Review figure does not exist in the database")
            if hashlib.sha256(caption.encode("utf-8")).hexdigest() != record[1]:
                raise ValueError(f"Caption was modified in the review sample: {fig_id}")
            if row.get("reviewed") is not True:
                raise ValueError(f"Sample incomplete: {fig_id} has not been reviewed")
            if not isinstance(row.get("gold_mentions"), list):
                raise ValueError(f"Invalid gold mentions for {fig_id}")
            gold = set()
            for g in row["gold_mentions"]:
                if not isinstance(g, dict):
                    raise ValueError("Gold annotation must be an object")
                start, end, term_id = g.get("start"), g.get("end"), g.get("term_id")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(caption) or term_id not in existing_ids):
                    raise ValueError(f"Invalid gold annotation in {fig_id}")
                gold.add((start, end, term_id))
            if len(gold) != len(row["gold_mentions"]):
                raise ValueError(f"Repeated gold annotation in {fig_id}")
            predicted = set(db.execute(
                "SELECT start_offset,end_offset,term_id FROM mentions "
                "WHERE figure_id=? AND term_id IS NOT NULL", (fig_id,)
            ).fetchall())
            tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
            for counts in (confusion, by_wg[wg]):
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn
                counts["figures"] += 1

    def metrics(c: Counter[str]) -> dict[str, float | int | None]:
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        return {"reviewed_figures": c["figures"], "true_positives": tp,
                "false_positives": fp, "false_negatives": fn,
                "precision": precision, "recall": recall, "f1": f1}

    result = {
        "report_type": "manual_gold_span_evaluation",
        "metric_definition": "Exact figure, original caption span, and glossary term ID match on manually annotated captions",
        "sample_size": len(seen_figures),
        "overall": metrics(confusion),
        "by_working_group": {wg: metrics(c) for wg, c in by_wg.items()},
        "limitations": [
            "Valid only for the reviewed, deliberately risk-oversampled sample; not an unbiased corpus-wide accuracy estimate.",
            "Measures glossary-linked caption mentions, not visual depiction or ontology relations.",
            "A missing concept outside the provided glossary cannot be counted as a glossary recall error.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Human evaluation of IPCC caption concept mappings")
    commands = parser.add_subparsers(dest="action", required=True)
    sample = commands.add_parser("sample", help="Produce local stratified sample for manual annotation")
    sample.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    sample.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    sample.add_argument("--output", type=Path, default=DEFAULT_SAMPLE)
    sample.add_argument("--size", type=int, default=90)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--force", action="store_true", help="Overwrite existing annotation file")
    score = commands.add_parser("score", help="Score completely reviewed gold annotations")
    score.add_argument("--input", type=Path, default=DEFAULT_SAMPLE)
    score.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    score.add_argument("--output", type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    if args.action == "sample":
        output = create_sample(args.corpus, args.database, args.output, size=args.size, seed=args.seed, force=args.force)
        print(f"Created {output['sample_size']} review cases in {args.output}")
        print("Keep the file local. Annotate all captions and set reviewed=true before score.")
    else:
        result = score_sample(args.input, args.database, args.output)
        print(f"Gold evaluation saved: {args.output}; results: {result['overall']}")


if __name__ == "__main__":
    main()

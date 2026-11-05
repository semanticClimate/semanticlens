"""Build the source-backed bipartite IPCC AR6 Figure--Concept RDF graph.

Only Figure and ScientificConcept resources are created. The sole
instance-to-instance edge, ipcc:mentionsConcept, records an accepted glossary
term match in a figure caption; it does not claim visual depiction or causality.

Figure image URLs and report location are literal properties, not extra nodes.
Caption-match spans, candidates and exclusions remain in concepts.db and are
validated here before accepted links are published. Original captions are not
exported unless explicitly requested and permitted by source reuse terms.

Run from the project root:
    python -m ontology.relationship
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from urllib.parse import urlsplit

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, RDF, SKOS, XSD

from ontology.schema import IPCC, RESOURCE, resource_iri


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.db"
DEFAULT_CONCEPTS = PROJECT_ROOT / "data" / "processed" / "concepts.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "ontology" / "ipcc_relationships.ttl"
WORKING_GROUPS = ("WG1", "WG2", "WG3")
_RESOLVED = frozenset({"lexical", "explicit_expansion", "chemical_formula"})
_UNRESOLVED = frozenset({"ambiguous", "excluded"})
_MATCH_KINDS = frozenset({"label", "alias", "mixed"})


@dataclass(frozen=True, slots=True)
class RelationshipResult:
    """Bipartite instance graph and source-quality counts."""

    graph: Graph
    summary: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_url(value: str, *, field: str, allow_http: bool = False) -> str:
    """Validate reference URLs without downloading them."""
    schemes = {"https", "http"} if allow_http else {"https"}
    if not isinstance(value, str):
        raise ValueError(f"Invalid {field}: expected {'HTTP(S)' if allow_http else 'HTTPS'} URL")
    parts = urlsplit(value)
    if parts.scheme not in schemes or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"Invalid {field}: expected {'HTTP(S)' if allow_http else 'HTTPS'} URL")
    return value


def _open_readonly(path: Path, expected_version: int) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"SQLite integrity check failed: {path}")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != expected_version:
            raise ValueError(f"Expected SQLite schema v{expected_version} in {path}; found v{version}")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError(f"Foreign key violations in {path}")
        return connection
    except BaseException:
        connection.close()
        raise


def _prefixes(graph: Graph) -> None:
    for prefix, namespace in (
        ("ipcc", IPCC), ("resource", RESOURCE),
        ("skos", SKOS), ("dcterms", DCTERMS), ("xsd", XSD),
    ):
        graph.bind(prefix, namespace)


def _add_concepts(graph: Graph, db: sqlite3.Connection, glossary_source: URIRef) -> set[str]:
    """Create concept nodes, labels and aliases; no attribution nodes."""
    known: set[str] = set()
    for term_id, label in db.execute(
        "SELECT term_id, canonical_label FROM concepts ORDER BY term_id"
    ):
        if not isinstance(term_id, str) or not term_id or not isinstance(label, str) or not label.strip():
            raise ValueError("Invalid glossary concept identity or label")
        node = resource_iri("concept", term_id)
        known.add(term_id)
        graph.add((node, RDF.type, IPCC.ScientificConcept))
        graph.add((node, DCTERMS.identifier, Literal(term_id)))
        graph.add((node, SKOS.prefLabel, Literal(label, lang="en")))
        graph.add((node, DCTERMS.source, glossary_source))

    if not known:
        raise ValueError("Concept database is empty")

    for term_id, alias in db.execute(
        "SELECT term_id, alias FROM concept_aliases ORDER BY term_id, alias"
    ):
        if term_id not in known or not isinstance(alias, str) or not alias.strip():
            raise ValueError("Invalid concept alias")
        graph.add((resource_iri("concept", term_id), SKOS.altLabel, Literal(alias, lang="en")))

    # The original glossary attributions stay in SQLite. Validate them, but do
    # not create WorkingGroup nodes or glossaryWorkingGroup relationships.
    for term_id, wg in db.execute(
        "SELECT term_id, working_group FROM concept_working_groups ORDER BY term_id, working_group"
    ):
        if term_id not in known or wg not in WORKING_GROUPS:
            raise ValueError("Invalid glossary working group attribution")
    return known


def _add_figures(
    graph: Graph, corpus: sqlite3.Connection, concepts: sqlite3.Connection,
    *, include_captions: bool,
) -> tuple[dict[str, str | None], dict[str, str], dict[str, str], int]:
    """Create Figure nodes and validate matching source records and image URLs."""
    caption_by_id: dict[str, str | None] = {}
    group_by_id: dict[str, str] = {}
    url_by_id: dict[str, str] = {}
    query = ("SELECT figure_id, working_group, section, caption, caption_status, "
             "image_status, source_url FROM figures ORDER BY figure_id")
    for f in corpus.execute(query):
        fid, wg, section = f["figure_id"], f["working_group"], f["section"]
        caption, status, image_status = f["caption"], f["caption_status"], f["image_status"]
        url = f["source_url"]
        if not isinstance(fid, str) or not fid or wg not in WORKING_GROUPS:
            raise ValueError(f"Invalid figure or working group: {fid!r}")
        if not isinstance(section, str) or not section.strip():
            raise ValueError(f"Invalid section for {fid}")
        _source_url(url, field=f"figure source URL for {fid}")
        if not url.startswith(f"https://www.ipcc.ch/report/ar6/{wg.lower()}/figures/"):
            raise ValueError(f"Figure source and working group disagree: {fid}")
        if status != ("available" if isinstance(caption, str) and caption.strip() else "missing"):
            raise ValueError(f"Inconsistent caption status for {fid}")
        if image_status not in {"referenced", "missing"}:
            raise ValueError(f"Invalid image status for {fid}")

        fig = resource_iri("figure", fid)
        graph.add((fig, RDF.type, IPCC.Figure))
        graph.add((fig, DCTERMS.identifier, Literal(fid)))
        graph.add((fig, DCTERMS.source, URIRef(url)))
        graph.add((fig, IPCC.workingGroupCode, Literal(wg)))
        graph.add((fig, IPCC.sectionCode, Literal(section)))
        graph.add((fig, IPCC.captionStatus, Literal(status)))
        if caption is not None and status == "available":
            graph.add((fig, IPCC.captionSHA256,
                       Literal(hashlib.sha256(caption.encode("utf-8")).hexdigest())))
            if include_captions:
                graph.add((fig, IPCC.captionText, Literal(caption)))

        caption_by_id[fid] = caption
        group_by_id[fid] = wg
        url_by_id[fid] = url

    # Reject stale concepts.db input rather than silently building bad links.
    registered: set[str] = set()
    for f in concepts.execute(
        "SELECT figure_id, working_group, source_url, caption_status, caption_sha256 "
        "FROM figures ORDER BY figure_id"
    ):
        fid = f["figure_id"]
        if fid not in caption_by_id:
            raise ValueError(f"Concept DB references unknown figure: {fid}")
        source_caption = caption_by_id[fid]
        expected_sha = (hashlib.sha256(source_caption.encode("utf-8")).hexdigest()
                        if source_caption and source_caption.strip() else None)
        if (f["working_group"] != group_by_id[fid] or f["source_url"] != url_by_id[fid]
                or f["caption_status"] != ("available" if expected_sha else "missing")
                or f["caption_sha256"] != expected_sha):
            raise ValueError(f"Stale or inconsistent concept figure record: {fid}")
        registered.add(fid)
    if not caption_by_id or registered != set(caption_by_id):
        raise ValueError("Figure sets disagree across corpus and concept databases")

    images_count = 0
    image_figures: set[str] = set()
    for fid, position, url in corpus.execute(
        "SELECT figure_id, position, image_url FROM figure_images ORDER BY figure_id, position"
    ):
        if fid not in caption_by_id or not isinstance(position, int) or position < 0:
            raise ValueError("Invalid image reference identity")
        _source_url(url, field="image reference URL", allow_http=True)
        # Image URLs are literal Figure properties, NOT separate image nodes.
        graph.add((resource_iri("figure", fid), IPCC.imageURL, Literal(url, datatype=XSD.anyURI)))
        images_count += 1
        image_figures.add(fid)

    for fid, status in corpus.execute("SELECT figure_id, image_status FROM figures"):
        if (fid in image_figures) != (status == "referenced"):
            raise ValueError(f"Image references disagree with status for {fid}")
    return caption_by_id, group_by_id, url_by_id, images_count


def _add_mentions(
    graph: Graph, concepts: sqlite3.Connection,
    captions: dict[str, str | None], known_terms: set[str],
) -> dict[str, int]:
    """Validate caption evidence; emit only accepted Figure--Concept links."""
    candidate_by_mention: dict[int, set[str]] = {}
    for mention_id, term_id in concepts.execute(
        "SELECT mention_id, term_id FROM mention_candidates ORDER BY mention_id, term_id"
    ):
        if term_id not in known_terms:
            raise ValueError(f"Unknown candidate concept ID: {term_id}")
        candidate_by_mention.setdefault(mention_id, set()).add(term_id)

    counts: Counter[str] = Counter()
    accepted_pairs: set[tuple[str, str]] = set()
    seen_candidates: set[int] = set()
    query = ("SELECT mention_id, figure_id, start_offset, end_offset, matched_text, "
             "match_kind, resolution, quality_rule, term_id FROM mentions "
             "ORDER BY figure_id, start_offset, end_offset, mention_id")
    for m in concepts.execute(query):
        mid, fid, start, end, matched = (m["mention_id"], m["figure_id"],
                                         m["start_offset"], m["end_offset"], m["matched_text"])
        caption = captions.get(fid)
        if (caption is None or not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or end <= start or end > len(caption)
                or caption[start:end] != matched):
            raise ValueError(f"Caption evidence offset/text mismatch at mention {mid}")
        if m["match_kind"] not in _MATCH_KINDS or m["resolution"] not in _RESOLVED | _UNRESOLVED:
            raise ValueError(f"Invalid mention classification: {mid}")

        selected = m["term_id"]
        candidates = candidate_by_mention.get(mid, set())
        if not candidates or (selected is not None and selected not in candidates):
            raise ValueError(f"Invalid candidate set for mention {mid}")
        if m["resolution"] in _UNRESOLVED and selected is not None:
            raise ValueError(f"Unresolved mention {mid} has an accepted concept")
        if m["resolution"] in _RESOLVED and selected not in known_terms:
            raise ValueError(f"Resolved mention {mid} lacks a known concept")

        # Ambiguous/excluded mention records are checked but never emitted as
        # RDF nodes or Figure--Concept edges.
        seen_candidates.add(mid)
        if selected is not None:
            fig = resource_iri("figure", fid)
            concept = resource_iri("concept", selected)
            graph.add((fig, IPCC.mentionsConcept, concept))
            accepted_pairs.add((fid, selected))
            counts["resolved_mentions"] += 1
        else:
            counts["unresolved_mentions"] += 1
            counts[f"{m['resolution']}_mentions"] += 1
        counts["total_mentions"] += 1

    if seen_candidates != set(candidate_by_mention):
        raise ValueError("Candidate table contains an orphan mention")
    db_pairs = concepts.execute("SELECT COUNT(*) FROM figure_concepts").fetchone()[0]
    if db_pairs != len(accepted_pairs):
        raise ValueError("Resolved figure-concept links disagree with the Step 3 source view")
    counts["figure_concept_links"] = len(accepted_pairs)
    counts["candidate_links"] = sum(map(len, candidate_by_mention.values()))
    return dict(counts)


def build_relationship_graph(
    corpus_path: Path = DEFAULT_CORPUS,
    concepts_path: Path = DEFAULT_CONCEPTS,
    *,
    include_captions: bool = False,
) -> RelationshipResult:
    """Build a verified bipartite instance graph without changing input DBs."""
    corpus_path, concepts_path = Path(corpus_path), Path(concepts_path)
    if corpus_path.resolve() == concepts_path.resolve():
        raise ValueError("Corpus and concept DB must be distinct files")
    corpus_digest = _sha256(corpus_path)
    concepts_digest = _sha256(concepts_path)
    graph = Graph()
    _prefixes(graph)

    with closing(_open_readonly(corpus_path, 1)) as corpus:
        with closing(_open_readonly(concepts_path, 2)) as concepts:
            meta = {key: value for key, value in concepts.execute("SELECT key, value FROM metadata")}
            if meta.get("corpus_sha256") != corpus_digest:
                raise ValueError("Concept database was built from a different corpus.db; rebuild Step 3")
            if not meta.get("vocabulary_sha256"):
                raise ValueError("Concept DB is missing vocabulary provenance")
            glossary = URIRef(_source_url(meta.get("vocabulary_derived_export"), field="glossary source"))
            _source_url(meta.get("vocabulary_official_glossary"), field="official glossary")

            known = _add_concepts(graph, concepts, glossary)
            captions, _, _, images_count = _add_figures(
                graph, corpus, concepts, include_captions=include_captions
            )
            mentions = _add_mentions(graph, concepts, captions, known)

    if _sha256(corpus_path) != corpus_digest or _sha256(concepts_path) != concepts_digest:
        raise ValueError("Source database changed while relationships were being built")
    summary = {
        "figures": len(captions),
        "figures_with_captions": sum(bool(c) for c in captions.values()),
        "concepts": len(known),
        "image_references": images_count,
        **mentions,
        "rdf_triples": len(graph),
    }
    return RelationshipResult(graph=graph, summary=summary)


def write_relationships(
    corpus_path: Path = DEFAULT_CORPUS,
    concepts_path: Path = DEFAULT_CONCEPTS,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    include_captions: bool = False,
) -> dict[str, int]:
    """Atomically export Turtle after checking an exact RDF round-trip."""
    corpus_path, concepts_path, output_path = map(Path, (corpus_path, concepts_path, output_path))
    if len({p.resolve() for p in (corpus_path, concepts_path, output_path)}) != 3:
        raise ValueError("Inputs and RDF output must have distinct paths")
    result = build_relationship_graph(corpus_path, concepts_path, include_captions=include_captions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".relationships-", suffix=".ttl", dir=output_path.parent,
        delete=False,
    ) as handle:
        staged = Path(handle.name)
    try:
        result.graph.serialize(destination=staged, format="turtle", encoding="utf-8")
        parsed = Graph().parse(staged, format="turtle")
        if len(parsed) != len(result.graph) or set(parsed) != set(result.graph):
            raise RuntimeError("Exported Turtle did not round-trip exactly")
        os.replace(staged, output_path)
    finally:
        staged.unlink(missing_ok=True)
    return result.summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export source-backed IPCC AR6 Figure--Concept RDF links."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-captions", action="store_true",
        help="Include full IPCC captions in RDF (review source reuse permissions first).",
    )
    args = parser.parse_args()
    summary = write_relationships(
        args.corpus, args.concepts, args.output, include_captions=args.include_captions
    )
    print(
        f"Relationships: {summary['figures']} figures; {summary['concepts']} concepts; "
        f"{summary['figure_concept_links']} accepted Figure--Concept links; "
        f"{summary['rdf_triples']} RDF triples."
    )
    print(f"Turtle: {args.output}")


if __name__ == "__main__":
    main()

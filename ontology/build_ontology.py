"""
Builds and validates the IPCC visual knowledge graph

Combines the ontology schema with source-backed figure-concept relationships
and exports the RDF graph with a quality report
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

from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, SH, SKOS

from concepts.vocabulary import load_vocabulary
from ontology.relationship import (
    DEFAULT_CONCEPTS,
    DEFAULT_CORPUS,
    WORKING_GROUPS,
    build_relationship_graph,
)
from ontology.schema import (
    IPCC,
    RESOURCE,
    build_ontology_schema,
    build_validation_shapes,
    resource_iri,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCABULARY = PROJECT_ROOT / "data" / "processed" / "ipcc_ar6_vocabulary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "ontology" / "ipcc_visual_knowledge_graph.ttl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "ontology_quality.json"
BUILD_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class OntologyResult:
    """
    Store the completed RDF graph, validation shapes, and quality report
    """

    graph: Graph
    shapes: Graph
    report: dict[str, Any]


def _sha256(path: Path) -> str:
    """
    Calculate the SHA-256 checksum of a required file
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(corpus: Path, concepts: Path, vocabulary: Path) -> dict[str, str]:
    """
    Calculate checksums for the corpus, concept database, and vocabulary files
    """
    return {
        "corpus_sha256": _sha256(corpus),
        "concepts_sha256": _sha256(concepts),
        "vocabulary_sha256": _sha256(vocabulary),
    }


def _verify_concept_sources(
    concepts: Path, hashes: dict[str, str], vocabulary: Any
) -> None:
    """
    Check that the concept database matches the selected corpus and vocabulary

    Verify database integrity, concept identities, and source checksums
    """
    with closing(
        sqlite3.connect(concepts.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Concept SQLite integrity check failed")
        if db.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise ValueError(
                "Step 3 concepts.db must use schema version 2; rebuild Step 3.5"
            )
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        observed = dict(db.execute("SELECT term_id, canonical_label FROM concepts"))
        expected = {term.term_id: term.label for term in vocabulary.terms}
        if observed != expected:
            raise ValueError(
                "Concept identities/labels disagree with the selected vocabulary"
            )
    for key in ("corpus_sha256", "vocabulary_sha256"):
        if metadata.get(key) != hashes[key]:
            raise ValueError(
                f"Step 3 concepts.db {key} does not match the selected input; "
                "rebuild using python -m concepts.build_concepts"
            )


def _verify_graph(
    graph: Graph,
    expected: dict[str, int],
    corpus_path: Path,
    concepts_path: Path,
) -> dict[str, Any]:
    """
    Check the RDF graph against the source databases and ontology structure

    Validate figures, concepts, properties, and relationships
    and summarize results by working group
    """
    figures = set(graph.subjects(RDF.type, IPCC.Figure))
    concepts = set(graph.subjects(RDF.type, IPCC.ScientificConcept))
    if len(figures) != expected["figures"] or len(concepts) != expected["concepts"]:
        raise ValueError("RDF figure/concept counts disagree with the source databases")
    if figures & concepts:
        raise ValueError("A resource cannot be both a Figure and ScientificConcept")

    if set(graph.subjects(RDF.type, OWL.Class)) != {
        IPCC.Figure,
        IPCC.ScientificConcept,
    } or set(graph.subjects(RDF.type, OWL.ObjectProperty)) != {IPCC.mentionsConcept}:
        raise ValueError("Ontology schema contains unwanted classes or relationships")
    entity_subjects = {
        subject
        for subject in graph.subjects()
        if isinstance(subject, URIRef) and str(subject).startswith(str(RESOURCE))
    }
    if entity_subjects != figures | concepts:
        raise ValueError("RDF contains unexpected instance resources")

    figure_properties = {
        RDF.type,
        DCTERMS.identifier,
        DCTERMS.source,
        IPCC.workingGroupCode,
        IPCC.sectionCode,
        IPCC.captionStatus,
        IPCC.captionSHA256,
        IPCC.captionText,
        IPCC.imageURL,
        IPCC.mentionsConcept,
    }
    concept_properties = {
        RDF.type,
        DCTERMS.identifier,
        DCTERMS.source,
        SKOS.prefLabel,
        SKOS.altLabel,
    }
    for subject in figures | concepts:
        properties = figure_properties if subject in figures else concept_properties
        kind = IPCC.Figure if subject in figures else IPCC.ScientificConcept
        if set(graph.objects(subject, RDF.type)) != {kind}:
            raise ValueError(f"Unexpected RDF type for {subject}")
        for predicate, obj in graph.predicate_objects(subject):
            if predicate not in properties:
                raise ValueError(
                    f"Unexpected instance relationship/property: {predicate}"
                )
            if isinstance(obj, URIRef) and predicate not in {
                RDF.type,
                DCTERMS.source,
                IPCC.mentionsConcept,
            }:
                raise ValueError(f"Unexpected resource-valued attribute: {predicate}")

    with closing(
        sqlite3.connect(corpus_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as corpus:
        source_figures = {
            row[0]: (row[1], row[2], row[3])
            for row in corpus.execute(
                "SELECT figure_id, working_group, section, source_url FROM figures"
            )
        }
        source_images: dict[str, set[str]] = {}
        for fid, url in corpus.execute(
            "SELECT figure_id, image_url FROM figure_images"
        ):
            source_images.setdefault(fid, set()).add(url)

    if figures != {resource_iri("figure", fid) for fid in source_figures}:
        raise ValueError("RDF figures do not match corpus.db")
    for fid, (wg, section, source_url) in source_figures.items():
        node = resource_iri("figure", fid)
        if (
            set(map(str, graph.objects(node, DCTERMS.identifier))) != {fid}
            or set(map(str, graph.objects(node, DCTERMS.source))) != {source_url}
            or set(map(str, graph.objects(node, IPCC.workingGroupCode))) != {wg}
            or set(map(str, graph.objects(node, IPCC.sectionCode))) != {section}
            or set(map(str, graph.objects(node, IPCC.imageURL)))
            != source_images.get(fid, set())
        ):
            raise ValueError(f"Figure metadata/images disagree with corpus.db: {fid}")

    with closing(
        sqlite3.connect(concepts_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        source_concepts = dict(
            db.execute("SELECT term_id, canonical_label FROM concepts")
        )
        source_pairs = {
            (resource_iri("figure", fid), resource_iri("concept", term_id))
            for fid, term_id in db.execute(
                "SELECT figure_id, term_id FROM figure_concepts"
            )
        }
    if concepts != {resource_iri("concept", term_id) for term_id in source_concepts}:
        raise ValueError("RDF concepts do not match concepts.db")
    for term_id, label in source_concepts.items():
        node = resource_iri("concept", term_id)
        if set(map(str, graph.objects(node, DCTERMS.identifier))) != {term_id} or set(
            map(str, graph.objects(node, SKOS.prefLabel))
        ) != {label}:
            raise ValueError(
                f"Concept label/identifier disagrees with concepts.db: {term_id}"
            )
    graph_pairs = set(graph.subject_objects(IPCC.mentionsConcept))
    if (
        graph_pairs != source_pairs
        or len(graph_pairs) != expected["figure_concept_links"]
        or any(
            fig not in figures or concept not in concepts
            for fig, concept in graph_pairs
        )
    ):
        raise ValueError(
            "RDF Figure--Concept links disagree with accepted SQLite pairs"
        )

    counts = Counter(wg for wg, _, _ in source_figures.values())
    pairs_by_wg: Counter[str] = Counter()
    concepts_by_wg: dict[str, set[URIRef]] = {wg: set() for wg in WORKING_GROUPS}
    for figure, concept in graph_pairs:
        wg = str(graph.value(figure, IPCC.workingGroupCode))
        pairs_by_wg[wg] += 1
        concepts_by_wg[wg].add(concept)
    return {
        "by_working_group": {
            wg: {
                "figures": counts[wg],
                "figure_concept_links": pairs_by_wg[wg],
                "distinct_linked_concepts": len(concepts_by_wg[wg]),
            }
            for wg in WORKING_GROUPS
        }
    }


def _validate_shacl(graph: Graph, shapes: Graph) -> None:
    """
    Validate the RDF graph against the SHACL shapes using pyshacl
    """
    try:
        from pyshacl import validate
    except ImportError as exc:
        raise RuntimeError(
            'Formal SHACL validation needs pyshacl. Install: pip install "pyshacl>=0.30,<1"'
        ) from exc
    conforms, _, description = validate(
        data_graph=graph,
        shacl_graph=shapes,
        inference="rdfs",
        advanced=False,
        abort_on_first=False,
        meta_shacl=False,
    )
    if not conforms:
        raise ValueError(
            f"RDF graph failed SHACL validation: {str(description)[:2000]}"
        )


def build_knowledge_graph(
    corpus_path: Path = DEFAULT_CORPUS,
    concepts_path: Path = DEFAULT_CONCEPTS,
    vocabulary_path: Path = DEFAULT_VOCABULARY,
    *,
    include_captions: bool = False,
    validate_shacl: bool = False,
) -> OntologyResult:
    """
    Build the complete IPCC visual knowledge graph from validated source files

    Combine the ontology schema and relationships, check graph integrity,
    and prepare a quality report
    """
    corpus_path, concepts_path, vocabulary_path = map(
        Path, (corpus_path, concepts_path, vocabulary_path)
    )
    source_paths = (corpus_path, concepts_path, vocabulary_path)
    if len({p.resolve() for p in source_paths}) != 3:
        raise ValueError(
            "Corpus, concepts DB and vocabulary JSON must be distinct files"
        )
    hashes = _source_hashes(*source_paths)
    vocabulary = load_vocabulary(vocabulary_path)
    _verify_concept_sources(concepts_path, hashes, vocabulary)
    relationship = build_relationship_graph(
        corpus_path, concepts_path, include_captions=include_captions
    )
    if relationship.summary["concepts"] != len(vocabulary.terms):
        raise ValueError("Concept DB and vocabulary disagree on glossary term count")

    schema = build_ontology_schema()
    shapes = build_validation_shapes()
    graph = schema + relationship.graph
    graph.bind("ipcc", IPCC, replace=True)
    graph.bind("skos", SKOS, replace=True)

    integrity = _verify_graph(graph, relationship.summary, corpus_path, concepts_path)
    if validate_shacl:
        _validate_shacl(graph, shapes)
    if _source_hashes(*source_paths) != hashes:
        raise ValueError(
            "An input was modified during ontology construction; rebuild from stable sources"
        )

    report = {
        "report_type": "ontology_build_quality",
        "build_version": BUILD_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            **hashes,
            "corpus_path": str(corpus_path),
            "concepts_path": str(concepts_path),
            "vocabulary_path": str(vocabulary_path),
        },
        "summary": {
            **relationship.summary,
            "rdf_triples": len(graph),
            "schema_triples": len(schema),
            "relationship_triples": len(relationship.graph),
        },
        **integrity,
        "validation": {
            "rdf_integrity": "passed",
            "shacl": "passed" if validate_shacl else "not_run",
            "shacl_shapes": len(set(shapes.subjects(RDF.type, SH.NodeShape))),
        },
        "notes": [
            "The only instance-to-instance edge is Figure -> mentionsConcept -> ScientificConcept.",
            "Caption mentions and exclusion candidates remain in concepts.db, not as graph nodes.",
            "Image URLs, working group and section are Figure attributes, not graph nodes.",
            "Accepted Figure--Concept links are caption matches, not proof of image depiction or causality.",
            "Full captions and glossary definitions are omitted unless explicitly permitted and opted in.",
            "Scientific extraction accuracy needs human review; RDF structural validity is not scientific validity.",
            "Published IPCC material and glossary-derived content may have reuse restrictions.",
        ],
    }
    return OntologyResult(graph=graph, shapes=shapes, report=report)


def write_ontology(
    corpus_path: Path = DEFAULT_CORPUS,
    concepts_path: Path = DEFAULT_CONCEPTS,
    vocabulary_path: Path = DEFAULT_VOCABULARY,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    *,
    include_captions: bool = False,
    validate_shacl: bool = False,
) -> dict[str, Any]:
    """
    Build and export the knowledge graph and its quality report

    Verify the serialized RDF and check source files before publication
    """
    corpus_path, concepts_path, vocabulary_path, output_path, report_path = map(
        Path, (corpus_path, concepts_path, vocabulary_path, output_path, report_path)
    )
    paths = (corpus_path, concepts_path, vocabulary_path, output_path, report_path)
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("All input and output paths must be different")
    result = build_knowledge_graph(
        corpus_path,
        concepts_path,
        vocabulary_path,
        include_captions=include_captions,
        validate_shacl=validate_shacl,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ontology-build-", dir=output_path.parent
    ) as graph_tmp:
        with tempfile.TemporaryDirectory(
            prefix=".ontology-report-", dir=report_path.parent
        ) as report_tmp:
            pending_graph = Path(graph_tmp) / "knowledge_graph.ttl"
            pending_report = Path(report_tmp) / "ontology_quality.json"
            result.graph.serialize(
                destination=pending_graph, format="turtle", encoding="utf-8"
            )
            reloaded = Graph().parse(pending_graph, format="turtle")
            if len(reloaded) != len(result.graph) or set(reloaded) != set(result.graph):
                raise RuntimeError(
                    "Serialized ontology graph failed exact RDF round-trip"
                )
            pending_report.write_text(
                json.dumps(result.report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if _source_hashes(corpus_path, concepts_path, vocabulary_path) != {
                key: result.report["inputs"][key]
                for key in ("corpus_sha256", "concepts_sha256", "vocabulary_sha256")
            }:
                raise ValueError("An input changed before final publication")
            os.replace(pending_graph, output_path)
            os.replace(pending_report, report_path)
    return result.report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the source-backed IPCC AR6 Figure--Concept ontology graph."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--include-captions",
        action="store_true",
        help="Include full IPCC captions in RDF (check IPCC reuse permissions first).",
    )
    parser.add_argument(
        "--validate-shacl",
        action="store_true",
        help="Run formal SHACL validation using installed pyshacl (fails if unavailable).",
    )
    args = parser.parse_args()
    report = write_ontology(
        args.corpus,
        args.concepts,
        args.vocabulary,
        args.output,
        args.report,
        include_captions=args.include_captions,
        validate_shacl=args.validate_shacl,
    )
    summary = report["summary"]
    print(
        f"Knowledge graph: {summary['figures']} figures; "
        f"{summary['concepts']} glossary concepts; "
        f"{summary['figure_concept_links']} caption-supported links; "
        f"{summary['rdf_triples']} RDF triples."
    )
    print(
        f"Integrity: {report['validation']['rdf_integrity']}; "
        f"SHACL: {report['validation']['shacl']}"
    )
    print(f"Turtle: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

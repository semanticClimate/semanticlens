"""Focused, source-faithful Figure--ScientificConcept RDF subgraphs.

Only accepted ``ipcc:mentionsConcept`` edges are traversed. The initial RDF
subset contains at most ``limit`` figures and ``related_concepts`` concepts;
other real source-graph neighbors can be revealed on demand. This module does
not create caption-evidence or image-reference nodes: source evidence remains
in concepts.db and figure images are literal properties / corpus.db records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Iterable, Literal

from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, RDF, RDFS, SKOS

from ontology.schema import IPCC, resource_iri
from .query import ConceptSummary, FigureHit, MatchMode

if TYPE_CHECKING:
    from .query import IPCCGraph


_MAX_FIGURES = 100
_MAX_EXPANSION_DEPTH = 3

# There is exactly one instance-to-instance RDF relationship in this ontology.
_RELATIONS = frozenset({IPCC.mentionsConcept})

# Source-backed literals stay with their Figure or ScientificConcept; they are
# not separate nodes. Do not copy full captions into an exported subgraph.
_METADATA = {
    IPCC.Figure: (
        DCTERMS.identifier, DCTERMS.source, IPCC.workingGroupCode,
        IPCC.sectionCode, IPCC.captionStatus, IPCC.captionSHA256, IPCC.imageURL,
    ),
    IPCC.ScientificConcept: (
        DCTERMS.identifier, DCTERMS.source, SKOS.prefLabel, SKOS.altLabel,
    ),
}

_PREDICATE_METADATA = (RDF.type, RDFS.label, RDFS.comment, RDFS.domain, RDFS.range)
_CLASS_METADATA = (RDF.type, RDFS.label, RDFS.comment, RDFS.subClassOf)


def _instance_type(source: Graph, node: URIRef) -> URIRef | None:
    """Identify only the two allowed RDF instance classes."""
    for kind in _METADATA:
        if (node, RDF.type, kind) in source:
            return kind
    return None


def _node_iri(node: URIRef | str) -> URIRef:
    if isinstance(node, URIRef):
        value = str(node)
    elif isinstance(node, str):
        value = node
    else:
        raise TypeError("node must be an RDF URIRef or an absolute resource IRI")
    if not value or value != value.strip() or not value.startswith("https://"):
        raise ValueError("node must be an absolute HTTPS resource IRI")
    return URIRef(value)


def _predicate_set(predicates: Iterable[URIRef] | None) -> frozenset[URIRef]:
    if predicates is None:
        return _RELATIONS
    selected = frozenset(predicates)
    if not all(isinstance(p, URIRef) and p in _RELATIONS for p in selected):
        raise ValueError("predicates must contain only ipcc:mentionsConcept")
    return selected


def _edges_for(
    source: Graph, node: URIRef, predicates: frozenset[URIRef],
) -> tuple[tuple[URIRef, URIRef, URIRef], ...]:
    """Return actual directed source triples incident on a figure or concept."""
    if IPCC.mentionsConcept not in predicates:
        return ()
    kind = _instance_type(source, node)
    if kind == IPCC.Figure:
        edges = (
            (node, IPCC.mentionsConcept, concept)
            for concept in source.objects(node, IPCC.mentionsConcept)
            if isinstance(concept, URIRef)
            and _instance_type(source, concept) == IPCC.ScientificConcept
        )
    elif kind == IPCC.ScientificConcept:
        edges = (
            (figure, IPCC.mentionsConcept, node)
            for figure in source.subjects(IPCC.mentionsConcept, node)
            if isinstance(figure, URIRef)
            and _instance_type(source, figure) == IPCC.Figure
        )
    else:
        return ()
    return tuple(sorted(edges, key=lambda triple: (str(triple[0]), str(triple[2]))))


@dataclass(frozen=True, slots=True)
class KnowledgeSubgraph:
    """Initial Figure/Concept selection plus access to real source neighbors.

    ``figures``, ``concepts`` and ``graph`` represent the initially requested
    view (or an explicit Python ``expand`` result). ``_source`` retains the
    complete, unchanged RDF graph for expansion; ``_kg`` provides the existing
    read-only SQLite-backed figure/caption evidence to the HTML renderer.
    """

    query: str
    match: Literal["concept", "caption"]
    total_figures: int
    figures: tuple[FigureHit, ...]
    concepts: tuple[ConceptSummary, ...]
    graph: Graph
    _source: Graph | None = field(default=None, repr=False, compare=False, kw_only=True)
    _wg_filter: str | None = field(default=None, repr=False, compare=False, kw_only=True)
    _kg: IPCCGraph | None = field(default=None, repr=False, compare=False, kw_only=True)

    @property
    def figure_ids(self) -> tuple[str, ...]:
        return tuple(hit.figure_id for hit in self.figures)

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(concept.term_id for concept in self.concepts)

    def __len__(self) -> int:
        return len(self.figures)

    def neighbors(
        self, node: URIRef | str, *, predicates: Iterable[URIRef] | None = None,
    ) -> tuple[tuple[URIRef, URIRef, URIRef], ...]:
        """List all real incident Figure--Concept links, even if not yet shown."""
        if self._source is None:
            raise RuntimeError("Source graph unavailable for neighborhood exploration")
        uri = _node_iri(node)
        if _instance_type(self._source, uri) is None:
            raise KeyError(f"Unknown IPCC figure/concept resource: {uri}")
        if not _has_resource(self.graph, uri):
            raise ValueError("Start node is not part of this view")
        return _edges_for(self._source, uri, _predicate_set(predicates))

    def expand(
        self, node: URIRef | str, *, depth: int = 1, max_nodes: int | None = None,
        predicates: Iterable[URIRef] | None = None,
    ) -> KnowledgeSubgraph:
        """Return a new view with real linked nodes revealed in both directions.

        One hop reveals *all* currently missing direct neighbors by default.
        Set ``max_nodes`` to a positive integer to bound *new* resources.
        A ``wg`` filter, if specified by the original search, continues to
        apply to newly revealed figures. Search ranking/results remain intact.
        """
        if type(depth) is not int or not 1 <= depth <= _MAX_EXPANSION_DEPTH:
            raise ValueError(f"depth must be between 1 and {_MAX_EXPANSION_DEPTH}")
        if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 1):
            raise ValueError("max_nodes must be a positive integer or None")

        uri = _node_iri(node)
        relations = _predicate_set(predicates)
        self.neighbors(uri, predicates=relations)  # Verify selected node.
        if self._source is None:  # Defensive: checked by neighbors above.
            raise RuntimeError("Source graph unavailable for neighborhood exploration")

        subset = _RDFSubset(self._source, initial=self.graph)
        present = _instance_resources(self._source, subset.graph)
        frontier = (uri,)
        added = 0
        for _ in range(depth):
            following: set[URIRef] = set()
            for center in frontier:
                for figure, predicate, concept in _edges_for(self._source, center, relations):
                    newly_visible = tuple(n for n in (figure, concept) if n not in present)
                    if (self._wg_filter and figure in newly_visible
                            and str(self._source.value(figure, IPCC.workingGroupCode)) != self._wg_filter):
                        continue
                    if max_nodes is not None and added + len(newly_visible) > max_nodes:
                        continue
                    subset.graph.add((figure, predicate, concept))
                    for target in newly_visible:
                        present.add(target)
                        subset.describe(target)
                        following.add(target)
                        added += 1
            if not following:
                break
            frontier = tuple(sorted(following, key=str))

        subset.schema_descriptions()
        return replace(self, graph=subset.graph)

    def show(self, *, height: int = 650) -> None:
        """Display this RDF view in Colab or Jupyter."""
        from .visualize import show_subgraph
        show_subgraph(self, height=height)

    def save(self, path: str | Path) -> Path:
        """Export HTML or RDF (including JSON-LD) without changing the source."""
        output = Path(path).expanduser()
        if output.suffix.lower() in (".html", ".htm"):
            from .visualize import save_html
            return save_html(self, output)
        rdf_format = {
            ".ttl": "turtle", ".nt": "nt", ".rdf": "xml",
            ".xml": "xml", ".jsonld": "json-ld", ".json": "json-ld",
        }.get(output.suffix.lower())
        if rdf_format is None:
            raise ValueError(
                "Unsupported RDF format; use .ttl, .nt, .rdf, .xml, .jsonld or .json"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", delete=False,
        ) as stage:
            temporary = Path(stage.name)
        try:
            self.graph.serialize(destination=temporary, format=rdf_format, encoding="utf-8")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output


def _has_resource(graph: Graph, node: URIRef) -> bool:
    return next(graph.triples((node, None, None)), None) is not None or next(
        graph.triples((None, None, node)), None
    ) is not None


def _instance_resources(source: Graph, graph: Graph) -> set[URIRef]:
    candidates = set(graph.subjects()) | {
        obj for obj in graph.objects() if isinstance(obj, URIRef)
    }
    return {node for node in candidates if isinstance(node, URIRef)
            and _instance_type(source, node) is not None}


class _RDFSubset:
    """Copy real source triples and safe, minimal instance descriptions."""

    def __init__(self, source: Graph, *, initial: Graph | None = None) -> None:
        self.source = source
        self.graph = Graph()
        for prefix, namespace in source.namespaces():
            self.graph.bind(prefix, namespace)
        if initial is not None:
            self.graph += initial

    def describe(self, uri: URIRef) -> None:
        kind = _instance_type(self.source, uri)
        if kind is None:
            raise ValueError(f"Not a Figure or ScientificConcept instance: {uri}")
        self.graph.add((uri, RDF.type, kind))
        for predicate in _METADATA[kind]:
            for value in self.source.objects(uri, predicate):
                self.graph.add((uri, predicate, value))

    def connect(self, subject: URIRef, predicate: URIRef, obj: URIRef) -> None:
        if (subject, predicate, obj) not in self.source:
            raise ValueError(f"Relationship absent from the source RDF: {predicate}")
        if predicate != IPCC.mentionsConcept:
            raise ValueError(f"Unsupported RDF relationship: {predicate}")
        if (_instance_type(self.source, subject) != IPCC.Figure
                or _instance_type(self.source, obj) != IPCC.ScientificConcept):
            raise ValueError("Only Figure--mentionsConcept-->ScientificConcept edges allowed")
        self.graph.add((subject, predicate, obj))
        self.describe(subject)
        self.describe(obj)

    def initial_figures(self, figure_ids: tuple[str, ...]) -> None:
        for figure_id in figure_ids:
            self.describe(resource_iri("figure", figure_id))

    def initial_concepts(self, concept_ids: tuple[str, ...]) -> None:
        for concept_id in concept_ids:
            self.describe(resource_iri("concept", concept_id))

    def connect_selected(
        self, figure_ids: tuple[str, ...], concept_ids: tuple[str, ...],
    ) -> None:
        """Include only accepted edges between the *selected* endpoints."""
        concepts = {resource_iri("concept", cid) for cid in concept_ids}
        for figure_id in figure_ids:
            fig = resource_iri("figure", figure_id)
            for concept in self.source.objects(fig, IPCC.mentionsConcept):
                if concept in concepts:
                    self.connect(fig, IPCC.mentionsConcept, concept)

    def schema_descriptions(self) -> None:
        """Include descriptions of the two classes and only edge predicate."""
        for klass in (IPCC.Figure, IPCC.ScientificConcept):
            for predicate in _CLASS_METADATA:
                for value in self.source.objects(klass, predicate):
                    self.graph.add((klass, predicate, value))
        for predicate in _PREDICATE_METADATA:
            for value in self.source.objects(IPCC.mentionsConcept, predicate):
                self.graph.add((IPCC.mentionsConcept, predicate, value))


def build_subgraph(
    kg: IPCCGraph,
    query: str,
    *,
    wg: str | None = None,
    limit: int = 20,
    related_concepts: int = 5,
    match: MatchMode = "auto",
) -> KnowledgeSubgraph:
    """Select a bounded Figure/Concept view; leave all other nodes hidden.

    ``related_concepts`` is a TOTAL count, including the seed concept when
    a glossary term was resolved. Extra source-backed Figure/Concept neighbors
    remain accessible through ``view.neighbors`` and ``view.expand``.
    """
    if type(limit) is not int or not 1 <= limit <= _MAX_FIGURES:
        raise ValueError(f"limit must be an integer between 1 and {_MAX_FIGURES}")
    if type(related_concepts) is not int or related_concepts < 0:
        raise ValueError("related_concepts must be a non-negative integer")

    results = kg.search(query, wg=wg, limit=limit, match=match)
    figure_ids = tuple(hit.figure_id for hit in results.items)
    seed = results.concept_id

    coverage = Counter(
        term_id
        for figure_id in figure_ids
        for term_id in kg._by_figure.get(figure_id, ())
        if term_id != seed
    )
    ranked = sorted(
        coverage,
        key=lambda term_id: (
            -coverage[term_id], kg._concept_labels[term_id].casefold(), term_id
        ),
    )
    selected = ((seed,) if seed and related_concepts else ()) + tuple(
        ranked[:max(0, related_concepts - (1 if seed else 0))]
    )

    subset = _RDFSubset(kg._graph)
    subset.initial_figures(figure_ids)
    subset.initial_concepts(selected)
    subset.connect_selected(figure_ids, selected)
    subset.schema_descriptions()

    return KnowledgeSubgraph(
        query=results.query,
        match=results.match,
        total_figures=results.total,
        figures=results.items,
        concepts=tuple(
            ConceptSummary(term_id, kg._concept_labels[term_id]) for term_id in selected
        ),
        graph=subset.graph,
        _source=kg._graph,
        _wg_filter=wg.strip().upper() if wg is not None else None,
        _kg=kg,
    )

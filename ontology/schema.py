from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.parse import quote

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SH, SKOS, XSD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_IRI = "https://neelkumar01.github.io/IPCC-visual-knowledge-graph/"
ONTOLOGY_IRI = URIRef(BASE_IRI + "ontology")
IPCC = Namespace(BASE_IRI + "ontology#")
RESOURCE = Namespace(BASE_IRI + "resource/")

DEFAULT_ONTOLOGY = PROJECT_ROOT / "ontology" / "ipcc_visual_ontology.ttl"
DEFAULT_SHAPES = PROJECT_ROOT / "ontology" / "ipcc_visual_shapes.ttl"

# Only these two classes may be instantiated in the knowledge graph.
_CLASSES = (
    ("Figure", "Figure", "A figure published on an IPCC AR6 figure page.", None),
    ("ScientificConcept", "Scientific concept", "A glossary-derived scientific term linked to figures through accepted caption matches.", None),
)

# A link records an accepted caption mention, not image depiction or causality.
_OBJECT_PROPERTIES = (
    ("mentionsConcept", "mentions concept", "The figure caption contains an accepted match for this scientific concept; the figure need not visually depict it.", IPCC.Figure, IPCC.ScientificConcept),
)

# These are Figure attributes, not nodes or object relationships.
# Image URLs can occur more than once per Figure; their original order is in
# corpus.db / figure_images, which the viewer also uses for ordered previews.
_DATA_PROPERTIES = (
    ("workingGroupCode", "working group code", "IPCC working group publishing the figure: WG1, WG2 or WG3.", IPCC.Figure, XSD.string),
    ("sectionCode", "section code", "Original chapter or section identifier from the figure corpus.", IPCC.Figure, XSD.string),
    ("captionText", "caption text", "Optional original caption; only include when permitted by IPCC reuse terms.", IPCC.Figure, XSD.string),
    ("captionStatus", "caption status", "Whether the original figure caption is available or missing.", IPCC.Figure, XSD.string),
    ("captionSHA256", "caption SHA-256", "Digest of the exact original caption where available.", IPCC.Figure, XSD.string),
    ("imageURL", "image URL", "A published candidate image URL for the figure, without a reachability guarantee.", IPCC.Figure, XSD.anyURI),
)


def _prefixes(graph: Graph) -> Graph:
    for prefix, namespace in (
        ("ipcc", IPCC), ("resource", RESOURCE), ("rdf", RDF), ("rdfs", RDFS),
        ("owl", OWL), ("skos", SKOS), ("dcterms", DCTERMS),
        ("sh", SH), ("xsd", XSD),
    ):
        graph.bind(prefix, namespace)
    return graph


def resource_iri(kind: str, identifier: str) -> URIRef:
    """Mint stable URL-safe IRIs for figures and scientific concepts only."""
    if kind not in {"figure", "concept"}:
        raise ValueError(f"Unknown resource kind: {kind!r}")
    if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
        raise ValueError("Resource identifier must be a nonempty, unpadded string")
    return URIRef(f"{RESOURCE}{kind}/{quote(identifier, safe='-._~')}")


def build_ontology_schema() -> Graph:
    """Return the bipartite graph's OWL classes and properties, without instances."""
    graph = _prefixes(Graph())
    graph.add((ONTOLOGY_IRI, RDF.type, OWL.Ontology))
    graph.add((ONTOLOGY_IRI, DCTERMS.title, Literal("IPCC AR6 Visual Knowledge Graph Ontology", lang="en")))
    graph.add((ONTOLOGY_IRI, DCTERMS.description, Literal(
        "Bipartite graph connecting IPCC AR6 figures and scientific concepts through "
        "source-backed caption matches, without claiming visual depiction or causality.", lang="en")))
    graph.add((ONTOLOGY_IRI, OWL.versionInfo, Literal("0.2.0")))
    graph.add((ONTOLOGY_IRI, RDFS.seeAlso, URIRef("https://www.ipcc.ch/assessment-report/ar6/")))

    for name, label, description, parent in _CLASSES:
        subject = IPCC[name]
        graph.add((subject, RDF.type, OWL.Class))
        graph.add((subject, RDFS.label, Literal(label, lang="en")))
        graph.add((subject, RDFS.comment, Literal(description, lang="en")))
        if parent is not None:
            graph.add((subject, RDFS.subClassOf, parent))

    for name, label, description, domain, range_ in _OBJECT_PROPERTIES:
        subject = IPCC[name]
        graph.add((subject, RDF.type, OWL.ObjectProperty))
        graph.add((subject, RDFS.label, Literal(label, lang="en")))
        graph.add((subject, RDFS.comment, Literal(description, lang="en")))
        graph.add((subject, RDFS.domain, domain))
        graph.add((subject, RDFS.range, range_))

    for name, label, description, domain, datatype in _DATA_PROPERTIES:
        subject = IPCC[name]
        graph.add((subject, RDF.type, OWL.DatatypeProperty))
        graph.add((subject, RDFS.label, Literal(label, lang="en")))
        graph.add((subject, RDFS.comment, Literal(description, lang="en")))
        graph.add((subject, RDFS.domain, domain))
        graph.add((subject, RDFS.range, datatype))

    return graph


def _stable_node(*parts: object) -> BNode:
    """Use stable blank-node IDs so regenerated Turtle avoids arbitrary churn."""
    seed = "|".join(str(part) for part in parts)
    return BNode("n" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24])


def _rdf_list(graph: Graph, values: tuple[URIRef | Literal | BNode, ...]) -> BNode:
    """Build a deterministic RDF collection (without rdflib's random IDs)."""
    if not values:
        raise ValueError("SHACL enumeration must not be empty")
    seed = "|".join(str(v) for v in values)
    cells = [_stable_node("list", seed, index) for index in range(len(values))]
    for index, value in enumerate(values):
        graph.add((cells[index], RDF.first, value))
        graph.add((cells[index], RDF.rest, cells[index + 1] if index + 1 < len(values) else RDF.nil))
    return cells[0]


def _shape_property(
    graph: Graph, shape: URIRef | BNode, path: URIRef, *,
    required: bool = False, single: bool = False,
    datatype: URIRef | None = None, value_class: URIRef | None = None,
    iri: bool = False, choices: tuple[str, ...] = (), pattern: str | None = None,
) -> None:
    prop = _stable_node("property", shape, path)
    graph.add((shape, SH.property, prop))
    graph.add((prop, SH.path, path))
    if required:
        graph.add((prop, SH.minCount, Literal(1)))
    if single:
        graph.add((prop, SH.maxCount, Literal(1)))
    if datatype is not None:
        graph.add((prop, SH.datatype, datatype))
    if value_class is not None:
        graph.add((prop, SH["class"], value_class))
        graph.add((prop, SH.nodeKind, SH.IRI))
    if iri:
        graph.add((prop, SH.nodeKind, SH.IRI))
    if choices:
        graph.add((prop, SH['in'], _rdf_list(graph, tuple(Literal(x) for x in choices))))
    if pattern is not None:
        graph.add((prop, SH.pattern, Literal(pattern)))


def build_validation_shapes() -> Graph:
    """Validate the two entity classes and their literal attributes."""
    graph = _prefixes(Graph())
    for name, klass in (
        ("FigureShape", IPCC.Figure),
        ("ScientificConceptShape", IPCC.ScientificConcept),
    ):
        shape = IPCC[name]
        graph.add((shape, RDF.type, SH.NodeShape))
        graph.add((shape, SH.targetClass, klass))

    fig = IPCC.FigureShape
    _shape_property(graph, fig, DCTERMS.identifier, required=True, single=True, datatype=XSD.string)
    _shape_property(graph, fig, DCTERMS.source, required=True, single=True, iri=True)
    _shape_property(graph, fig, IPCC.workingGroupCode, required=True, single=True, datatype=XSD.string, choices=("WG1", "WG2", "WG3"))
    _shape_property(graph, fig, IPCC.sectionCode, required=True, single=True, datatype=XSD.string)
    _shape_property(graph, fig, IPCC.captionStatus, required=True, single=True, datatype=XSD.string, choices=("available", "missing"))
    _shape_property(graph, fig, IPCC.captionText, single=True, datatype=XSD.string)
    _shape_property(graph, fig, IPCC.captionSHA256, single=True, datatype=XSD.string, pattern="^[0-9a-f]{64}$")
    _shape_property(graph, fig, IPCC.imageURL, datatype=XSD.anyURI)
    _shape_property(graph, fig, IPCC.mentionsConcept, value_class=IPCC.ScientificConcept)

    concept = IPCC.ScientificConceptShape
    _shape_property(graph, concept, DCTERMS.identifier, required=True, single=True, datatype=XSD.string)
    _shape_property(graph, concept, SKOS.prefLabel, required=True)
    _shape_property(graph, concept, DCTERMS.source, required=True, single=True, iri=True)
    return graph


def write_schema(
    ontology_path: Path = DEFAULT_ONTOLOGY,
    shapes_path: Path = DEFAULT_SHAPES,
) -> tuple[Path, Path]:
    """Serialize independent, deterministic Turtle files; never export records."""
    ontology_path, shapes_path = Path(ontology_path), Path(shapes_path)
    if ontology_path.resolve() == shapes_path.resolve():
        raise ValueError("Ontology and shapes outputs must be different paths")
    for path in (ontology_path, shapes_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    build_ontology_schema().serialize(destination=ontology_path, format="turtle")
    build_validation_shapes().serialize(destination=shapes_path, format="turtle")
    return ontology_path, shapes_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the IPCC figure ontology and SHACL shapes")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    args = parser.parse_args()
    ontology, shapes = write_schema(args.ontology, args.shapes)
    print(f"Ontology: {ontology}")
    print(f"SHACL: {shapes}")


if __name__ == "__main__":
    main()

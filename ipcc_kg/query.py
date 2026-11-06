"""
Provides access to the IPCC visual knowledge graph and its figure data

Supports concept and caption searches, figure retrieval,
and exploration of related concepts
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing, contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator, Literal

from rdflib import Graph
from rdflib.namespace import DCTERMS, RDF

from concepts.vocabulary import normalize_term
from ontology.schema import IPCC

MatchMode = Literal["auto", "concept", "caption"]
_WORKING_GROUPS = frozenset({"WG1", "WG2", "WG3"})


@dataclass(frozen=True, slots=True)
class CaptionMatch:
    """
    Store a caption match with its text position and concept details
    """

    start_offset: int
    end_offset: int
    text: str
    concept_id: str | None = None
    resolution: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionMention:
    """
    Store a caption mention with its matching details and candidate concepts
    """

    mention_id: int
    start_offset: int
    end_offset: int
    text: str
    normalized_text: str
    match_kind: str
    resolution: str
    concept_id: str | None
    candidate_concept_ids: tuple[str, ...]
    quality_rule: str | None


@dataclass(frozen=True, slots=True)
class ConceptSummary:
    """
    Store the ID and label of a concept
    """

    term_id: str
    label: str


@dataclass(frozen=True, slots=True)
class Concept:
    """
    Store a concept with its definition, aliases, sources, and figure count
    """

    term_id: str
    label: str
    definition: str | None
    aliases: tuple[str, ...]
    glossary_working_groups: tuple[str, ...]
    source_series: tuple[str, ...]
    figure_count: int


@dataclass(frozen=True, slots=True)
class Figure:
    """
    Store a figure with its metadata, images, concepts, and caption mentions
    """

    figure_id: str
    label: str | None
    title: str | None
    working_group: str
    chapter: str | None
    section: str
    caption: str | None
    caption_status: str
    image_urls: tuple[str, ...]
    source_url: str
    copyright_policy_url: str
    concepts: tuple[ConceptSummary, ...]
    mentions: tuple[CaptionMention, ...]

    @property
    def image_url(self) -> str | None:
        """
        Return the first image URL or None if no images are available
        """
        return self.image_urls[0] if self.image_urls else None


@dataclass(frozen=True, slots=True)
class FigureHit:
    """
    Store a matching figure with its source details and caption matches
    """

    figure_id: str
    label: str | None
    title: str | None
    working_group: str
    section: str
    source_url: str
    image_url: str | None
    matches: tuple[CaptionMatch, ...]

    @property
    def match_count(self) -> int:
        """
        Return the number of caption matches found in the figure
        """
        return len(self.matches)


@dataclass(frozen=True, slots=True)
class SearchResults:
    """
    Store figure search results with the query, match type, and total count
    """

    query: str
    match: Literal["concept", "caption"]
    total: int
    items: tuple[FigureHit, ...]
    concept_id: str | None = None

    def __len__(self) -> int:
        """
        Return the number of figures included in the results
        """
        return len(self.items)

    def __iter__(self) -> Iterator[FigureHit]:
        """
        Iterate over the matching figures
        """
        return iter(self.items)

    def __getitem__(self, index: int | slice) -> FigureHit | tuple[FigureHit, ...]:
        """
        Return a figure or slice of figures from the results
        """
        return self.items[index]

    def __repr__(self) -> str:
        """
        Return a short description of the search results
        """
        return (
            f"SearchResults(query={self.query!r}, match={self.match!r}, "
            f"total={self.total}, shown={len(self.items)})"
        )

    def to_dataframe(self):
        """
        Convert the search results into a pandas DataFrame
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() requires pandas: pip install pandas"
            ) from exc
        return pd.DataFrame.from_records(
            (
                {
                    "figure_id": hit.figure_id,
                    "figure_label": hit.label,
                    "title": hit.title,
                    "working_group": hit.working_group,
                    "section": hit.section,
                    "match_count": hit.match_count,
                    "image_url": hit.image_url,
                    "source_url": hit.source_url,
                }
                for hit in self.items
            ),
            columns=(
                "figure_id",
                "figure_label",
                "title",
                "working_group",
                "section",
                "match_count",
                "image_url",
                "source_url",
            ),
        )


def _read_only_connection(path: Path) -> sqlite3.Connection:
    """
    Open a SQLite database connection in read-only mode
    """
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only = ON")
    return db


@contextmanager
def _database(path: Path) -> Iterator[sqlite3.Connection]:
    """
    Provide a read-only database connection and close it after use
    """
    with closing(_read_only_connection(path)) as db:
        yield db


def _sha256(path: Path) -> str:
    """
    Calculate the SHA-256 checksum of a file
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _working_group(value: str | None) -> str | None:
    """
    Validate and standardize a working-group code
    """
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().upper() not in _WORKING_GROUPS:
        raise ValueError("wg must be one of WG1, WG2, WG3, or None")
    return value.strip().upper()


def _nonempty(value: str, name: str) -> str:
    """
    Validate and return a nonempty string without surrounding spaces
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


class IPCCGraph:
    """
    Provide search and retrieval access to the IPCC visual knowledge graph
    """

    def __init__(
        self,
        *,
        graph_path: str | Path,
        corpus_path: str | Path,
        concepts_path: str | Path,
    ) -> None:
        self.graph_path = Path(graph_path).expanduser().resolve()
        self.corpus_path = Path(corpus_path).expanduser().resolve()
        self.concepts_path = Path(concepts_path).expanduser().resolve()
        for path in (self.graph_path, self.corpus_path, self.concepts_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required published artifact not found: {path}"
                )

        self._graph = Graph()
        self._graph.parse(self.graph_path, format="turtle")
        self._concept_labels: dict[str, str] = {}
        aliases: dict[str, set[str]] = defaultdict(set)

        with _database(self.concepts_path) as db:
            for term_id, label in db.execute(
                "SELECT term_id, canonical_label FROM concepts"
            ):
                self._concept_labels[term_id] = label
                aliases[normalize_term(label)].add(term_id)
            for term_id, alias in db.execute(
                "SELECT term_id, alias FROM concept_aliases"
            ):
                aliases[normalize_term(alias)].add(term_id)
            corpus_digest = db.execute(
                "SELECT value FROM metadata WHERE key = 'corpus_sha256'"
            ).fetchone()
            if corpus_digest and corpus_digest[0] != _sha256(self.corpus_path):
                raise ValueError("concepts.db was built from a different corpus.db")
            stored_pairs = {
                (row[0], row[1])
                for row in db.execute("SELECT figure_id, term_id FROM figure_concepts")
            }

        self._lookup = {key: tuple(sorted(ids)) for key, ids in aliases.items()}
        self._by_concept: dict[str, frozenset[str]] = {}
        self._by_figure: dict[str, frozenset[str]] = {}
        self._index_graph(stored_pairs)

    @classmethod
    def from_repo(cls, root: str | Path | None = None) -> IPCCGraph:
        """
        Load the knowledge graph from the repository's published files
        """
        repo = (
            Path(root).expanduser().resolve()
            if root is not None
            else Path(__file__).resolve().parents[1]
        )
        corpus = repo / "data/processed/corpus.db"
        concepts = repo / "data/processed/concepts.db"
        graph = repo / "ontology/ipcc_visual_knowledge_graph.ttl"
        report = repo / "reports/ontology_quality.json"

        if report.is_file() and corpus.is_file() and concepts.is_file():
            expected = json.loads(report.read_text(encoding="utf-8")).get("inputs", {})
            for key, file in (("corpus_sha256", corpus), ("concepts_sha256", concepts)):
                if key in expected and _sha256(file) != expected[key]:
                    raise ValueError(
                        f"{file.name} differs from the published ontology build; "
                        "use matching corpus, concepts and RDF artifacts."
                    )
        return cls(graph_path=graph, corpus_path=corpus, concepts_path=concepts)

    def _index_graph(self, stored_pairs: set[tuple[str, str]]) -> None:
        """
        Check graph relationships against the databases and index concept links
        """
        with _database(self.corpus_path) as db:
            known_figures = {
                row[0] for row in db.execute("SELECT figure_id FROM figures")
            }

        graph_figures = {
            str(self._graph.value(node, DCTERMS.identifier))
            for node in self._graph.subjects(RDF.type, IPCC.Figure)
        }
        graph_concepts = {
            str(self._graph.value(node, DCTERMS.identifier))
            for node in self._graph.subjects(RDF.type, IPCC.ScientificConcept)
        }
        if graph_figures != known_figures or graph_concepts != set(
            self._concept_labels
        ):
            raise ValueError(
                "RDF figures/concepts do not match the supplied SQLite databases"
            )

        by_concept: dict[str, set[str]] = defaultdict(set)
        by_figure: dict[str, set[str]] = defaultdict(set)
        graph_pairs: set[tuple[str, str]] = set()
        for figure_node, concept_node in self._graph.subject_objects(
            IPCC.mentionsConcept
        ):
            figure_id = str(self._graph.value(figure_node, DCTERMS.identifier))
            term_id = str(self._graph.value(concept_node, DCTERMS.identifier))
            graph_pairs.add((figure_id, term_id))
            by_concept[term_id].add(figure_id)
            by_figure[figure_id].add(term_id)
        if graph_pairs != stored_pairs:
            raise ValueError(
                "Accepted RDF figure–concept links differ from concepts.db"
            )
        self._by_concept = {key: frozenset(value) for key, value in by_concept.items()}
        self._by_figure = {key: frozenset(value) for key, value in by_figure.items()}

    def _resolve_concept(self, query: str) -> str | None:
        """
        Find a concept ID from its label, alias, or ID and reject ambiguous matches
        """
        if query in self._concept_labels:
            return query
        matches = self._lookup.get(normalize_term(query), ())
        if len(matches) > 1:
            options = ", ".join(
                f"{self._concept_labels[term_id]} ({term_id})" for term_id in matches
            )
            raise ValueError(
                f"Ambiguous glossary term {query!r}: {options}. "
                "Use a full concept label or term ID."
            )
        return matches[0] if matches else None

    def search(
        self,
        query: str,
        *,
        wg: str | None = None,
        limit: int | None = 20,
        match: MatchMode = "auto",
    ) -> SearchResults:
        """
        Search figures by concept or caption text with optional working-group filtering
        """
        query = _nonempty(query, "query")
        wg = _working_group(wg)
        if match not in ("auto", "concept", "caption"):
            raise ValueError("match must be 'auto', 'concept', or 'caption'")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer or None")

        term_id = self._resolve_concept(query) if match != "caption" else None
        selected_mode: Literal["concept", "caption"] = (
            "concept" if term_id or match == "concept" else "caption"
        )
        hits: list[FigureHit] = []
        with _database(self.corpus_path) as corpus:
            rows = corpus.execute(
                "SELECT figure_id, figure_label, title, working_group, section, "
                "source_url, image_url, caption FROM figures"
                + (" WHERE working_group = ?" if wg else ""),
                (wg,) if wg else (),
            )
            if selected_mode == "concept":
                valid_figures = self._by_concept.get(term_id or "", frozenset())
                with _database(self.concepts_path) as concepts:
                    evidence: dict[str, list[CaptionMatch]] = defaultdict(list)
                    if term_id:
                        for row in concepts.execute(
                            "SELECT figure_id, start_offset, end_offset, matched_text, resolution "
                            "FROM mentions WHERE term_id = ? ORDER BY start_offset, end_offset",
                            (term_id,),
                        ):
                            evidence[row[0]].append(
                                CaptionMatch(row[1], row[2], row[3], term_id, row[4])
                            )
                for row in rows:
                    if row["figure_id"] in valid_figures:
                        hits.append(self._hit(row, tuple(evidence[row["figure_id"]])))
            else:
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                for row in rows:
                    if not row["caption"]:
                        continue
                    found = tuple(
                        CaptionMatch(m.start(), m.end(), m.group())
                        for m in pattern.finditer(row["caption"])
                    )
                    if found:
                        hits.append(self._hit(row, found))

        hits.sort(key=lambda h: (-h.match_count, h.working_group, h.figure_id))
        return SearchResults(
            query,
            selected_mode,
            len(hits),
            tuple(hits if limit is None else hits[:limit]),
            term_id,
        )

    @staticmethod
    def _hit(row: sqlite3.Row, matches: tuple[CaptionMatch, ...]) -> FigureHit:
        """
        Convert a database row and its caption matches into a figure search result
        """
        return FigureHit(
            row["figure_id"],
            row["figure_label"],
            row["title"],
            row["working_group"],
            row["section"],
            row["source_url"],
            row["image_url"],
            matches,
        )

    def concept(self, query: str) -> Concept:
        """
        Retrieve a concept with its glossary details and linked figure count
        """
        query = _nonempty(query, "query")
        term_id = self._resolve_concept(query)
        if term_id is None:
            raise KeyError(f"No glossary concept matches {query!r}")
        with _database(self.concepts_path) as db:
            row = db.execute(
                "SELECT canonical_label, definition FROM concepts WHERE term_id = ?",
                (term_id,),
            ).fetchone()
            aliases = tuple(
                r[0]
                for r in db.execute(
                    "SELECT alias FROM concept_aliases WHERE term_id = ? ORDER BY alias",
                    (term_id,),
                )
            )
            groups = tuple(
                r[0]
                for r in db.execute(
                    "SELECT working_group FROM concept_working_groups "
                    "WHERE term_id = ? ORDER BY working_group",
                    (term_id,),
                )
            )
            series = tuple(
                r[0]
                for r in db.execute(
                    "SELECT series FROM concept_source_series "
                    "WHERE term_id = ? ORDER BY series",
                    (term_id,),
                )
            )
        return Concept(
            term_id,
            row["canonical_label"],
            row["definition"],
            aliases,
            groups,
            series,
            len(self._by_concept.get(term_id, ())),
        )

    def explore(
        self,
        query: str,
        *,
        wg: str | None = None,
        limit: int = 20,
        related_concepts: int = 5,
        match: MatchMode = "auto",
    ) -> "KnowledgeSubgraph":
        """
        Build a knowledge subgraph around a search query and related concepts
        """
        from .subgraph import build_subgraph

        return build_subgraph(
            self,
            query,
            wg=wg,
            limit=limit,
            related_concepts=related_concepts,
            match=match,
        )

    def figure(self, figure_id: str) -> Figure:
        """
        Retrieve a figure with its metadata, images, concepts, and caption mentions
        """
        figure_id = _nonempty(figure_id, "figure_id")
        with _database(self.corpus_path) as db:
            row = db.execute(
                "SELECT * FROM figures WHERE figure_id = ?", (figure_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown figure ID: {figure_id!r}")
            images = tuple(
                r[0]
                for r in db.execute(
                    "SELECT image_url FROM figure_images WHERE figure_id = ? "
                    "ORDER BY position",
                    (figure_id,),
                )
            )
        with _database(self.concepts_path) as db:
            candidates: dict[int, list[str]] = defaultdict(list)
            for mention_id, candidate_id in db.execute(
                "SELECT mc.mention_id, mc.term_id FROM mention_candidates mc "
                "JOIN mentions m ON m.mention_id = mc.mention_id "
                "WHERE m.figure_id = ? ORDER BY mc.mention_id, mc.term_id",
                (figure_id,),
            ):
                candidates[mention_id].append(candidate_id)
            mentions = tuple(
                CaptionMention(
                    m["mention_id"],
                    m["start_offset"],
                    m["end_offset"],
                    m["matched_text"],
                    m["normalized_text"],
                    m["match_kind"],
                    m["resolution"],
                    m["term_id"],
                    tuple(candidates[m["mention_id"]]),
                    m["quality_rule"],
                )
                for m in db.execute(
                    "SELECT * FROM mentions WHERE figure_id = ? "
                    "ORDER BY start_offset, end_offset, mention_id",
                    (figure_id,),
                )
            )
        linked = self._by_figure.get(figure_id, frozenset())
        summaries = tuple(
            sorted(
                (
                    ConceptSummary(term_id, self._concept_labels[term_id])
                    for term_id in linked
                ),
                key=lambda c: (c.label.casefold(), c.term_id),
            )
        )
        return Figure(
            row["figure_id"],
            row["figure_label"],
            row["title"],
            row["working_group"],
            row["chapter"],
            row["section"],
            row["caption"],
            row["caption_status"],
            images,
            row["source_url"],
            row["copyright_policy_url"],
            summaries,
            mentions,
        )

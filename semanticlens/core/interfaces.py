"""Component contracts for the SemanticLens document intelligence pipeline.

The protocols describe how components exchange Core data structures without
coupling the pipeline to a particular parser, model, or storage backend.
Concrete implementations and their dependencies live in their own modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeAlias

from .schemas import (
    EmbeddingSpec,
    EvidenceUnit,
    GenerationDraft,
    Query,
    RetrievalCandidate,
    SearchHit,
)

Vector: TypeAlias = Sequence[float]
Embedding: TypeAlias = Vector | Sequence[Vector]
IndexMatch: TypeAlias = tuple[str, float]


class DocumentProcessor(Protocol):
    """Turn a source PDF into evidence units and their rendered page images."""

    def process(self, pdf_path: Path) -> Iterable[EvidenceUnit]:
        """Yield evidence units for one PDF.

        Each page used for visual retrieval must have an ``image_ref`` that
        the embedding implementation can resolve. Directory traversal belongs
        to the pipeline, not the processor.
        """
        ...


class Embedder(Protocol):
    """Create compatible embeddings for evidence units and text queries."""

    @property
    def spec(self) -> EmbeddingSpec:
        """Describe the embeddings produced by this implementation."""
        ...

    def embed_units(self, units: Sequence[EvidenceUnit]) -> Sequence[Embedding]:
        """Return one embedding per unit, preserving input order.

        Each embedding must match ``spec``. Backend tensors are converted to
        plain numerical sequences before crossing the Core boundary.
        """
        ...

    def embed_query(self, query: Query) -> Embedding:
        """Encode a query using the embedding representation in ``spec``."""
        ...


class VectorIndex(Protocol):
    """Store and search evidence embeddings using a compatible specification."""

    @property
    def spec(self) -> EmbeddingSpec:
        """Describe the representation accepted by this index."""
        ...

    def upsert(self, vectors: Mapping[str, Embedding]) -> None:
        """Store embeddings by evidence ID, replacing existing entries."""
        ...

    def search(
        self, query_embedding: Embedding, *, top_k: int
    ) -> Sequence[IndexMatch]:
        """Return up to ``top_k`` ranked (evidence ID, score) matches.

        The index performs the interaction specified by ``spec``. Retrieval
        scores are similarity values, not probabilities.
        """
        ...


class Retriever(Protocol):
    """Find ranked evidence candidates for a query."""

    def retrieve(
        self,
        query: Query,
        *,
        embedder: Embedder,
        index: VectorIndex,
        top_k: int,
    ) -> Sequence[RetrievalCandidate]:
        """Retrieve evidence IDs using the provided embedding and index components.

        ``top_k`` must be positive. Retrieval strategies may use additional
        dependencies configured on their concrete implementations.
        """
        ...


class EvidenceResolver(Protocol):
    """Persist evidence metadata and resolve ranked candidates to sources."""

    def add(self, units: Sequence[EvidenceUnit]) -> None:
        """Persist evidence by unit ID, replacing records with matching IDs."""
        ...

    def resolve(
        self, candidates: Sequence[RetrievalCandidate]
    ) -> Sequence[SearchHit]:
        """Resolve every candidate without changing its order, score, or label.

        Missing evidence is an error: never silently drop or substitute a
        source for a retrieved candidate.
        """
        ...


class Generator(Protocol):
    """Produce a structured answer draft using resolved document evidence.

    The generator receives a ``Query`` and the exact ``SearchHit`` objects
    selected for generation. A visual implementation reads the referenced
    source images and prepares its multimodal model input internally; Core
    does not depend on image or model libraries.

    Draft evidence IDs are *claims* made by the model, not verified citations.
    Downstream grounding must resolve them against the supplied hits before
    constructing an ``AnswerResult``.
    """

    def generate(
        self, query: Query, *, hits: Sequence[SearchHit]
    ) -> GenerationDraft:
        """Return an untrusted structured answer draft for the given evidence.

        - Use only evidence from ``hits``; do not invent document references.
        - Refer to sources using actual ``EvidenceUnit.unit_id`` values, not
          temporary prompt labels or model-generated page numbers.
        - Do not silently exclude supplied hits while keeping them eligible
          for citation. Any evidence limit must be applied before this call.
        - For empty ``hits``, return an ``insufficient_evidence`` draft without
          invoking the model to guess an answer.
        - Do not mutate ``query`` or the supplied evidence.

        Model selection, inference options, image loading and failure handling
        belong to the concrete generator. The pipeline may call its optional
        ``close()`` method during cleanup, as with other components.
        """
        ...

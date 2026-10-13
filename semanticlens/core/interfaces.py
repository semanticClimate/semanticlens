"""
Interfaces that connect the SemanticLens pipeline components

These contracts define what each component must do not how it does it

A PDF processor, embedding model or vector database can be replaced as long
as its implementation follows the same interface

Interfaces use only shared Core types. Model inference, PDF rendering and
vector database operations belong in their respective modules
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeAlias

from .schemas import (
    EmbeddingSpec,
    EvidenceUnit,
    Query,
    RetrievalCandidate,
    SearchHit,
)

Vector: TypeAlias = Sequence[float]
Embedding: TypeAlias = Vector | Sequence[Vector]
IndexMatch: TypeAlias = tuple[str, float]

class DocumentProcessor(Protocol):
    """
    Turn a PDF into searchable evidence units and their page images
    """

    def process(self, pdf_path: Path) -> Iterable[EvidenceUnit]:
        """
        Process one PDF and yield its evidence units

        For visual retrieval, each page unit must have an image_ref that
        the embedder can use to load the rendered page. A directory of PDFs
        is handled by the pipeline, not by this method.
        """
        ...


class Embedder(Protocol):
    """
    Create compatible embeddings for document units and text queries
    """

    @property
    def spec(self) -> EmbeddingSpec:
        """
        Describe the model and preprocessing used for these embeddings
        """
        ...

    def embed_units(self, units: Sequence[EvidenceUnit]) -> Sequence[Embedding]:
        """
        Embed a batch of evidence units, preserving their input order

        Return exactly one embedding per input unit. Each embedding must
        match spec including its vector dimension and representation

        Model specific tensors are converted to numeric sequences here
        """
        ...

    def embed_query(self, query: Query) -> Embedding:
        """
        Embed a text query using the same embedding specification
        """
        ...


class VectorIndex(Protocol):
    """
    Store evidence embeddings and search them by similarity
    """

    @property
    def spec(self) -> EmbeddingSpec:
        """
        Describe the embeddings this index accepts and searches
        """
        ...

    def upsert(self, vectors: Mapping[str, Embedding]) -> None:
        """
        Store embeddings by unit ID, replacing any existing entries

        Each embedding must match spec. The index receives only IDs
        and embeddings; source details are resolved separately.
        """
        ...

    def search(
        self, query_embedding: Embedding, *, top_k: int
    ) -> Sequence[IndexMatch]:
        """
        Return up to top_k (unit ID, score) pairs, best match first

        top_k must be positive. Scores are retrieval similarities,
        not probabilities. The index implementation performs the 
        interaction defined by spec (such as MaxSim for multi-vector embeddings)
        """
        ...


class Retriever(Protocol):
    """
    Turn a user query into ranked evidence candidates
    """

    def retrieve(
        self,
        query: Query,
        *,
        embedder: Embedder,
        index: VectorIndex,
        top_k: int,
    ) -> Sequence[RetrievalCandidate]:
        """
        Use the supplied components to retrieve relevant evidence IDs

        top_k must be positive. The pipeline passes its own embedder
        and index, so retrieval uses the same components as indexing. 

        A retrieval strategy may also use additional dependencies configured on its implementation.
        """
        ...


class EvidenceResolver(Protocol):
    """
    Store source references and attach them to retrieval results
    """

    def add(self, units: Sequence[EvidenceUnit]) -> None:
        """
        Persist evidence metadata by unit ID, replacing existing records
        """
        ...

    def resolve(
        self, candidates: Sequence[RetrievalCandidate]
    ) -> Sequence[SearchHit]:
        """
        Return complete search hits in the same order as the candidates.

        Every candidate must resolve to its original document and page.
        
        Missing evidence should raise a clear error rather than silently
        dropping or misattributing a result
        """
        ...

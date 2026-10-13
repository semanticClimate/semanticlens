"""
Connect the SemanticLens components into an indexing and search pipeline

This module controls the order of operations. PDF rendering, embedding models,
vector databases and evidence storage are implemented by other modules
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType

from .config import PipelineConfig
from .exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingCompatibilityError,
    EmbeddingError,
    EvidenceResolutionError,
    IndexOperationError,
    RetrievalError,
    SemanticLensError,
)
from .interfaces import (
    DocumentProcessor,
    Embedder,
    EvidenceResolver,
    Retriever,
    VectorIndex,
)
from .schemas import (
    EmbeddingSpec,
    EvidenceUnit,
    IndexSummary,
    Query,
    RetrievalCandidate,
    SearchHit,
)


class SemanticLens:
    """Index PDF evidence and retrieve matching pages using pluggable components.

    Supply implementations of the five Core interfaces. The pipeline does not
    create those components or manage their storage locations; configure them
    separately to use the same workspace and persistent collection.

    Indexing writes evidence metadata before vectors, so a newly searchable
    vector has a source reference. Writes are not transactional: if a batch
    fails, earlier batches (or some records in that batch) may remain stored.
    Re-indexing the same IDs is supported by the component upsert contracts.
    """

    def __init__(
        self,
        *,
        processor: DocumentProcessor,
        embedder: Embedder,
        index: VectorIndex,
        retriever: Retriever,
        resolver: EvidenceResolver,
        config: PipelineConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(config, PipelineConfig):
            raise ConfigurationError("config must be a PipelineConfig instance")

        components = (
            ("processor", processor, ("process",)),
            ("embedder", embedder, ("embed_units", "embed_query")),
            ("index", index, ("upsert", "search")),
            ("retriever", retriever, ("retrieve",)),
            ("resolver", resolver, ("add", "resolve")),
        )
        for name, component, methods in components:
            for method in methods:
                if not callable(getattr(component, method, None)):
                    raise ConfigurationError(f"{name} must implement {method}()")

        self.config = config if config is not None else PipelineConfig()
        self._processor = processor
        self._embedder = embedder
        self._vector_index = index
        self._retriever = retriever
        self._resolver = resolver
        self._closed = False
        self._check_embedding_compatibility()

    def index(self, source: str | Path) -> IndexSummary:
        """Index one PDF or all PDFs in a directory (including subdirectories).

        Pages are embedded in batches. Evidence IDs must be unique within this
        call. Existing IDs from an earlier call may be replaced by the backend.
        """
        self._require_open()
        self._check_embedding_compatibility()
        pdf_files = self._find_pdfs(source)

        documents_processed = 0
        units_indexed = 0
        seen_ids: set[str] = set()
        batch: list[EvidenceUnit] = []

        for pdf_path in pdf_files:
            for unit in self._read_units(pdf_path):
                if unit.unit_id in seen_ids:
                    raise DocumentProcessingError(
                        f"Duplicate evidence unit ID during indexing: {unit.unit_id!r}"
                    )
                seen_ids.add(unit.unit_id)
                batch.append(unit)

                if len(batch) == self.config.batch_size:
                    self._store_batch(batch)
                    units_indexed += len(batch)
                    batch.clear()

            documents_processed += 1

        if batch:
            self._store_batch(batch)
            units_indexed += len(batch)

        return IndexSummary(
            documents_processed=documents_processed,
            units_indexed=units_indexed,
            units_skipped=0,
        )

    def search(
        self, query: str | Query, *, top_k: int | None = None
    ) -> list[SearchHit]:
        """Find relevant evidence and return source-linked results in rank order."""
        self._require_open()
        self._check_embedding_compatibility()

        if isinstance(query, str):
            query = Query(text=query)
        elif not isinstance(query, Query):
            raise TypeError("query must be a string or Query instance")

        limit = self.config.default_top_k if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("top_k must be a positive integer")

        try:
            candidates = self._retriever.retrieve(
                query, embedder=self._embedder, index=self._vector_index, top_k=limit
            )
        except SemanticLensError:
            raise
        except Exception as exc:
            raise RetrievalError("Failed to retrieve evidence candidates") from exc

        if not isinstance(candidates, Sequence) or len(candidates) > limit:
            raise RetrievalError("Retriever returned an invalid number of candidates")
        if any(not isinstance(item, RetrievalCandidate) for item in candidates):
            raise RetrievalError("Retriever must return RetrievalCandidate objects")
        candidate_ids = [item.unit_id for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RetrievalError("Retriever returned duplicate evidence IDs")

        try:
            hits = self._resolver.resolve(candidates)
        except SemanticLensError:
            raise
        except Exception as exc:
            raise EvidenceResolutionError(
                "Failed to resolve retrieved evidence"
            ) from exc

        if not isinstance(hits, Sequence) or len(hits) != len(candidates):
            raise EvidenceResolutionError("Resolver must return one hit per candidate")
        for candidate, hit in zip(candidates, hits):
            if not isinstance(hit, SearchHit) or (
                hit.evidence.unit_id != candidate.unit_id
                or hit.score != candidate.score
                or hit.retriever != candidate.retriever
            ):
                raise EvidenceResolutionError(
                    "Resolver changed or reordered the retrieved candidates"
                )

        return list(hits)

    def close(self) -> None:
        """Close components that provide a close() method; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        closed_ids: set[int] = set()

        for component in (
            self._resolver,
            self._retriever,
            self._vector_index,
            self._embedder,
            self._processor,
        ):
            if id(component) in closed_ids:
                continue
            closed_ids.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc

        if first_error is not None:
            raise SemanticLensError(
                "Failed to close a pipeline component"
            ) from first_error

    def __enter__(self) -> SemanticLens:
        """Use the pipeline with a ``with`` block for automatic cleanup."""
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise SemanticLensError("This SemanticLens pipeline is closed")

    def _check_embedding_compatibility(self) -> None:
        try:
            embedder_spec = self._embedder.spec
            index_spec = self._vector_index.spec
        except Exception as exc:
            raise ConfigurationError("Could not read embedding specifications") from exc

        if not isinstance(embedder_spec, EmbeddingSpec) or not isinstance(
            index_spec, EmbeddingSpec
        ):
            raise ConfigurationError("Embedder and index must expose an EmbeddingSpec")
        if embedder_spec != index_spec:
            raise EmbeddingCompatibilityError(
                "Embedder and index use different embedding specifications"
            )

    @staticmethod
    def _find_pdfs(source: str | Path) -> list[Path]:
        if not isinstance(source, (str, Path)):
            raise TypeError("source must be a PDF path or directory path")
        if isinstance(source, str) and not source.strip():
            raise ValueError("source path must not be empty")

        path = Path(source).expanduser()
        try:
            if path.is_file():
                if path.suffix.lower() != ".pdf":
                    raise DocumentProcessingError(f"Not a PDF file: {path}")
                return [path]
            if path.is_dir():
                pdfs = sorted(
                    item for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() == ".pdf"
                )
                if not pdfs:
                    raise DocumentProcessingError(f"No PDF files found in: {path}")
                return pdfs
        except OSError as exc:
            raise DocumentProcessingError(
                f"Cannot access document path: {path}"
            ) from exc
        raise DocumentProcessingError(f"Document path does not exist: {path}")

    def _read_units(self, pdf_path: Path) -> Iterator[EvidenceUnit]:
        try:
            for unit in self._processor.process(pdf_path):
                if not isinstance(unit, EvidenceUnit):
                    raise TypeError("DocumentProcessor must yield EvidenceUnit objects")
                yield unit
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to process PDF: {pdf_path}") from exc

    def _store_batch(self, units: Sequence[EvidenceUnit]) -> None:
        try:
            embeddings = self._embedder.embed_units(units)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to embed document evidence") from exc

        if not isinstance(embeddings, Sequence) or len(embeddings) != len(units):
            raise EmbeddingError("Embedder must return one embedding per evidence unit")
        vectors = dict(zip((unit.unit_id for unit in units), embeddings))

        # Store provenance first: searchable vectors must have resolvable evidence.
        try:
            self._resolver.add(units)
        except EvidenceResolutionError:
            raise
        except Exception as exc:
            raise EvidenceResolutionError("Failed to store evidence metadata") from exc

        try:
            self._vector_index.upsert(vectors)
        except IndexOperationError:
            raise
        except Exception as exc:
            raise IndexOperationError("Failed to store evidence embeddings") from exc

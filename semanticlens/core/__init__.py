"""Public, model-independent interfaces and data types for SemanticLens."""

from .config import PipelineConfig
from .interfaces import (
    DocumentProcessor,
    Embedder,
    EvidenceResolver,
    Generator,
    Retriever,
    VectorIndex,
)
from .pipeline import SemanticLens
from .schemas import (
    AnswerCitation,
    AnswerResult,
    DocumentRef,
    EmbeddingSpec,
    EvidenceUnit,
    GeneratedStatement,
    GenerationDraft,
    IndexSummary,
    Query,
    RetrievalCandidate,
    SearchHit,
)

__all__ = [
    "SemanticLens",
    "PipelineConfig",
    "DocumentProcessor",
    "Embedder",
    "EvidenceResolver",
    "Generator",
    "Retriever",
    "VectorIndex",
    "AnswerCitation",
    "AnswerResult",
    "DocumentRef",
    "EmbeddingSpec",
    "EvidenceUnit",
    "GeneratedStatement",
    "GenerationDraft",
    "IndexSummary",
    "Query",
    "RetrievalCandidate",
    "SearchHit",
]

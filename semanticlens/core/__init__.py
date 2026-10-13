"""
Public API for SemanticLens core components
"""

from .config import PipelineConfig
from .interfaces import (
    DocumentProcessor,
    Embedder,
    EvidenceResolver,
    Retriever,
    VectorIndex,
)
from .pipeline import SemanticLens
from .schemas import (
    DocumentRef,
    EmbeddingSpec,
    EvidenceUnit,
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
    "Retriever",
    "VectorIndex",
    "DocumentRef",
    "EmbeddingSpec",
    "EvidenceUnit",
    "IndexSummary",
    "Query",
    "RetrievalCandidate",
    "SearchHit",
]

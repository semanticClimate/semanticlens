"""
Exceptions used across the SemanticLens pipeline

Catch a specific exception when you can handle that failure or catch
SemanticLensError to handle any SemanticLens specific failure.

When wrapping an error from another library use `raise ... from error`
to preserve the original cause.
"""


class SemanticLensError(Exception):
    """
    Base class for errors raised by SemanticLens components
    """


class ConfigurationError(SemanticLensError):
    """
    Raised when pipeline settings or connected components are incompatible

    Invalid `PipelineConfig` field values are reported by Pydantic itself
    """


class DocumentProcessingError(SemanticLensError):
    """
    Raised when a source document cannot be read or processed
    """


class EmbeddingError(SemanticLensError):
    """
    Raised when document or query embedding generation fails
    """


class EmbeddingCompatibilityError(EmbeddingError):
    """
    Raised when embedding and index specifications do not match
    """


class IndexOperationError(SemanticLensError):
    """
    Raised when creating, loading, updating, or searching an index fails
    """


class RetrievalError(SemanticLensError):
    """
    Raised when a retrieval strategy cannot complete a search
    """


class EvidenceResolutionError(SemanticLensError):
    """
    Raised when a retrieved result cannot be linked to its source evidence
    """

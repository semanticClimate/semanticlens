"""Exceptions shared by SemanticLens pipeline components.

Catch a specific subclass when a failure can be handled locally, or catch
SemanticLensError to handle errors raised by the SemanticLens toolkit.

When translating a third-party exception, use ``raise ... from exc`` to
preserve its original cause for debugging.
"""


class SemanticLensError(Exception):
    """Base class for errors raised by SemanticLens components."""


class ConfigurationError(SemanticLensError):
    """Raised when pipeline settings or connected components are incompatible.

    Invalid ``PipelineConfig`` fields are reported by Pydantic itself.
    """


class DocumentProcessingError(SemanticLensError):
    """Raised when a source document cannot be read or processed."""


class EmbeddingError(SemanticLensError):
    """Raised when document or query embedding generation fails."""


class EmbeddingCompatibilityError(EmbeddingError):
    """Raised when embedding and index specifications do not match."""


class IndexOperationError(SemanticLensError):
    """Raised when creating, loading, updating, or searching an index fails."""


class RetrievalError(SemanticLensError):
    """Raised when a retrieval strategy cannot complete a search."""


class EvidenceResolutionError(SemanticLensError):
    """Raised when a retrieved result cannot be linked to its source evidence."""


class GenerationError(SemanticLensError):
    """Raised when evidence-grounded answer generation cannot complete.

    Examples include model inference failures, invalid generated output, or
    failure to validate model-provided evidence references.

    An answer that cannot be supported by the retrieved evidence is a normal
    ``insufficient_evidence`` result, not a ``GenerationError``.
    """

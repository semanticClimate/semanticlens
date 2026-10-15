"""
Shared data structures for SemanticLens

This module defines the common data formats used across the VDR pipeline

It allows different components to exchange data consistently without
depending on specific models, parsers or databases
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class CoreSchema(BaseModel):
    """
    Base model for shared data structures with strict validation
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
    )


class DocumentRef(CoreSchema):
    """
    Information used to identify and reference an original document
    """

    document_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    title: str | None = None
    page_count: int | None = Field(default=None, gt=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceUnit(CoreSchema):
    """
    A searchable part of a document
    
    Currently represents document pages with support for figures,
    tables, and smaller regions as the pipeline evolves
    """

    unit_id: str = Field(min_length=1)
    document: DocumentRef
    page_idx: int = Field(ge=0)  
    kind: str = Field(default="page", min_length=1)
    bbox: tuple[float, float, float, float] | None = None
    image_ref: str | None = None  
    text: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x0, y0, x1, y1 = value
        if not all(0 <= c <= 1 for c in value) or x0 >= x1 or y0 >= y1:
            raise ValueError(
                "bbox must be normalized top-left (x0, y0, x1, y1), "
                "within [0, 1], with x0 < x1 and y0 < y1"
            )
        return value

    @model_validator(mode="after")
    def validate_page(self) -> EvidenceUnit:
        if self.document.page_count is not None and self.page_idx >= self.document.page_count:
            raise ValueError("page_idx is outside the source document")
        return self

    @property
    def page_number(self) -> int:
        """Human-facing one-based page number (not necessarily printed page label)."""
        return self.page_idx + 1


class Query(CoreSchema):
    """
    A user's search query used to retrieve relevant document evidence
    """

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            raise ValueError("query text must not be blank")
        return stripped


class EmbeddingSpec(CoreSchema):
    """
    Describes how document and query embeddings are created and compared
    
    Stores model and preprocessing information to help ensure that
    embeddings are compatible with the index used for retrieval
    """

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    preprocessing_id: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    representation: Literal["single_vector", "multi_vector"]
    interaction: Literal["none", "maxsim"]
    similarity: Literal["cosine", "dot"]

    @model_validator(mode="after")
    def validate_interaction(self) -> EmbeddingSpec:
        if self.representation == "multi_vector" and self.interaction != "maxsim":
            raise ValueError("V0.1 multi-vector embeddings require maxsim")
        if self.representation == "single_vector" and self.interaction != "none":
            raise ValueError("Single-vector embeddings require interaction='none'")
        return self


class RetrievalCandidate(CoreSchema):
    """
    A retrieved evidence ID and its relevance score, before source details are loaded
    """

    unit_id: str = Field(min_length=1)
    score: float  
    retriever: str = Field(min_length=1)


class SearchHit(CoreSchema):
    """
    A final search result containing retrieved evidence and its relevance score
    """

    evidence: EvidenceUnit
    score: float
    retriever: str = Field(min_length=1)


class IndexSummary(CoreSchema):
    """
    Summary of the documents and evidence units processed during indexing
    """

    documents_processed: int = Field(ge=0)
    units_indexed: int = Field(ge=0)
    units_skipped: int = Field(default=0, ge=0)


class GeneratedStatement(CoreSchema):
    """One answer statement and the retrieved evidence units it refers to.

    The generator must resolve any short prompt labels (such as E1 or E2) to
    actual EvidenceUnit.unit_id values before producing this object. A reference
    here is only a model assertion until validated against the retrieved hits.
    """

    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()

    @field_validator("text")
    @classmethod
    def nonblank_statement(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("statement text must not be blank")
        return text

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not unit_id or unit_id != unit_id.strip() for unit_id in value):
            raise ValueError("evidence IDs must be nonblank and have no surrounding whitespace")
        if len(value) != len(set(value)):
            raise ValueError("evidence IDs within a statement must be unique")
        return value


class GenerationDraft(CoreSchema):
    """Untrusted, structured output from a generation backend.

    These references have not yet been checked against SearchHit objects.
    The grounding stage validates them and constructs an AnswerResult.
    """

    status: Literal["answered", "insufficient_evidence"]
    statements: tuple[GeneratedStatement, ...] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)

    @field_validator("model_id", "model_revision")
    @classmethod
    def validate_model_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("model identity must be nonblank without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_answered_statements(self) -> GenerationDraft:
        if self.status == "answered" and any(
            not statement.evidence_ids for statement in self.statements
        ):
            raise ValueError("every answered statement must cite at least one evidence unit")
        return self


class AnswerCitation(CoreSchema):
    """A numbered citation pointing to a real, resolved evidence unit."""

    number: int = Field(gt=0)
    evidence: EvidenceUnit

    @property
    def marker(self) -> str:
        """Citation marker inserted into the rendered answer, e.g. '[1]'."""
        return f"[{self.number}]"


class AnswerResult(CoreSchema):
    """Final result of a grounded question-answering operation.

    The grounding stage builds this from a validated GenerationDraft and the
    exact SearchHit objects used for generation. Citation references are
    checked structurally; scientific support is not automatically proven.
    """

    query: Query
    answer: str = Field(min_length=1)
    status: Literal["answered", "insufficient_evidence"]
    statements: tuple[GeneratedStatement, ...] = Field(min_length=1)
    citations: tuple[AnswerCitation, ...] = ()
    hits: tuple[SearchHit, ...] = ()
    model_id: str = Field(min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)

    @field_validator("answer")
    @classmethod
    def nonblank_answer(cls, value: str) -> str:
        answer = value.strip()
        if not answer:
            raise ValueError("answer must not be blank")
        return answer

    @field_validator("model_id", "model_revision")
    @classmethod
    def validate_model_identity(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("model identity must be nonblank without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_citations(self) -> AnswerResult:
        if self.status == "answered":
            if not self.citations:
                raise ValueError("an answered result must have at least one citation")
            if any(not statement.evidence_ids for statement in self.statements):
                raise ValueError("every answered statement must cite evidence")

        hit_by_id = {hit.evidence.unit_id: hit.evidence for hit in self.hits}
        if len(hit_by_id) != len(self.hits):
            raise ValueError("retrieved hits must have unique evidence IDs")

        if tuple(citation.number for citation in self.citations) != tuple(
            range(1, len(self.citations) + 1)
        ):
            raise ValueError("citation numbers must be consecutive, starting at 1")

        cited_ids: set[str] = set()
        for citation in self.citations:
            unit_id = citation.evidence.unit_id
            if unit_id in cited_ids:
                raise ValueError("each evidence unit can have only one citation number")
            cited_ids.add(unit_id)

            if hit_by_id.get(unit_id) != citation.evidence:
                raise ValueError("citations must refer to the original retrieved evidence")
            if citation.marker not in self.answer:
                raise ValueError(f"citation marker {citation.marker} is missing from the answer")

        referenced_ids = {
            unit_id for statement in self.statements for unit_id in statement.evidence_ids
        }
        if referenced_ids != cited_ids:
            raise ValueError("statement evidence IDs must match the validated citations")

        return self

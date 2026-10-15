"""Bind generated statements to the evidence supplied by SemanticLens retrieval.

GenerationDraft is untrusted model output. This module checks that every cited
unit ID belongs to the exact SearchHit sequence given to the generator, then
builds a deterministic, source-linked AnswerResult. It never asks the model to
invent document metadata or source page numbers.

These checks establish *citation provenance*, not factual entailment: a real
retrieved page may still fail to support the claim made about it. Scientific
claim verification requires separate evaluation or verification methods.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import ValidationError

from semanticlens.core.exceptions import GenerationError
from semanticlens.core.schemas import (
    AnswerCitation,
    AnswerResult,
    GenerationDraft,
    Query,
    SearchHit,
)


# The model must not insert its own numeric citation markers: those are
# assigned only after its evidence IDs have been checked against SearchHit.
# This intentionally matches [1], [23], etc., not scientific intervals [1, 3].
_MODEL_CITATION_PATTERN = re.compile(r"\[[1-9][0-9]*\]")


def build_answer_result(
    query: Query,
    *,
    draft: GenerationDraft,
    hits: Sequence[SearchHit],
) -> AnswerResult:
    """Return a source-linked answer from a model draft and retrieved hits.

    The original retrieval order, scores, and complete EvidenceUnit objects
    are preserved in ``AnswerResult.hits``. Citations are numbered in order
    of first appearance in the generated statements, starting at one. A
    repeated reference to the same evidence unit always uses the same number.

    ``insufficient_evidence`` returns one explanation without citations but
    retains the retrieved hits for inspection. Missing, duplicate, or foreign
    references raise GenerationError rather than fabricating provenance.

    This function performs no model inference, file I/O, or scientific
    entailment checks and does not mutate its inputs.
    """
    if not isinstance(query, Query):
        raise TypeError("query must be a Query instance")
    if not isinstance(draft, GenerationDraft):
        raise TypeError("draft must be a GenerationDraft instance")
    if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
        raise TypeError("hits must be a sequence of SearchHit objects")

    # Take a snapshot so validation and the result operate on one hit set.
    resolved_hits = tuple(hits)
    if any(not isinstance(hit, SearchHit) for hit in resolved_hits):
        raise TypeError("hits must contain only SearchHit objects")

    hit_by_id: dict[str, SearchHit] = {}
    for hit in resolved_hits:
        unit_id = hit.evidence.unit_id
        if unit_id in hit_by_id:
            raise GenerationError(f"Duplicate retrieved evidence ID: {unit_id!r}")
        hit_by_id[unit_id] = hit

    # Reject numeric markers written as plain model text, even when a real
    # citation with that number happens to exist. Only this function assigns
    # trusted-looking [N] markers in the final answer.
    if any(_MODEL_CITATION_PATTERN.search(statement.text) for statement in draft.statements):
        raise GenerationError("Generated statements must not contain citation markers")

    if draft.status == "insufficient_evidence":
        if len(draft.statements) != 1 or draft.statements[0].evidence_ids:
            raise GenerationError(
                "Insufficient-evidence output requires one uncited explanation"
            )
        answer = draft.statements[0].text
        citations: tuple[AnswerCitation, ...] = ()
    else:
        if not resolved_hits:
            raise GenerationError("An answered draft requires retrieved evidence")

        citation_numbers: dict[str, int] = {}
        citation_list: list[AnswerCitation] = []
        answer_parts: list[str] = []

        for statement in draft.statements:
            if not statement.evidence_ids:
                raise GenerationError("Each answered statement must reference evidence")

            markers: list[str] = []
            for unit_id in statement.evidence_ids:
                hit = hit_by_id.get(unit_id)
                if hit is None:
                    raise GenerationError(
                        f"Generated answer references evidence not retrieved: {unit_id!r}"
                    )

                if unit_id not in citation_numbers:
                    number = len(citation_list) + 1
                    citation_numbers[unit_id] = number
                    citation_list.append(
                        AnswerCitation(number=number, evidence=hit.evidence)
                    )
                markers.append(f"[{citation_numbers[unit_id]}]")

            answer_parts.append(f"{statement.text} {' '.join(markers)}")

        answer = "\n\n".join(answer_parts)
        citations = tuple(citation_list)

    try:
        return AnswerResult(
            query=query,
            answer=answer,
            status=draft.status,
            statements=draft.statements,
            citations=citations,
            hits=resolved_hits,
            model_id=draft.model_id,
            model_revision=draft.model_revision,
        )
    except ValidationError as exc:
        raise GenerationError("Could not construct a valid grounded answer") from exc

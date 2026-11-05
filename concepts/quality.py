"""Auditable, conservative quality rules for IPCC caption concept mentions.

Runs *after* extraction and glossary normalization. It fixes one documented
upstream alias collision and excludes a narrow bibliographic false positive.
Every decision retains the original caption span and candidate concept IDs.
No inferred synonyms, ontology relations, or visual-content claims are made.

The IPCC glossary uses CO2 for carbon dioxide and CO2-eq for carbon dioxide
*equivalent* (distinct concepts). The derived vocabulary also assigns the
unqualified alias CO2 to 'Equivalent carbon dioxide emission'. This module
resolves that collision by notation, never by a figure's Working Group.

Scientific reference: https://www.ipcc.ch/report/ar6/syr/annexes-and-index/
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal

from concepts.normalize import NormalizedMention
from concepts.vocabulary import Vocabulary, normalize_term


QualityResolution = Literal[
    "lexical", "explicit_expansion", "ambiguous", "chemical_formula", "excluded"
]
Rule = Literal["co2_formula", "co2_equivalent_qualifier", "citation_source_label"]

# A qualifier *directly following* CO2 denotes an equivalent metric.
# We don't infer meaning from a broad sentence window, WG, or other concepts.
_EQ_SUFFIX = re.compile(
    r"\A\s*(?:[-‐‑‒–—]\s*)?(?:eq(?:uivalent)?s?|e)\b", re.IGNORECASE
)
_SOURCE_LABEL = re.compile(r"\A\s*:\s*")


@dataclass(frozen=True, slots=True)
class AssessedMention:
    """The original lexical match plus its transparent quality decision.

    ``original`` is *never modified*. ``term_id`` is None for both ambiguous
    and explicitly excluded records; their candidates remain inspectable.
    The annotation ``rule`` identifies any automatic intervention.
    """

    original: NormalizedMention
    term_id: str | None
    resolution: QualityResolution
    rule: Rule | None = None

    @property
    def is_excluded(self) -> bool:
        return self.resolution == "excluded"

    @property
    def is_ambiguous(self) -> bool:
        return self.resolution == "ambiguous"


class MentionQuality:
    """Deterministic mention adjudication using exact, vocabulary-backed IDs."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("vocabulary must be a Vocabulary")
        self._terms = {term.term_id: term for term in vocabulary.terms}
        self._by_label = {normalize_term(t.label): t.term_id for t in vocabulary.terms}

    def assess(
        self, mentions: Iterable[NormalizedMention], caption: str | None
    ) -> tuple[AssessedMention, ...]:
        """Apply narrow, provenance-backed corrections to one figure's mentions.

        Unmatched terminology is outside this function's scope. Already
        resolved concepts are retained except bibliographic ``Source:``.
        Ambiguities not covered by the source-backed CO2 rule stay unresolved.
        """
        if caption is not None and not isinstance(caption, str):
            raise TypeError("caption must be a string or None")
        text = caption or ""
        result: list[AssessedMention] = []
        figure_id: str | None = None
        last_end = 0
        for mention in mentions:
            if not isinstance(mention, NormalizedMention):
                raise TypeError("Expected NormalizedMention entries")
            if figure_id is None:
                figure_id = mention.figure_id
            elif mention.figure_id != figure_id:
                raise ValueError("Mentions from different figures cannot be assessed together")
            if (mention.start < last_end or mention.start < 0
                    or mention.end > len(text) or mention.start >= mention.end
                    or text[mention.start:mention.end] != mention.matched_text):
                raise ValueError("Invalid or non-monotonic original caption evidence")
            last_end = mention.end

            if mention.term_id is not None:
                term = self._terms.get(mention.term_id)
                if term is None:
                    raise ValueError(f"Unknown concept ID: {mention.term_id}")
                if mention.term_id not in mention.candidate_term_ids:
                    raise ValueError("Resolved concept is not among lexical candidates")
                # 'Source: ...' in figure captions introduces bibliography,
                # not the glossary's GHG-emitting 'Source' process.
                if (normalize_term(term.label) == "source"
                        and mention.matched_text == "Source"
                        and _SOURCE_LABEL.match(text[mention.end:])):
                    result.append(AssessedMention(mention, None, "excluded", "citation_source_label"))
                else:
                    result.append(AssessedMention(mention, mention.term_id, mention.resolution))
                continue

            if mention.resolution != "ambiguous" or not mention.candidate_term_ids:
                raise ValueError("Invalid unresolved normalized mention")
            if mention.normalized_text not in {"co2", "co 2"}:
                result.append(AssessedMention(mention, None, "ambiguous"))
                continue

            # This decision is only eligible when *both* IDs occur among the
            # original extractor's candidates: no new concept is invented.
            carbon = self._by_label.get("carbon dioxide")
            equivalent = self._by_label.get("equivalent carbon dioxide emission")
            candidate_ids = set(mention.candidate_term_ids)
            if not carbon or not equivalent or not {carbon, equivalent} <= candidate_ids:
                result.append(AssessedMention(mention, None, "ambiguous"))
                continue

            is_equivalent = bool(_EQ_SUFFIX.match(text[mention.end:]))
            term_id = equivalent if is_equivalent else carbon
            rule: Rule = "co2_equivalent_qualifier" if is_equivalent else "co2_formula"
            result.append(AssessedMention(mention, term_id, "chemical_formula", rule))

        return tuple(result)

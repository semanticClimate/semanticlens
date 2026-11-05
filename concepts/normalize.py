"""
Normalize IPCC caption mentions using source-backed glossary terms.

- Map extracted term IDs to canonical glossary labels.
- Preserve original evidence spans.
- Keep ambiguous terms unresolved for later review.
- Resolve ambiguity only using an adjacent parenthetical expansion
  in the same caption, such as CO2 (carbon dioxide).
- Do not use co-occurrence, working-group labels, or guessed synonyms.
- Run after ConceptExtractor and before corpus/database building.
- Do not access the network, filesystem, modify vocabulary, or write to databases.
- Lexical matches do not confirm visual depiction or validated ontology relationships.

Usage:
    from concepts.vocabulary import load_vocabulary
    from concepts.extract import ConceptExtractor

    vocabulary = load_vocabulary()
    extractor = ConceptExtractor(vocabulary)
    normalizer = ConceptNormalizer(vocabulary)

    caption = "CO2 (carbon dioxide) contributes to warming."
    mentions = normalizer.normalize(
        extractor.extract("figure-1", caption), caption
    )
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal

from concepts.extract import ConceptMention, MatchKind
from concepts.vocabulary import Vocabulary, normalize_term


Resolution = Literal["lexical", "explicit_expansion", "ambiguous"]

# Require an immediately adjacent pair such as ``full name (ABBR)`` or
# ``ABBR (full name)``; do not extrapolate from unrelated surrounding text.
_OPEN_PAREN = re.compile(r"\s*\(\s*\Z")
_CLOSE_PAREN = re.compile(r"\A\s*\)")


@dataclass(frozen=True, slots=True)
class NormalizedMention:
    """
    A caption mention with an optional canonical glossary term identity.

    ``term_id`` and ``canonical_label`` are None when unresolved. The original
    caption text and [start, end) offsets are unchanged. ``candidate_term_ids``
    always retains the complete set of lexical candidates, even if an explicit
    in-caption expansion selects one. These IDs refer to *glossary entries*,
    not newly inferred entities or cross-glossary synonym clusters.
    """

    figure_id: str
    start: int
    end: int
    matched_text: str
    normalized_text: str
    term_id: str | None
    canonical_label: str | None
    candidate_term_ids: tuple[str, ...]
    match_kind: MatchKind
    resolution: Resolution

    @property
    def is_ambiguous(self) -> bool:
        return self.resolution == "ambiguous"


class ConceptNormalizer:
    """
    Normalize a caption's lexical matches against one frozen vocabulary.

    A ``Vocabulary`` can be reused across many figures. The input must be a
    single figure's extractor results, not mentions from multiple captions.
    """

    def __init__(self, vocabulary: Vocabulary) -> None:
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("vocabulary must be a Vocabulary")
        self._vocabulary = vocabulary

    def normalize(
        self, mentions: Iterable[ConceptMention], caption: str | None
    ) -> tuple[NormalizedMention, ...]:
        """
        Return normalized mentions in caption order, without dropping any.

        Fail closed on invalid evidence, unknown IDs, or inconsistent extractor
        decisions rather than producing fabricated figure-concept associations.
        A missing caption is valid only when there are no extracted mentions.
        """
        if caption is not None and not isinstance(caption, str):
            raise TypeError("caption must be a string or None")

        items = tuple(mentions)
        if not items:
            return ()
        if caption is None:
            raise ValueError("Cannot normalize mentions without their source caption")

        if not all(isinstance(item, ConceptMention) for item in items):
            raise TypeError("mentions must contain ConceptMention objects")
        ordered = sorted(items, key=lambda m: (m.start, m.end))
        self._validate_mentions(ordered, caption)

        result: list[NormalizedMention] = []
        for position, mention in enumerate(ordered):
            term_id = mention.term_id
            resolution: Resolution = "lexical" if term_id is not None else "ambiguous"
            if term_id is None:
                expanded = self._resolve_expansion(ordered, position, caption)
                if expanded is not None:
                    term_id = expanded
                    resolution = "explicit_expansion"

            term = self._vocabulary.get(term_id) if term_id is not None else None
            result.append(
                NormalizedMention(
                    figure_id=mention.figure_id,
                    start=mention.start,
                    end=mention.end,
                    matched_text=mention.matched_text,
                    normalized_text=mention.normalized_text,
                    term_id=term_id,
                    canonical_label=term.label if term is not None else None,
                    candidate_term_ids=mention.candidate_term_ids,
                    match_kind=mention.match_kind,
                    resolution=resolution,
                )
            )
        return tuple(result)

    def _validate_mentions(self, mentions: list[ConceptMention], caption: str) -> None:
        figure_id = mentions[0].figure_id
        previous_end = 0
        for mention in mentions:
            if not isinstance(figure_id, str) or not figure_id.strip() or mention.figure_id != figure_id:
                raise ValueError("All mentions must belong to the same nonempty figure_id")
            if (
                not isinstance(mention.start, int)
                or isinstance(mention.start, bool)
                or not isinstance(mention.end, int)
                or isinstance(mention.end, bool)
                or mention.start < previous_end
                or mention.start >= mention.end
                or mention.end > len(caption)
            ):
                raise ValueError("Mentions must have valid, non-overlapping caption spans")
            if caption[mention.start:mention.end] != mention.matched_text:
                raise ValueError("Mention evidence does not match the original caption")
            if not isinstance(mention.normalized_text, str) or not mention.normalized_text:
                raise ValueError("Mention is missing its normalized matching key")
            if mention.match_kind not in ("label", "alias", "mixed"):
                raise ValueError("Unsupported match kind")
            candidates = mention.candidate_term_ids
            if (
                not isinstance(candidates, tuple)
                or not candidates
                or len(candidates) != len(set(candidates))
            ):
                raise ValueError("Mention must list distinct candidate term IDs")
            if any(
                not isinstance(key, str) or self._vocabulary.get(key) is None
                for key in candidates
            ):
                raise ValueError("Mention references a term ID absent from the vocabulary")
            if len(candidates) == 1:
                if mention.term_id != candidates[0]:
                    raise ValueError("Single-candidate mention must have its term_id set")
            elif mention.term_id is not None:
                raise ValueError("Ambiguous mention must not select a term_id")
            previous_end = mention.end

    @staticmethod
    def _parenthetical_pair(
        left: ConceptMention, right: ConceptMention, caption: str
    ) -> bool:
        """
        Only a directly adjacent, closed parenthesis is expansion evidence
        """
        return bool(
            _OPEN_PAREN.fullmatch(caption[left.end:right.start])
            and _CLOSE_PAREN.match(caption[right.end:])
        )

    def _resolve_expansion(
        self, mentions: list[ConceptMention], position: int, caption: str
    ) -> str | None:
        """
        Choose a candidate only when the caption explicitly expands it.

        The neighboring match must be an unambiguous *full glossary headword*,
        not another short acronym or alias. This supports ``CO2 (carbon dioxide)``
        while refusing guesses based on WG membership or mere co-occurrence.
        """
        ambiguous = mentions[position]
        candidates = set(ambiguous.candidate_term_ids)
        supported: set[str] = set()

        for neighbor_index in (position - 1, position + 1):
            if not 0 <= neighbor_index < len(mentions):
                continue
            neighbor = mentions[neighbor_index]
            if neighbor.term_id is None or neighbor.term_id not in candidates:
                continue
            headword = self._vocabulary.get(neighbor.term_id)
            if (
                headword is None
                or len(headword.label) < 5
                or neighbor.match_kind not in ("label", "mixed")
                or normalize_term(neighbor.matched_text) != normalize_term(headword.label)
            ):
                continue
            left, right = sorted((ambiguous, neighbor), key=lambda m: m.start)
            if self._parenthetical_pair(left, right, caption):
                supported.add(neighbor.term_id)

        return next(iter(supported)) if len(supported) == 1 else None

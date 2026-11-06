"""
Extracts glossary concepts from IPCC figure captions

Matches terms and aliases, handles text variations, and keeps
their original positions in the caption
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import unicodedata

from concepts.vocabulary import Vocabulary, normalize_term

MatchKind = Literal["label", "alias", "mixed"]
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})

_SPACED_FORMULAS = {
    "CO2": ("CO 2",),
    "CH4": ("CH 4",),
    "N2O": ("N 2 O", "N 2O", "N2 O"),
    "O3": ("O 3",),
    "SF6": ("SF 6",),
}


@dataclass(frozen=True, slots=True)
class ConceptMention:
    """
    Store a concept mention with its position, match details, and possible terms
    """

    figure_id: str
    start: int
    end: int
    matched_text: str
    normalized_text: str
    term_id: str | None
    candidate_term_ids: tuple[str, ...]
    match_kind: MatchKind

    @property
    def is_ambiguous(self) -> bool:
        """
        Check whether the mention has no single matching concept
        """
        return self.term_id is None


@dataclass(slots=True)
class _TrieNode:
    """
    Store a character and its connections in the term matching trie
    """

    children: dict[str, _TrieNode] = field(default_factory=dict)
    terminal: str | None = None


@dataclass(frozen=True, slots=True)
class _Spelling:
    """
    Store the spelling variants and concept details for a normalized term
    """

    variants: tuple[tuple[str, str, MatchKind], ...]


def _normalized_caption(text: str) -> tuple[str, list[int], list[int]]:
    """
    Normalize caption text while keeping track of original character positions
    """
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cluster_start = 0

    def append_cluster(start: int, end: int) -> None:
        """
        Normalize a group of characters and record their original positions
        """
        transformed = (
            unicodedata.normalize("NFKC", text[start:end]).translate(_DASHES).casefold()
        )
        for char in transformed:
            if char.isspace() and characters and characters[-1] == " ":
                ends[-1] = end
            else:
                characters.append(" " if char.isspace() else char)
                starts.append(start)
                ends.append(end)

    for position in range(1, len(text)):
        if not unicodedata.category(text[position]).startswith("M"):
            append_cluster(cluster_start, position)
            cluster_start = position
    if text:
        append_cluster(cluster_start, len(text))
    return "".join(characters), starts, ends


def _word_character(char: str) -> bool:
    """
    Check whether a character belongs to a word
    """
    return char.isalnum() or char == "_" or unicodedata.category(char).startswith("M")


def _valid_boundaries(text: str, start: int, end: int) -> bool:
    """
    Check that a match does not start or end inside a word or number
    """
    if start and _word_character(text[start - 1]) and _word_character(text[start]):
        return False
    if (
        end < len(text)
        and _word_character(text[end - 1])
        and _word_character(text[end])
    ):
        return False
    if text[start : start + 1].isdigit() and start >= 2:
        if text[start - 1] == "." and text[start - 2].isdigit():
            return False
    if end < len(text) - 1 and text[end - 1 : end].isdigit():
        if text[end] == "." and text[end + 1].isdigit():
            return False
    return True


def _requires_exact_case(spelling: str) -> bool:
    """
    Check whether a short spelling needs an exact case match
    """
    return len(spelling.replace(" ", "")) <= 3 and any(c.isupper() for c in spelling)


def _normalized_case(text: str) -> str:
    """
    Normalize Unicode characters, dashes, and spaces without changing case
    """
    return " ".join(unicodedata.normalize("NFKC", text).translate(_DASHES).split())


class ConceptExtractor:
    """
    Find glossary terms and aliases in figure captions using a trie
    """

    def __init__(self, vocabulary: Vocabulary) -> None:
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("vocabulary must be a Vocabulary")
        root = _TrieNode()
        entries: dict[str, list[tuple[str, str, MatchKind]]] = {}
        for term in vocabulary.terms:
            for kind, spelling in [
                ("label", term.label),
                *(("alias", a) for a in term.aliases),
            ]:
                variants = (spelling, *_SPACED_FORMULAS.get(spelling, ()))
                for variant in variants:
                    key = normalize_term(variant)
                    if key:
                        entries.setdefault(key, []).append(
                            (variant, term.term_id, kind)
                        )

        spellings: dict[str, _Spelling] = {}
        for key, variants in entries.items():
            unique = sorted(set(variants), key=lambda item: (item[1], item[0], item[2]))
            spellings[key] = _Spelling(tuple(unique))
            node = root
            for char in key:
                node = node.children.setdefault(char, _TrieNode())
            node.terminal = key

        self._root = root
        self._spellings = spellings

    def _candidate(
        self, key: str, original: str
    ) -> tuple[tuple[str, ...], MatchKind] | None:
        entry = self._spellings[key]
        valid: list[tuple[str, MatchKind]] = []
        for spelling, term_id, kind in entry.variants:
            if not _requires_exact_case(spelling) or _normalized_case(
                original
            ) == _normalized_case(spelling):
                valid.append((term_id, kind))
        if not valid:
            return None
        ids = tuple(sorted({term_id for term_id, _ in valid}))
        kinds = {kind for _, kind in valid}
        kind: MatchKind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        return ids, kind

    def extract(
        self, figure_id: str, caption: str | None
    ) -> tuple[ConceptMention, ...]:
        if not isinstance(figure_id, str) or not figure_id.strip():
            raise ValueError("figure_id must be a nonempty string")
        if caption is None:
            return ()
        if not isinstance(caption, str):
            raise TypeError("caption must be a string or None")
        if not caption.strip():
            return ()

        normalized, starts, ends = _normalized_caption(caption)
        result: list[ConceptMention] = []
        position = 0
        while position < len(normalized):
            node = self._root
            cursor = position
            best: tuple[int, tuple[str, ...], MatchKind, str] | None = None
            while cursor < len(normalized):
                node = node.children.get(normalized[cursor])
                if node is None:
                    break
                cursor += 1
                key = node.terminal
                if key is None or not _valid_boundaries(normalized, position, cursor):
                    continue
                original = caption[starts[position] : ends[cursor - 1]]
                accepted = self._candidate(key, original)
                if accepted is not None:
                    ids, kind = accepted
                    best = (cursor, ids, kind, key)

            if best is None:
                position += 1
                continue

            stop, ids, kind, key = best
            start_original, end_original = starts[position], ends[stop - 1]
            result.append(
                ConceptMention(
                    figure_id=figure_id,
                    start=start_original,
                    end=end_original,
                    matched_text=caption[start_original:end_original],
                    normalized_text=key,
                    term_id=ids[0] if len(ids) == 1 else None,
                    candidate_term_ids=ids,
                    match_kind=kind,
                )
            )
            position = stop

        return tuple(result)

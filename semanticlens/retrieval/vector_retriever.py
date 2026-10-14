"""
Single stage vector retrieval for SemanticLens

Converts a text query into an embedding and delegates similarity search to a
VectorIndex. The embedder owns model inference; the index owns scoring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from semanticlens.core.exceptions import (
    ConfigurationError,
    EmbeddingCompatibilityError,
    RetrievalError,
    SemanticLensError,
)
from semanticlens.core.interfaces import Embedder, VectorIndex
from semanticlens.core.schemas import EmbeddingSpec, Query, RetrievalCandidate


class VectorRetriever:
    """
    Retrieve ranked evidence IDs using compatible embedder and index adapters

    Stateless: the pipeline supplies both components on each call. This class
    neither owns their resources nor modifies the index's scores or ranking.
    """

    def retrieve(
        self,
        query: Query,
        *,
        embedder: Embedder,
        index: VectorIndex,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Embed a query, search the index, and return ranked candidates"""
        if not isinstance(query, Query):
            raise TypeError("query must be a Query instance")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not callable(getattr(embedder, "embed_query", None)):
            raise ConfigurationError("embedder must implement embed_query()")
        if not callable(getattr(index, "search", None)):
            raise ConfigurationError("index must implement search()")

        try:
            embedder_spec = embedder.spec
            index_spec = index.spec
        except Exception as exc:
            raise ConfigurationError("Cannot read embedder or index specification") from exc

        if not isinstance(embedder_spec, EmbeddingSpec) or not isinstance(
            index_spec, EmbeddingSpec
        ):
            raise ConfigurationError("embedder and index must expose an EmbeddingSpec")
        if embedder_spec != index_spec:
            raise EmbeddingCompatibilityError(
                "Embedder and vector index use different embedding specifications"
            )

        try:
            query_embedding = embedder.embed_query(query)
        except SemanticLensError:
            raise
        except Exception as exc:
            raise RetrievalError("Failed to embed retrieval query") from exc

        try:
            matches = index.search(query_embedding, top_k=top_k)
        except SemanticLensError:
            raise
        except Exception as exc:
            raise RetrievalError("Failed to search vector index") from exc

        if (
            not isinstance(matches, Sequence)
            or isinstance(matches, (str, bytes))
            or len(matches) > top_k
        ):
            raise RetrievalError("Index returned an invalid number or type of matches")

        candidates: list[RetrievalCandidate] = []
        seen_ids: set[str] = set()

        for match in matches:
            if not isinstance(match, tuple) or len(match) != 2:
                raise RetrievalError("Index matches must be (unit_id, score) tuples")

            unit_id, score = match

            if not isinstance(unit_id, str) or not unit_id.strip():
                raise RetrievalError("Index returned an invalid evidence unit ID")
            if unit_id in seen_ids:
                raise RetrievalError("Index returned duplicate evidence unit IDs")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RetrievalError("Index returned a non-finite or invalid similarity score")

            seen_ids.add(unit_id)
            candidates.append(
                RetrievalCandidate(
                    unit_id=unit_id,
                    score=float(score),
                    retriever="vector",
                )
            )

        return candidates

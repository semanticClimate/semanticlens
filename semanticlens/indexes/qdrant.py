"""
Persistent Qdrant index for SemanticLens multivector embeddings

One Qdrant point stores one evidence unit's token vectors

Qdrant computes MaxSim scores; this adapter handles IDs, validation and index compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from semanticlens.core.exceptions import (
    ConfigurationError,
    EmbeddingCompatibilityError,
    IndexOperationError,
)
from semanticlens.core.interfaces import Embedding, IndexMatch
from semanticlens.core.schemas import EmbeddingSpec

from .config import QdrantConfig

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


_MANIFEST_VERSION = 1
_UNIT_ID_KEY = "semanticlens_unit_id"
_FLOAT32_MAX = 3.4028234663852886e38


class QdrantVectorIndex:
    """
    Store and search multivector evidence using local Qdrant

    The collection is created on first use and reopened on later runs

    Its embedding specification is saved in a separate manifest so a different
    model or preprocessing version cannot silently reuse incompatible vectors.
    """

    def __init__(
        self,
        *,
        spec: EmbeddingSpec,
        config: QdrantConfig | None = None,
    ) -> None:
        if not isinstance(spec, EmbeddingSpec):
            raise ConfigurationError("spec must be an EmbeddingSpec instance")
        if config is not None and not isinstance(config, QdrantConfig):
            raise ConfigurationError("config must be a QdrantConfig instance")
        if spec.representation != "multi_vector" or spec.interaction != "maxsim":
            raise ConfigurationError("Qdrant V1 requires multivector embeddings with MaxSim")

        self.config = config if config is not None else QdrantConfig()
        self._spec = spec
        self._closed = False
        self._client: QdrantClient | None = None

        try:
            storage_path = self.config.path.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError("Cannot resolve Qdrant storage path") from exc

        name_hash = hashlib.sha256(self.config.collection_name.encode("utf-8")).hexdigest()[:16]
        self._manifest_path = storage_path.parent / (
            f".{storage_path.name}.{name_hash}.semanticlens.json"
        )

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise IndexOperationError(
                'Qdrant is not installed. Install the optional "qdrant-client" dependency.'
            ) from exc

        try:
            self._client = QdrantClient(path=str(storage_path))
            self._open_or_create_collection()
        except (IndexOperationError, EmbeddingCompatibilityError):
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise IndexOperationError("Failed to initialize local Qdrant index") from exc

    @property
    def spec(self) -> EmbeddingSpec:
        """The embedding specification this collection accepts"""
        return self._spec

    def upsert(self, vectors: Mapping[str, Embedding]) -> None:
        """Insert or replace evidence embeddings using stable point IDs"""
        client = self._require_open()
        if not isinstance(vectors, Mapping):
            raise TypeError("vectors must map evidence unit IDs to embeddings")
        if not vectors:
            return

        from qdrant_client import models

        points = []
        for unit_id, embedding in vectors.items():
            if not isinstance(unit_id, str) or not unit_id.strip():
                raise ValueError("evidence unit IDs must be non-empty strings")
            matrix = self._validate_matrix(embedding)
            points.append(
                models.PointStruct(
                    id=self._point_id(unit_id),
                    vector=matrix,
                    payload={_UNIT_ID_KEY: unit_id},
                )
            )

        try:
            client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise IndexOperationError("Failed to upsert embeddings into Qdrant") from exc

    def search(self, query_embedding: Embedding, *, top_k: int) -> list[IndexMatch]:
        """Return the best matching evidence IDs and their MaxSim scores"""
        client = self._require_open()
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query = self._validate_matrix(query_embedding)

        try:
            response = client.query_points(
                collection_name=self.config.collection_name,
                query=query,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise IndexOperationError("Failed to search Qdrant index") from exc

        if len(response.points) > top_k:
            raise IndexOperationError("Qdrant returned more results than requested")

        results: list[IndexMatch] = []
        seen_ids: set[str] = set()
        for point in response.points:
            payload = point.payload
            unit_id = payload.get(_UNIT_ID_KEY) if isinstance(payload, Mapping) else None
            if not isinstance(unit_id, str) or not unit_id.strip():
                raise IndexOperationError("Qdrant result has no valid evidence unit ID")
            if str(point.id) != self._point_id(unit_id):
                raise IndexOperationError("Qdrant result ID does not match its evidence ID")
            if unit_id in seen_ids:
                raise IndexOperationError("Qdrant returned duplicate evidence IDs")
            seen_ids.add(unit_id)
            try:
                if isinstance(point.score, bool):
                    raise ValueError("Boolean score")
                score = float(point.score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise IndexOperationError("Qdrant returned an invalid similarity score") from exc
            if not math.isfinite(score):
                raise IndexOperationError("Qdrant returned a non-finite similarity score")
            results.append((unit_id, score))
        return results

    def close(self) -> None:
        """Release the local Qdrant lock; calling twice is safe"""
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                raise IndexOperationError("Failed to close the Qdrant client") from exc

    @staticmethod
    def _point_id(unit_id: str) -> str:
        """Map an arbitrary evidence ID to the same Qdrant UUID every time"""
        return str(uuid5(NAMESPACE_URL, f"semanticlens:evidence:{unit_id}"))

    def _require_open(self) -> QdrantClient:
        if self._closed or self._client is None:
            raise IndexOperationError("This Qdrant index is closed")
        return self._client

    def _validate_matrix(self, embedding: Embedding) -> list[list[float]]:
        """Validate a non-empty token matrix with the configured dimension."""
        if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)):
            raise ValueError("embedding must be a non-empty sequence of token vectors")
        if not embedding:
            raise ValueError("embedding must have at least one token vector")

        matrix: list[list[float]] = []
        for row in embedding:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise ValueError("each token vector must be a numeric sequence")
            if len(row) != self.spec.dimension:
                raise ValueError(
                    f"token vectors must have dimension {self.spec.dimension}"
                )
            values: list[float] = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("embedding values must be real numbers")
                number = float(value)
                if not math.isfinite(number) or abs(number) > _FLOAT32_MAX:
                    raise ValueError("embedding values must be finite float32 numbers")
                values.append(number)
            matrix.append(values)
        return matrix

    def _open_or_create_collection(self) -> None:
        client = self._require_open()
        collection_name = self.config.collection_name
        try:
            exists = client.collection_exists(collection_name)
        except Exception as exc:
            raise IndexOperationError("Cannot inspect Qdrant collection") from exc

        has_manifest = self._manifest_path.is_file()
        if exists and not has_manifest:
            raise IndexOperationError(
                "Qdrant collection exists without a SemanticLens compatibility record; "
                "choose another collection or restore its record"
            )
        if has_manifest and not exists:
            raise IndexOperationError(
                "SemanticLens compatibility record exists but its collection is missing"
            )

        if exists:
            self._read_and_check_manifest()
            self._check_collection_settings()
            return

        from qdrant_client import models

        distance = (
            models.Distance.DOT if self.spec.similarity == "dot" else models.Distance.COSINE
        )
        try:
            created = client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.spec.dimension,
                    distance=distance,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            )
            if not created:
                raise IndexOperationError("Qdrant did not create the requested collection")
        except IndexOperationError:
            raise
        except Exception as exc:
            raise IndexOperationError("Failed to create Qdrant collection") from exc

        try:
            self._write_manifest()
        except IndexOperationError:
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass 
            raise

    def _check_collection_settings(self) -> None:
        from qdrant_client import models

        try:
            info = self._require_open().get_collection(self.config.collection_name)
            params = info.config.params.vectors
            distance = (
                models.Distance.DOT if self.spec.similarity == "dot" else models.Distance.COSINE
            )
            if (
                not isinstance(params, models.VectorParams)
                or params.size != self.spec.dimension
                or params.distance != distance
                or params.multivector_config is None
                or params.multivector_config.comparator
                != models.MultiVectorComparator.MAX_SIM
            ):
                raise EmbeddingCompatibilityError(
                    "Qdrant collection vector settings do not match the embedding specification"
                )
        except EmbeddingCompatibilityError:
            raise
        except Exception as exc:
            raise IndexOperationError("Cannot validate Qdrant collection settings") from exc

    def _read_and_check_manifest(self) -> None:
        try:
            with self._manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if (
                not isinstance(manifest, dict)
                or manifest.get("format_version") != _MANIFEST_VERSION
                or manifest.get("collection_name") != self.config.collection_name
            ):
                raise ValueError("Unsupported or invalid SemanticLens index record")
            saved_spec = EmbeddingSpec.model_validate(manifest["embedding_spec"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexOperationError("Invalid SemanticLens index compatibility record") from exc
        if saved_spec != self.spec:
            raise EmbeddingCompatibilityError(
                "Existing Qdrant index uses a different embedding model or preprocessing"
            )

    def _write_manifest(self) -> None:
        """Atomically save the embedding contract for safe index reopening"""
        manifest = {
            "format_version": _MANIFEST_VERSION,
            "collection_name": self.config.collection_name,
            "embedding_spec": self.spec.model_dump(mode="json"),
        }
        temporary: Path | None = None
        try:
            self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._manifest_path.parent,
                prefix=f".{self._manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._manifest_path)
        except OSError as exc:
            raise IndexOperationError("Failed to save Qdrant index compatibility record") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

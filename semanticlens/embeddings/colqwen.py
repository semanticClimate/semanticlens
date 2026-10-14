"""
ColQwen2 multivector embeddings for visual document retrieval

This adapter implements Core's Embedder contract using Sentence Transformers.

It encodes page images and text queries without changing the model's native
preprocessing, token representations or late interaction scoring behavior
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from semanticlens.core.exceptions import ConfigurationError, EmbeddingError
from semanticlens.core.schemas import EmbeddingSpec, EvidenceUnit, Query

from .config import ColQwenConfig

if TYPE_CHECKING:
    from PIL import Image
    from sentence_transformers import MultiVectorEncoder


_EMBEDDING_DIMENSION = 128
_PREPROCESSING_ID = "colqwen2-v1.0-hf:sentence-transformers-6-native-v1"


class ColQwenEmbedder:
    """
    Encode page images and text queries with the pinned ColQwen2 checkpoint

    The workspace must match the Documents module's workspace because
    EvidenceUnit.image_ref contains a workspace relative PNG path

    The model is loaded only when an embedding is first requested. Outputs are
    lists of token vectors so Core and vector indexes need no PyTorch dependency.
    One instance should be used sequentially, not for concurrent inference.
    """

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: ColQwenConfig | None = None,
    ) -> None:
        if not isinstance(workspace, (str, Path)) or (
            isinstance(workspace, str) and not workspace.strip()
        ):
            raise ConfigurationError("workspace must be a non-empty path")
        if config is not None and not isinstance(config, ColQwenConfig):
            raise ConfigurationError("config must be a ColQwenConfig instance")

        try:
            self.workspace = Path(workspace).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError("Cannot resolve embedding workspace") from exc

        self.config = config if config is not None else ColQwenConfig()
        self._spec = EmbeddingSpec(
            model_id=self.config.model_id,
            model_revision=self.config.revision,
            preprocessing_id=_PREPROCESSING_ID,
            dimension=_EMBEDDING_DIMENSION,
            representation="multi_vector",
            interaction="maxsim",
            similarity="dot",
        )
        self._model: MultiVectorEncoder | None = None
        self._closed = False

    @property
    def spec(self) -> EmbeddingSpec:
        """
        Describe the embeddings required by a compatible vector index
        """
        return self._spec

    def embed_units(self, units: Sequence[EvidenceUnit]) -> list[list[list[float]]]:
        """
        Embed images in input order returning one token matrix per unit
        """
        self._require_open()
        if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
            raise TypeError("units must be a sequence of EvidenceUnit objects")
        if not units:
            return []

        paths = [self._image_path(unit) for unit in units]
        model = self._get_model()
        embeddings: list[list[list[float]]] = []
        size = self.config.inference_batch_size

        for start in range(0, len(paths), size):
            images: list[Image.Image] = []
            try:
                for path in paths[start : start + size]:
                    images.append(self._load_image(path))

                outputs = model.encode_document(
                    images,
                    batch_size=size,
                    show_progress_bar=False,
                )
                embeddings.extend(self._convert_batch(outputs, len(images), "page"))
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError("Failed to encode PDF page images") from exc
            finally:
                for image in images:
                    image.close()

        return embeddings

    def embed_query(self, query: Query) -> list[list[float]]:
        """
        Embed one text query using ColQwen's query specific processing
        """
        self._require_open()
        if not isinstance(query, Query):
            raise TypeError("query must be a Query instance")

        model = self._get_model()
        try:
            outputs = model.encode_query(
                [query.text],
                batch_size=1,
                show_progress_bar=False,
            )
            return self._convert_batch(outputs, 1, "query")[0]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to encode search query") from exc

    def close(self) -> None:
        """
        Release the model reference; safe to call more than once
        """
        if self._closed:
            return
        self._closed = True
        self._model = None

    def _require_open(self) -> None:
        if self._closed:
            raise EmbeddingError("This ColQwen embedder has been closed")

    def _get_model(self) -> MultiVectorEncoder:
        """
        Load the model once without changing its default processor
        """
        if self._model is not None:
            return self._model

        try:
            import torch
            from sentence_transformers import MultiVectorEncoder
        except ImportError as exc:
            raise EmbeddingError(
                'ColQwen requires PyTorch and sentence-transformers. '
                'Install the optional embedding dependencies.'
            ) from exc

        device = self.config.device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ConfigurationError("CUDA was requested but is unavailable")
            if ":" in device and int(device.split(":", 1)[1]) >= torch.cuda.device_count():
                raise ConfigurationError(f"CUDA device does not exist: {device}")
        elif device == "mps" and not torch.backends.mps.is_available():
            raise ConfigurationError("MPS was requested but is unavailable")

        model_kwargs: dict[str, Any] = {}
        if self.config.dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self.config.dtype)

        try:
            model = MultiVectorEncoder(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                device=None if device == "auto" else device,
                trust_remote_code=False,
                model_kwargs=model_kwargs or None,
            )
            dimension = model.get_embedding_dimension()
            if dimension is not None and dimension != self.spec.dimension:
                raise EmbeddingError(
                    f"ColQwen dimension mismatch: expected {self.spec.dimension}, "
                    f"got {dimension}"
                )
            model.eval()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load ColQwen checkpoint {self.config.model_id!r} "
                f"at revision {self.config.revision}"
            ) from exc

        self._model = model
        return model

    def _image_path(self, unit: EvidenceUnit) -> Path:
        """
        Resolve a document image strictly within the configured workspace
        """
        if not isinstance(unit, EvidenceUnit):
            raise TypeError("units must contain EvidenceUnit objects")
        if not unit.image_ref:
            raise EmbeddingError(f"Evidence unit {unit.unit_id!r} has no image_ref")

        relative = Path(unit.image_ref)
        if relative.is_absolute() or ".." in relative.parts:
            raise EmbeddingError(f"Invalid image_ref for evidence unit {unit.unit_id!r}")
        try:
            path = (self.workspace / relative).resolve(strict=True)
            if not path.is_relative_to(self.workspace) or not path.is_file():
                raise EmbeddingError(
                    f"Image for evidence unit {unit.unit_id!r} is not a workspace file"
                )
        except (OSError, RuntimeError) as exc:
            raise EmbeddingError(
                f"Cannot access image for evidence unit {unit.unit_id!r}"
            ) from exc
        return path

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        """
        Read a complete RGB PNG into an image owned by this adapter
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise EmbeddingError("Install Pillow to encode PDF page images") from exc

        try:
            with Image.open(path) as source:
                if source.format != "PNG" or source.mode != "RGB":
                    raise EmbeddingError(f"Expected an RGB PNG page image: {path}")
                source.load()
                return source.copy()
        except EmbeddingError:
            raise
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise EmbeddingError(f"Cannot read page image: {path}") from exc

    def _convert_batch(
        self, output: Any, expected_count: int, context: str
    ) -> list[list[list[float]]]:
        """
        Check output dimensions and convert tensors to portable float lists
        """
        try:
            import torch
        except ImportError as exc:
            raise EmbeddingError("PyTorch is required to handle ColQwen outputs") from exc

        if not isinstance(output, (list, tuple)) or len(output) != expected_count:
            raise EmbeddingError(
                f"ColQwen returned an invalid number of {context} embeddings"
            )

        result: list[list[list[float]]] = []
        for tensor in output:
            if not isinstance(tensor, torch.Tensor):
                raise EmbeddingError(f"ColQwen returned a non-tensor {context} embedding")
            if (
                tensor.ndim != 2
                or tensor.shape[0] < 1
                or tensor.shape[1] != self.spec.dimension
            ):
                raise EmbeddingError(
                    f"ColQwen returned an invalid {context} embedding shape: "
                    f"{tuple(tensor.shape)}"
                )
            try:
                values = tensor.detach().to(device="cpu", dtype=torch.float32)
                if not bool(torch.isfinite(values).all()):
                    raise EmbeddingError(
                        f"ColQwen returned non-finite {context} embedding values"
                    )
                result.append(values.tolist())
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not convert {context} embeddings to float lists"
                ) from exc
        return result

"""Qwen3-VL visual answer-generation adapter for SemanticLens.

Implements Core's Generator protocol with Hugging Face Transformers.
Retrieved evidence remains in its original order; the model sees page images
and short labels, never internal evidence IDs. The returned GenerationDraft is
still untrusted and must be validated by the grounding stage.

Model weights are loaded lazily. One instance serializes inference calls and
must not be used after close(). No model-specific imports occur at module load.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from semanticlens.core.exceptions import ConfigurationError, GenerationError
from semanticlens.core.schemas import (
    GeneratedStatement,
    GenerationDraft,
    Query,
    SearchHit,
)

from .config import Qwen3VLConfig
from .prompting import ImagePart, TextPart, build_generation_prompt

if TYPE_CHECKING:
    from PIL import Image


# Resource-safety guard on the *decoded source*, independent of the configured
# pixel budget applied by the Qwen processor for model input.
_MAX_SOURCE_PIXELS = 80_000_000


class Qwen3VLGenerator:
    """Generate evidence-referenced answer drafts with a Qwen3-VL model.

    ``workspace`` must be the same directory used by PDFPageProcessor, since
    ``EvidenceUnit.image_ref`` is a workspace-relative file path.

    The model and processor are loaded on the first nonempty generation call.
    ``device='auto'`` chooses CUDA, then MPS, then CPU. ``dtype='auto'`` uses
    BF16 on supported CUDA hardware, FP16 on other GPUs, and FP32 on CPU.
    Other model families require their own Generator implementation.
    """

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: Qwen3VLConfig | None = None,
    ) -> None:
        if not isinstance(workspace, (str, Path)) or (
            isinstance(workspace, str) and not workspace.strip()
        ):
            raise ConfigurationError("workspace must be a non-empty path")
        if config is not None and not isinstance(config, Qwen3VLConfig):
            raise ConfigurationError("config must be a Qwen3VLConfig instance")

        try:
            self.workspace = Path(workspace).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError("Cannot resolve generation workspace") from exc

        self.config = config if config is not None else Qwen3VLConfig()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._closed = False
        self._lock = threading.RLock()

    def generate(self, query: Query, *, hits: Sequence[SearchHit]) -> GenerationDraft:
        """Answer from all supplied pages, returning unverified evidence IDs.

        For zero hits, return insufficient evidence without loading a model.
        Unsupported inputs, missing images, or too many pages are errors; no
        evidence pages are silently omitted. Malformed model output raises
        GenerationError rather than creating an ungrounded answer.
        """
        with self._lock:
            self._require_open()
            if not isinstance(query, Query):
                raise TypeError("query must be a Query instance")
            if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
                raise TypeError("hits must be a sequence of SearchHit objects")
            if any(not isinstance(hit, SearchHit) for hit in hits):
                raise TypeError("hits must contain only SearchHit objects")
            if not hits:
                return GenerationDraft(
                    status="insufficient_evidence",
                    statements=(
                        GeneratedStatement(
                            text="No document evidence was retrieved to answer this question.",
                            evidence_ids=(),
                        ),
                    ),
                    model_id=self.config.model_id,
                    model_revision=self.config.revision,
                )

            prompt = build_generation_prompt(
                query, hits, max_evidence_pages=self.config.max_evidence_pages
            )

            # Open and validate images before allocating model weights. Keep
            # image objects alive until the processor finishes encoding them.
            with ExitStack() as stack:
                user_content: list[dict[str, Any]] = []
                for part in prompt.user_parts:
                    if isinstance(part, TextPart):
                        user_content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImagePart):
                        image = self._open_image(part.image_ref, stack)
                        user_content.append({"type": "image", "image": image})
                    else:
                        raise GenerationError("Unsupported multimodal prompt segment")

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": prompt.system_text}]},
                    {"role": "user", "content": user_content},
                ]
                raw_output = self._infer(messages)

            return _parse_generation_output(
                raw_output,
                label_to_unit_id=prompt.label_to_unit_id,
                model_id=self.config.model_id,
                model_revision=self.config.revision,
            )

    def close(self) -> None:
        """Release owned model references. Closing twice is harmless."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._model = None
            self._processor = None

    def _require_open(self) -> None:
        if self._closed:
            raise GenerationError("This Qwen3-VL generator has been closed")

    def _open_image(self, image_ref: str, stack: ExitStack) -> Image.Image:
        """Decode a workspace-contained image without fetching remote content."""
        from PIL import Image, UnidentifiedImageError

        relative = Path(image_ref)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise GenerationError("Evidence image reference must be a workspace-relative path")
        try:
            path = (self.workspace / relative).resolve(strict=True)
            if not path.is_relative_to(self.workspace) or not path.is_file():
                raise GenerationError("Evidence image is outside the workspace or not a file")
            if (self.workspace / relative).is_symlink():
                raise GenerationError("Symlinked evidence images are not supported")

            # Convert to a standalone RGB image; the originating file handle
            # closes immediately and the RGB copy is closed after inference.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as source:
                    if source.width <= 0 or source.height <= 0:
                        raise GenerationError("Evidence image has invalid dimensions")
                    if source.width * source.height > _MAX_SOURCE_PIXELS:
                        raise GenerationError("Evidence image exceeds the decoded-pixel safety limit")
                    source.load()
                    image = source.convert("RGB")
            stack.callback(image.close)
            return image
        except GenerationError:
            raise
        except (
            OSError,
            RuntimeError,
            ValueError,
            SyntaxError,
            UnidentifiedImageError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ) as exc:
            raise GenerationError(f"Cannot read evidence image: {image_ref!r}") from exc

    def _infer(self, messages: list[dict[str, Any]]) -> str:
        """Run one multimodal chat completion and decode only new tokens."""
        model, processor, torch = self._get_runtime()
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
                raise GenerationError("Qwen3-VL processor did not return tokenized inputs")

            # Some processor versions supply token_type_ids that are not
            # accepted by the Qwen3-VL model's generation signature.
            inputs.pop("token_type_ids", None)
            prompt_length = inputs["input_ids"].shape[-1]
            inputs = inputs.to(model.device)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
            }
            if self.config.do_sample:
                generation_kwargs["temperature"] = self.config.temperature
                generation_kwargs["top_p"] = self.config.top_p

            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
            completion_ids = output_ids[0][prompt_length:]
            decoded = processor.batch_decode(
                [completion_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(
                decoded[0], str
            ):
                raise GenerationError("Qwen3-VL processor returned an invalid decoded answer")
            return decoded[0]
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError("Qwen3-VL multimodal inference failed") from exc

    def _get_runtime(self) -> tuple[Any, Any, Any]:
        """Load the model and processor once, using checked device settings."""
        if self._model is not None and self._processor is not None:
            import torch

            return self._model, self._processor, torch

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise GenerationError(
                "Qwen3-VL needs PyTorch and a Transformers version with "
                "Qwen3VLForConditionalGeneration support"
            ) from exc

        device = _resolve_device(self.config.device, torch)
        dtype = _resolve_dtype(self.config.dtype, device, torch)
        if self.config.attn_implementation == "flash_attention_2":
            if not device.startswith("cuda") or dtype == torch.float32:
                raise ConfigurationError(
                    "flash_attention_2 requires CUDA and float16 or bfloat16"
                )
            if torch.cuda.get_device_capability(device)[0] < 8:
                raise ConfigurationError(
                    "flash_attention_2 requires an Ampere-generation or newer CUDA GPU"
                )

        loading_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": self.config.attn_implementation,
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": False,
        }
        if self.config.revision is not None:
            loading_kwargs["revision"] = self.config.revision
        if self.config.device_map is not None:
            loading_kwargs["device_map"] = self.config.device_map

        processor_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": False,
        }
        if self.config.revision is not None:
            processor_kwargs["revision"] = self.config.revision

        try:
            processor = AutoProcessor.from_pretrained(
                self.config.model_id, **processor_kwargs
            )
            image_processor = processor.image_processor
            image_processor.size = {
                "shortest_edge": self.config.min_image_pixels,
                "longest_edge": self.config.max_image_pixels,
            }
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_id, **loading_kwargs
            )
            if self.config.device_map is None:
                model = model.to(device)
            model.eval()
        except Exception as exc:
            raise GenerationError("Failed to load the Qwen3-VL model or processor") from exc

        self._processor = processor
        self._model = model
        return model, processor, torch


def _resolve_device(requested: str, torch: Any) -> str:
    """Resolve and check a single runtime device without choosing a fixed GPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is unavailable")
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise ConfigurationError(f"CUDA device does not exist: {requested}")
    elif requested == "mps" and not torch.backends.mps.is_available():
        raise ConfigurationError("MPS was requested but is unavailable")
    return requested


def _resolve_dtype(requested: str, device: str, torch: Any) -> Any:
    """Select a supported default weight precision; never alter explicit choices."""
    if requested == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if _cuda_bf16_supported(device, torch) else torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    dtype = getattr(torch, requested)
    if device == "cpu" and requested == "float16":
        raise ConfigurationError("float16 CPU inference is unsupported by this adapter")
    if device == "mps" and requested == "bfloat16":
        raise ConfigurationError("bfloat16 MPS inference is unsupported by this adapter")
    if device.startswith("cuda") and requested == "bfloat16":
        if not _cuda_bf16_supported(device, torch):
            raise ConfigurationError("This CUDA GPU does not support bfloat16")
    return dtype


def _cuda_bf16_supported(device: str, torch: Any) -> bool:
    """Check BF16 on the selected GPU, not necessarily CUDA device zero."""
    with torch.cuda.device(device):
        return bool(torch.cuda.is_bf16_supported())


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def _parse_generation_output(
    text: str,
    *,
    label_to_unit_id: Mapping[str, str],
    model_id: str,
    model_revision: str | None,
) -> GenerationDraft:
    """Validate a model-produced JSON object and resolve prompt-local labels.

    Unknown labels are never converted to real source citations. Only the
    later grounding stage can validate those IDs against the original hits.
    """
    try:
        payload = json.loads(
            text.strip(),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
        if not isinstance(payload, dict) or set(payload) != {"status", "statements"}:
            raise ValueError("Expected only 'status' and 'statements' JSON fields")
        status = payload["status"]
        statements = payload["statements"]
        if status not in ("answered", "insufficient_evidence") or not isinstance(
            statements, list
        ) or not statements:
            raise ValueError("Invalid generation status or empty statements")
        if status == "insufficient_evidence" and len(statements) != 1:
            raise ValueError("Insufficient evidence requires exactly one explanation")

        validated: list[GeneratedStatement] = []
        for statement in statements:
            if not isinstance(statement, dict) or set(statement) != {"text", "evidence_ids"}:
                raise ValueError("Each statement needs only 'text' and 'evidence_ids'")
            if not isinstance(statement["text"], str) or not isinstance(
                statement["evidence_ids"], list
            ):
                raise ValueError("Statement text and evidence IDs have invalid types")
            labels = statement["evidence_ids"]
            if any(not isinstance(label, str) for label in labels):
                raise ValueError("Evidence labels must be strings")
            if status == "insufficient_evidence" and labels:
                raise ValueError("Insufficient-evidence explanations cannot cite evidence")
            unit_ids = tuple(label_to_unit_id[label] for label in labels)
            validated.append(
                GeneratedStatement(text=statement["text"], evidence_ids=unit_ids)
            )
        return GenerationDraft(
            status=status,
            statements=tuple(validated),
            model_id=model_id,
            model_revision=model_revision,
        )
    except (ValueError, TypeError, KeyError, ValidationError) as exc:
        raise GenerationError("Qwen3-VL returned invalid or ungrounded JSON output") from exc

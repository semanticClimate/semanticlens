"""Configuration for the Qwen3-VL answer-generation adapter.

Configuration is validated without importing or loading inference libraries.
Hardware support and model compatibility are checked by the adapter at runtime.
Other generator backends may define their own configurations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from semanticlens.core.schemas import CoreSchema


class Qwen3VLConfig(CoreSchema):
    """Settings for visual question answering with a Qwen3-VL checkpoint.

    The default checkpoint is a reference model, not a requirement for Core.
    Other Qwen3-VL checkpoints may be selected with ``model_id``; unrelated
    model architectures should use a different Generator implementation.

    Loading is handled lazily by the adapter. Constructing this configuration
    does not download a model or require an available GPU.
    """

    model_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        min_length=1,
        description="Hugging Face model ID or local path to a compatible Qwen3-VL checkpoint.",
    )
    revision: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional Hugging Face revision. Pin a commit SHA for reproducible runs; "
            "None uses the checkpoint's current default revision."
        ),
    )
    local_files_only: bool = Field(
        default=False,
        description="Load model and processor files only from the local cache or path.",
    )

    device: str = Field(
        default="auto",
        pattern=r"^(?:auto|cpu|mps|cuda(?::(?:0|[1-9][0-9]*))?)$",
        description=(
            "Preferred inference device: auto, cpu, mps, cuda, or cuda:<index>. "
            "The adapter resolves 'auto' using the available hardware."
        ),
    )
    device_map: Literal["auto"] | None = Field(
        default=None,
        description=(
            "Set to 'auto' for Accelerate-managed placement across devices. "
            "Requires device='auto'; None uses a single resolved device."
        ),
    )
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = Field(
        default="auto",
        description=(
            "Model weight precision. The adapter chooses a device-supported dtype "
            "for 'auto' rather than blindly trusting the checkpoint default."
        ),
    )
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = Field(
        default="sdpa",
        description=(
            "Attention implementation. FlashAttention-2 requires separately installed "
            "dependencies and compatible GPU hardware."
        ),
    )

    max_evidence_pages: int = Field(
        default=5,
        gt=0,
        description=(
            "Maximum number of retrieved page images accepted in one generation call. "
            "The adapter must report an over-budget input rather than silently drop pages."
        ),
    )
    min_image_pixels: int = Field(
        default=256 * 32 * 32,
        gt=0,
        description=(
            "Minimum pixels per image for Qwen3-VL's image processor "
            "(size['shortest_edge']); smaller images may be upscaled."
        ),
    )
    max_image_pixels: int = Field(
        default=2048 * 32 * 32,
        gt=0,
        description=(
            "Maximum pixels per image for Qwen3-VL's image processor "
            "(size['longest_edge']); increase for dense charts or reduce for memory savings."
        ),
    )

    max_new_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum number of tokens generated for one answer, excluding the prompt.",
    )
    do_sample: bool = Field(
        default=False,
        description="Use sampling instead of deterministic greedy decoding.",
    )
    temperature: float | None = Field(
        default=None,
        gt=0,
        le=2,
        description="Sampling temperature; required when do_sample=True.",
    )
    top_p: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Nucleus sampling threshold; required when do_sample=True.",
    )

    @field_validator("model_id", "revision")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        """Reject whitespace-only model identifiers and revisions."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("model_id and revision must not be blank")
        return value

    @model_validator(mode="after")
    def validate_combined_settings(self) -> Qwen3VLConfig:
        """Reject ambiguous hardware, image and decoding configurations."""
        if self.device_map is not None and self.device != "auto":
            raise ValueError("device_map='auto' requires device='auto'")

        if self.min_image_pixels > self.max_image_pixels:
            raise ValueError("min_image_pixels must not exceed max_image_pixels")

        if self.do_sample:
            if self.temperature is None or self.top_p is None:
                raise ValueError(
                    "temperature and top_p are required when do_sample=True"
                )
        elif self.temperature is not None or self.top_p is not None:
            raise ValueError(
                "temperature and top_p must be None when do_sample=False"
            )

        return self

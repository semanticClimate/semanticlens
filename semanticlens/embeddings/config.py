"""
Model loading settings for the SemanticLens ColQwen2 embedding adapter

These settings control how the model runs
"""

from __future__ import annotations
from typing import Literal
from pydantic import Field

from semanticlens.core.schemas import CoreSchema


class ColQwenConfig(CoreSchema):
    """
    Settings for the ColQwen2 model used in the first VDR pipeline
    """

    model_id: Literal["vidore/colqwen2-v1.0-hf"] = Field(
        default="vidore/colqwen2-v1.0-hf",
        description="Supported ColQwen2 checkpoint for the V1 adapter.",
    )
    revision: str = Field(
        default="ddc07d2317c80f75fc742b7362ee9ad1912908f9",
        pattern=r"^[0-9a-f]{40}$",
        description="Exact Hugging Face commit SHA used to load the checkpoint.",
    )
    device: str = Field(
        default="auto",
        pattern=r"^(?:auto|cpu|mps|cuda(?::(?:0|[1-9][0-9]*))?)$",
        description="Inference device: auto, cpu, mps, cuda, or cuda:<index>.",
    )
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = Field(
        default="auto",
        description=(
            "Model weight precision. 'auto' uses the checkpoint's default; "
            "explicit dtypes must be supported by the selected device."
        ),
    )
    inference_batch_size: int = Field(
        default=1,
        gt=0,
        description=(
            "Maximum number of pages encoded in one model forward batch. "
            "Independent of Core's indexing batch size."
        ),
    )
    local_files_only: bool = Field(
        default=False,
        description="Only load the model from existing local cache files.",
    )

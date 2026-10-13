"""
Shared configuration for the SemanticLens pipeline

Only settings used across pipeline components belong here

Embedding models, PDF processors and vector databases manage their own specific settings
"""

from __future__ import annotations
from pathlib import Path
from pydantic import Field, ValidationInfo, field_validator
from .schemas import CoreSchema


class PipelineConfig(CoreSchema):
    """
    Settings for indexing documents and searching for evidence

    The workspace is a location for persistent data not a directory created by this class
    The pipeline decides when to create or open it
    """

    workspace: Path = Field(
        default=Path("./lens_data"),
        description="Directory for indexes, page images, and evidence metadata.",
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        description="Maximum number of evidence units sent to the embedder per batch.",
    )
    default_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of search results to return when top_k is not specified.",
    )

    @field_validator("workspace", mode="before")
    @classmethod
    def accept_workspace_path(cls, value: object, info: ValidationInfo) -> object:
        """
        Accept a Path or a non empty path string
        """
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("workspace must not be empty")
            return value if info.mode == "json" else Path(value)
        return value

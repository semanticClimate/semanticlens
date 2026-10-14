"""
Configuration for the Qdrant index used by SemanticLens.

V1 uses Qdrant's local, persistent storage.
"""

from __future__ import annotations
from pathlib import Path
from pydantic import Field, ValidationInfo, field_validator

from semanticlens.core.schemas import CoreSchema


class QdrantConfig(CoreSchema):
    """
    Settings for a local Qdrant collection
    """

    path: Path = Field(
        default=Path("./lens_data/qdrant"),
        description="Directory where local Qdrant stores its data.",
    )
    collection_name: str = Field(
        default="semanticlens_pages",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description=(
            "Name of the Qdrant collection. Use ASCII letters, digits, "
            "underscores, hyphens, or periods; start with a letter or digit."
        ),
    )

    @field_validator("path", mode="before")
    @classmethod
    def accept_storage_path(cls, value: object, info: ValidationInfo) -> object:
        """
        Accept a Path or a non empty string without touching the filesystem
        """
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("path must not be empty")
            if "\x00" in value:
                raise ValueError("path must not contain null characters")
            return value if info.mode == "json" else Path(value)
        return value

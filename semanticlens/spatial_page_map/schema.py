from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, validate_assignment=True
    )


class Document(_Model):
    filename: str | None = None
    sha256: str | None = None
    page_count: int | None = Field(default=None, ge=0)
    properties: dict[str, JsonValue] | None = None


class Parser(_Model):
    name: str | None = None
    version: str | None = None
    tier: str | None = None
    mode: str | None = None
    source_schema: str | None = None
    source_schema_version: str | None = None
    extensions: dict[str, JsonValue] | None = None


class Region(_Model):
    id: str
    parent_id: str | None = None
    type: str | None = None
    reading_order: int | None = Field(default=None, ge=0)
    source_index: int | None = Field(default=None, ge=0)
    bbox: list[Annotated[float, Field(ge=0, le=1)]] | None = Field(
        default=None, 
        min_length=4, 
        max_length=4
    )
    text: str | None = None
    html: str | None = None
    latex: str | None = None
    spans: list[dict[str, JsonValue]] | None = None
    heading_level: int | None = Field(default=None, ge=1)
    continues_previous: bool | None = None
    image: dict[str, JsonValue] | None = None
    attributes: dict[str, JsonValue] | None = None

    @field_validator("bbox")
    @classmethod
    def _ordered_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and (value[2] <= value[0] or value[3] <= value[1]):
            raise ValueError("bbox must satisfy x1 > x0 and y1 > y0")
        return value


class Page(_Model):
    page_idx: int = Field(ge=0)
    width_pt: float | None = Field(default=None, gt=0)
    height_pt: float | None = Field(default=None, gt=0)
    regions: list[Region]


class SpatialPageMap(_Model):
    schema_version: Literal["1.0"] = "1.0"
    coordinate_system: Literal["normalized_top_left_xyxy"] = "normalized_top_left_xyxy"
    is_full_document: bool
    document: Document | None = None
    parser: Parser | None = None
    pages: list[Page]

FIELD_PATHS = frozenset(SpatialPageMap.model_fields) | frozenset(
    f"{prefix}.{name}"
    for prefix, model in (
        ("document", Document),
        ("parser", Parser),
        ("pages", Page),
        ("pages.regions", Region),
    )
    for name in model.model_fields
)
REQUIRED_PATHS = frozenset(
    {
        "schema_version",
        "coordinate_system",
        "is_full_document",
        "pages.page_idx",
        "pages.regions.id",
        "pages.regions.parent_id",
    }
)

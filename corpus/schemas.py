"""
Defines the data model for IPCC AR6 figures

Validates figure details, image links, timestamps and source
information to keep the collected data consistent
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WorkingGroup = Literal["WG1", "WG2", "WG3"]


class FigureRecord(BaseModel):
    """
    Store and validate the details of an IPCC figure
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.0"]
    figure_id: str = Field(min_length=1)
    working_group: WorkingGroup
    figure_label: str | None
    title: str | None
    chapter: str | None
    section: str = Field(min_length=1)
    caption: str | None
    caption_html: str | None
    image_url: str | None
    image_urls: list[str]
    source_url: str
    source_collection: str
    copyright_policy_url: str
    collected_at: str

    @field_validator(
        "source_url", "source_collection", "copyright_policy_url", "image_url"
    )
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        """
        Check if the URL is a valid HTTP or HTTPS link
        """
        if value is None:
            return None
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Expected an absolute HTTP(S) URL")
        return value

    @field_validator("image_urls")
    @classmethod
    def validate_image_urls(cls, values: list[str]) -> list[str]:
        """
        Check that all image links are valid URLs
        """
        for url in values:
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError(f"Invalid image URL: {url!r}")
        return values

    @field_validator("collected_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        """
        Check that the timestamp includes a timezone
        """
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Expected an ISO 8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_provenance(self) -> FigureRecord:
        """
        Check that the figure details match their source
        """
        root = f"https://www.ipcc.ch/report/ar6/{self.working_group.lower()}/figures/"
        if self.source_collection != root:
            raise ValueError("source_collection does not match working_group")
        if not self.source_url.startswith(root):
            raise ValueError(
                "source_url is outside the working group's figure collection"
            )
        if self.source_url == root:
            raise ValueError(
                "source_url must identify a figure, not the collection root"
            )

        relative_path = urlsplit(self.source_url).path.removeprefix(urlsplit(root).path)
        segments = [segment for segment in relative_path.split("/") if segment]
        expected_id = f"AR6-{self.working_group}-" + "--".join(segments)
        if self.figure_id != expected_id:
            raise ValueError("figure_id does not match source_url")
        if self.image_url is not None and self.image_url not in self.image_urls:
            raise ValueError("image_url must also occur in image_urls")
        return self

    @property
    def has_caption(self) -> bool:
        """
        Check if the figure has a non empty caption
        """
        return bool(self.caption and self.caption.strip())

"""
Configuration for PDF page rendering in the Documents module

These settings control how PDF pages are rendered and whether saved images
may be reused. Storage paths and pipeline wide settings are configured elsewhere.
"""

from __future__ import annotations
from pydantic import Field
from semanticlens.core.schemas import CoreSchema


class DocumentsConfig(CoreSchema):
    """
    Settings for producing RGB PNG images from PDF pages

    The renderer applies the rendering settings
    and the processor manages image caching
    """

    dpi: int = Field(
        default=180,
        ge=36,
        le=600,
        description="PDF rendering resolution in dots per inch.",
    )
    max_page_pixels: int = Field(
        default=25_000_000,
        gt=0,
        description=(
            "Maximum allowed width multiplied by height of a rendered page. "
            "The renderer must check this before allocating the image."
        ),
    )
    render_annotations: bool = Field(
        default=True,
        description="Include visible PDF annotations in rendered page images.",
    )
    reuse_cached_pages: bool = Field(
        default=True,
        description=(
            "Reuse existing page images only when the processor verifies that "
            "the PDF content and all rendering settings match."
        ),
    )

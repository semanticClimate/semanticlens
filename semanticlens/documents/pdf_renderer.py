"""
Render PDF pages as RGB images for visual document retrieval

This module handles PDF rendering only. 

The document processor is responsible for saving images, caching them
and creating evidence records
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from semanticlens.core.exceptions import ConfigurationError, DocumentProcessingError

from .config import DocumentsConfig


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """
    A rendered PDF page and its position in the source document.

    The image is an independent RGB Pillow image. The caller is responsible
    for closing it after saving or processing it.
    """

    page_idx: int
    page_count: int
    image: Image.Image


class PDFRenderer:
    """
    Render one PDF at a time using PDFium.

    Pages are yielded individually to avoid keeping all page images in memory.
    PDFium objects are released after rendering each page. PDFium should not
    be used concurrently from multiple threads.
    """

    def __init__(self, config: DocumentsConfig | None = None) -> None:
        if config is not None and not isinstance(config, DocumentsConfig):
            raise ConfigurationError("config must be a DocumentsConfig instance")
        self.config = config if config is not None else DocumentsConfig()

    def render_pages(self, pdf_path: str | Path) -> Iterator[RenderedPage]:
        """Yield RGB page images in source order, starting at page index zero.

        Pages use the configured DPI and a white background. Oversized pages
        raise an error before bitmap allocation instead of being resized.
        Close each yielded image after use. If stopping early, close the
        iterator to release the open PDF document.
        """
        if not isinstance(pdf_path, (str, Path)):
            raise TypeError("pdf_path must be a string or Path")
        if isinstance(pdf_path, str) and not pdf_path.strip():
            raise ValueError("pdf_path must not be empty")

        path = Path(pdf_path).expanduser()
        try:
            if not path.is_file():
                raise DocumentProcessingError(f"PDF path is not an existing file: {path}")

            with pdfium.PdfDocument(path) as document:
                page_count = len(document)
                if page_count == 0:
                    raise DocumentProcessingError(f"PDF has no pages: {path}")

                scale = self.config.dpi / 72.0
                for page_idx in range(page_count):
                    try:
                        page = document.get_page(page_idx)
                        try:
                            width, height = page.get_size()
                            if not (
                                math.isfinite(width)
                                and math.isfinite(height)
                                and width > 0
                                and height > 0
                            ):
                                raise DocumentProcessingError(
                                    f"Invalid dimensions on page {page_idx + 1} of {path}"
                                )

                            width_px = math.ceil(width * scale)
                            height_px = math.ceil(height * scale)
                            if width_px * height_px > self.config.max_page_pixels:
                                raise DocumentProcessingError(
                                    f"Page {page_idx + 1} of {path} would render at "
                                    f"{width_px}x{height_px} pixels, exceeding "
                                    f"max_page_pixels={self.config.max_page_pixels}. "
                                    "Lower dpi or increase the pixel limit."
                                )

                            bitmap = page.render(
                                scale=scale,
                                draw_annots=self.config.render_annotations,
                                fill_color=(255, 255, 255, 255),
                                rev_byteorder=True,
                            )
                            try:
                                # Copy the pixels before PDFium releases its bitmap
                                image = bitmap.to_pil().convert("RGB")
                            finally:
                                bitmap.close()
                        finally:
                            page.close()
                    except DocumentProcessingError:
                        raise
                    except Exception as exc:
                        raise DocumentProcessingError(
                            f"Failed to render page {page_idx + 1} of {path}"
                        ) from exc

                    yield RenderedPage(
                        page_idx=page_idx,
                        page_count=page_count,
                        image=image,
                    )

        except DocumentProcessingError:
            raise
        except pdfium.PdfiumError as exc:
            raise DocumentProcessingError(f"Could not open or read PDF {path}: {exc}") from exc
        except OSError as exc:
            raise DocumentProcessingError(f"Could not access PDF {path}: {exc}") from exc
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to process PDF: {path}") from exc

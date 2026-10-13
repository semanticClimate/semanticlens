"""
Prepare PDF pages and source references for visual document retrieval.

This processor implements Core's DocumentProcessor interface. It uses the PDF
renderer to create page images, saves them under a workspace and yields one
EvidenceUnit per page. OCR and layout parsing are deliberately separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image
from pypdfium2 import version as pdfium_version

from semanticlens.core.exceptions import ConfigurationError, DocumentProcessingError
from semanticlens.core.schemas import DocumentRef, EvidenceUnit

from .config import DocumentsConfig
from .pdf_renderer import PDFRenderer


_CACHE_VERSION = 1
_RENDER_PROFILE = "pdfium-rgb-png-v1"
_HASH_CHUNK_SIZE = 1024 * 1024


class PDFPageProcessor:
    """
    Render a PDF into cached PNG pages and source linked evidence units

    Images are stored beneath `workspace/pages`. An evidence unit's
    `image_ref` is relative to `workspace`, so other components must use
    the same workspace to read it.

    Different source files have different document IDs, even if their bytes
    are identical. Identical PDF bytes can share the same rendered images.

    Rendered images are cached only after all pages have been written and a
    cache manifest has been saved. Cache writes are not coordinated between
    concurrent processes; V1 is intended for sequential indexing.
    """

    def __init__(
        self,
        *,
        workspace: str | Path,
        config: DocumentsConfig | None = None,
        renderer: PDFRenderer | None = None,
    ) -> None:
        if not isinstance(workspace, (str, Path)) or (
            isinstance(workspace, str) and not workspace.strip()
        ):
            raise ConfigurationError("workspace must be a non-empty path")
        if config is not None and not isinstance(config, DocumentsConfig):
            raise ConfigurationError("config must be a DocumentsConfig instance")
        if renderer is not None and not isinstance(renderer, PDFRenderer):
            raise ConfigurationError("renderer must be a PDFRenderer instance")

        selected_config = (
            config
            if config is not None
            else (renderer.config if renderer is not None else DocumentsConfig())
        )
        if renderer is not None and renderer.config != selected_config:
            raise ConfigurationError("renderer and processor must use the same DocumentsConfig")

        self.config = selected_config
        self.renderer = renderer if renderer is not None else PDFRenderer(selected_config)
        try:
            self.workspace = Path(workspace).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError(f"Cannot resolve workspace path: {workspace}") from exc

    def process(self, pdf_path: Path) -> Iterator[EvidenceUnit]:
        """
        Yield one page evidence unit per PDF page, in document order.

        The PDF must be a readable local file. Images are saved before their
        evidence units are yielded. Close this iterator if stopping early.
        A completed run writes a manifest used to validate future cache hits.
        """
        source = _source_path(pdf_path)
        initial_stat = _file_state(source)
        source_hash = _file_sha256(source)
        _assert_unchanged(source, initial_stat)

        source_uri = source.as_uri()
        document_id = hashlib.sha256(
            f"{source_uri}\0{source_hash}".encode("utf-8")
        ).hexdigest()
        render_key = self._render_key()
        relative_dir = Path("pages") / source_hash[:2] / source_hash / render_key
        output_dir = self.workspace / relative_dir

        cached = (
            self._read_cache(output_dir, source_hash, render_key)
            if self.config.reuse_cached_pages
            else None
        )
        if cached is not None:
            doc = _document_ref(document_id, source_uri, source, source_hash, len(cached))
            for idx, item in enumerate(cached):
                yield _page_unit(doc, idx, relative_dir, item)
            _assert_unchanged(source, initial_stat)
            return

        records: list[dict[str, int | str]] = []
        doc: DocumentRef | None = None
        expected_count: int | None = None

        try:
            with closing(self.renderer.render_pages(source)) as pages:
                for rendered in pages:
                    try:
                        if expected_count is None:
                            expected_count = rendered.page_count
                            if expected_count <= 0:
                                raise DocumentProcessingError("PDF renderer returned no pages")
                            doc = _document_ref(
                                document_id, source_uri, source, source_hash, expected_count
                            )
                        if (
                            rendered.page_count != expected_count
                            or rendered.page_idx != len(records)
                            or rendered.page_idx >= expected_count
                        ):
                            raise DocumentProcessingError(
                                "PDF renderer returned inconsistent page indices or page counts"
                            )

                        width, height = rendered.image.size
                        if (
                            rendered.image.mode != "RGB"
                            or width <= 0
                            or height <= 0
                            or width * height > self.config.max_page_pixels
                        ):
                            raise DocumentProcessingError(
                                f"Invalid rendered image on page {rendered.page_idx + 1}"
                            )
                        target = output_dir / _page_filename(rendered.page_idx)
                        _save_png_atomic(rendered.image, target)
                        item: dict[str, int | str] = {
                            "width_px": width,
                            "height_px": height,
                            "sha256": _file_sha256(target),
                        }
                        records.append(item)
                    finally:
                        rendered.image.close()

                    assert doc is not None
                    yield _page_unit(doc, rendered.page_idx, relative_dir, item)
        except DocumentProcessingError:
            raise
        except (OSError, ValueError) as exc:
            raise DocumentProcessingError(f"Could not prepare PDF pages for {source}") from exc

        if expected_count is None or len(records) != expected_count:
            raise DocumentProcessingError(
                f"PDF renderer produced {len(records)} pages, expected {expected_count}"
            )
        _assert_unchanged(source, initial_stat)
        self._write_manifest(output_dir, source_hash, render_key, records)

    def _render_key(self) -> str:
        """Identify the settings and renderer version that affect saved pages."""
        identity = {
            "profile": _RENDER_PROFILE,
            "renderer": f"{type(self.renderer).__module__}.{type(self.renderer).__qualname__}",
            "pypdfium2_version": str(pdfium_version.PYPDFIUM_INFO),
            "pdfium_version": str(pdfium_version.PDFIUM_INFO),
            "dpi": self.config.dpi,
            "max_page_pixels": self.config.max_page_pixels,
            "render_annotations": self.config.render_annotations,
        }
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]

    def _read_cache(
        self, directory: Path, source_hash: str, render_key: str
    ) -> list[dict[str, int | str]] | None:
        """Return verified page metadata, or None when a cache must be rebuilt."""
        manifest_path = directory / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeError):
            return None

        if not isinstance(manifest, dict) or (
            manifest.get("cache_version") != _CACHE_VERSION
            or manifest.get("source_sha256") != source_hash
            or manifest.get("render_key") != render_key
        ):
            return None

        pages = manifest.get("pages")
        count = manifest.get("page_count")
        if not isinstance(pages, list) or not isinstance(count, int) or (
            isinstance(count, bool) or count <= 0 or len(pages) != count
        ):
            return None

        verified: list[dict[str, int | str]] = []
        for idx, item in enumerate(pages):
            if not isinstance(item, dict):
                return None
            width, height, digest = (
                item.get("width_px"),
                item.get("height_px"),
                item.get("sha256"),
            )
            if (
                type(width) is not int
                or type(height) is not int
                or width <= 0
                or height <= 0
                or width * height > self.config.max_page_pixels
                or not _valid_sha256(digest)
            ):
                return None

            image_path = directory / _page_filename(idx)
            try:
                if image_path.is_symlink() or _file_sha256(image_path) != digest:
                    return None
                with Image.open(image_path) as image:
                    if (
                        image.format != "PNG"
                        or image.mode != "RGB"
                        or image.size != (width, height)
                    ):
                        return None
                    image.verify()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                return None
            verified.append({"width_px": width, "height_px": height, "sha256": digest})
        return verified

    @staticmethod
    def _write_manifest(
        directory: Path,
        source_hash: str,
        render_key: str,
        records: list[dict[str, int | str]],
    ) -> None:
        manifest = {
            "cache_version": _CACHE_VERSION,
            "source_sha256": source_hash,
            "render_key": render_key,
            "page_count": len(records),
            "pages": records,
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            _write_atomic(directory / "manifest.json", payload)
        except OSError as exc:
            raise DocumentProcessingError("Failed to save PDF page cache manifest") from exc


def _source_path(value: Path) -> Path:
    if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value.strip()):
        raise DocumentProcessingError("PDF source must be a non-empty file path")
    try:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise DocumentProcessingError(f"Not a PDF file: {path}")
        return path
    except (OSError, RuntimeError) as exc:
        raise DocumentProcessingError(f"Cannot access PDF: {value}") from exc


def _file_state(path: Path) -> tuple[int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise DocumentProcessingError(f"Cannot read PDF file information: {path}") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _assert_unchanged(path: Path, expected: tuple[int, int, int, int]) -> None:
    if _file_state(path) != expected:
        raise DocumentProcessingError(f"PDF changed during processing: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DocumentProcessingError(f"Cannot read file: {path}") from exc
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        digit in "0123456789abcdef" for digit in value
    )


def _document_ref(
    document_id: str, source_uri: str, source: Path, sha256: str, page_count: int
) -> DocumentRef:
    return DocumentRef(
        document_id=document_id,
        source_uri=source_uri,
        sha256=sha256,
        title=source.stem,
        page_count=page_count,
    )


def _page_filename(idx: int) -> str:
    return f"page-{idx + 1:04d}.png"


def _page_unit(
    doc: DocumentRef, idx: int, relative_dir: Path, item: dict[str, int | str]
) -> EvidenceUnit:
    return EvidenceUnit(
        unit_id=f"{doc.document_id}:page:{idx}",
        document=doc,
        page_idx=idx,
        kind="page",
        image_ref=(relative_dir / _page_filename(idx)).as_posix(),
        metadata={
            "width_px": item["width_px"],
            "height_px": item["height_px"],
        },
    )


def _save_png_atomic(image: Image.Image, target: Path) -> None:
    """Write a complete PNG before replacing the final image path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.stem}-", suffix=".png", delete=False
        ) as stream:
            temporary = Path(stream.name)
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_atomic(target: Path, content: bytes) -> None:
    """Write a file atomically using a temporary file in the same directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.stem}-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

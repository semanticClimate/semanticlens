import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .mineru_adapter import parse_pdf
from ..schema import FIELD_PATHS, REQUIRED_PATHS


def create_spatial_page_map(
    pdf_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    pages: Sequence[int] | None = None,
    include_fields: Sequence[str] | None = None,
    exclude_fields: Sequence[str] | None = None,
    tier: str = "standard",
) -> dict[str, Any]:
    
    if include_fields is not None and exclude_fields is not None:
        raise ValueError("include_fields and exclude_fields cannot be used together")
    
    include = _fields(include_fields, "include_fields")
    exclude = _fields(exclude_fields, "exclude_fields")

    for field in exclude or ():
        if any(
            required == field or required.startswith(field + ".")
            for required in REQUIRED_PATHS
        ):
            raise ValueError(f"Cannot exclude required structural field: {field!r}")
        
    selected = None

    if pages is not None:
        if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
            raise TypeError("pages must be a sequence of zero-based integers")
        if any(type(page) is not int or page < 0 for page in pages):
            raise ValueError("pages must contain non-negative integers (not booleans)")
        selected = set(pages)

    path = Path(pdf_path).expanduser().resolve(strict=True)

    if not path.is_file():
        raise ValueError(f"pdf_path must be a file: {path}")
    
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ValueError(f"Input does not have a PDF header: {path}")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()

    destination = None

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        if destination == path or (destination.exists() and destination.samefile(path)):
            raise ValueError("output_path must not overwrite the input PDF")
        if destination.is_dir():
            raise ValueError("output_path must name a file, not a directory")

    model = parse_pdf(path, tier=tier)
    result = model.model_dump(mode="json", exclude_none=True)
    result["document"]["sha256"] = digest
    available = {page["page_idx"] for page in result["pages"]}
    page_count = result["document"].get("page_count")

    if result["is_full_document"] and page_count is not None:
        if available != set(range(page_count)):
            raise RuntimeError("MinerU reported a full document with missing pages")
        
    if selected is not None:
        missing = selected - available
        if missing:
            raise ValueError(f"Page indices not present in document: {sorted(missing)}")
        result["pages"] = [
            page for page in result["pages"] if page["page_idx"] in selected
        ]
        result["is_full_document"] = (
            result["is_full_document"] and selected == available
        )

    if include is not None or exclude is not None:
        result = _project(
            result,
            "",
            include | REQUIRED_PATHS if include is not None else None,
            exclude or frozenset(),
        )

    if destination is not None:
        _save(result, destination)

    return result


def _fields(values: Sequence[str] | None, name: str) -> frozenset[str] | None:
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of field-path strings")
    for value in values:
        if not isinstance(value, str) or value not in FIELD_PATHS:
            raise ValueError(
                f"Invalid field in {name}: {value!r}. "
                f"Valid paths: {', '.join(sorted(FIELD_PATHS))}"
            )
    return frozenset(values)


def _project(
    value: Any, path: str, include: frozenset[str] | None, exclude: frozenset[str]
) -> Any:
    if isinstance(value, list):
        return [_project(item, path, include, exclude) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        child = f"{path}.{key}" if path else key
        if any(child == field or child.startswith(field + ".") for field in exclude):
            continue
        if include is None or any(
            child == field
            or child.startswith(field + ".")
            or field.startswith(child + ".")
            for field in include
        ):
            result[key] = _project(item, child, include, exclude)
    return result


def _save(result: dict[str, Any], path: Path) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

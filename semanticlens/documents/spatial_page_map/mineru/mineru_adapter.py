from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..schema import Document, Page, Parser, Region, SpatialPageMap

ALLOWED_TIERS = {"flash", "basic", "standard", "advanced"}

def parse_pdf(path: Path, tier: str = "standard") -> SpatialPageMap:
    if tier not in ALLOWED_TIERS:
        raise ValueError(
            f"Invalid tier: {tier!r}. "
            f"Allowed tiers: {', '.join(sorted(ALLOWED_TIERS))}"
        )
    try:
        installed = version("mineru")
    except PackageNotFoundError as exc:
        raise ImportError("Install mineru==4.0.10 to parse PDFs with SPM.") from exc
    
    if installed != "4.0.10":
        raise RuntimeError(
            f"SPM supports MinerU 4.0.10; found {installed}. "
            "Revalidate the adapter before changing the dependency pin."
        )

    from mineru.parser import parse

    result = parse(path, tier=tier, ocr_mode="auto", page_range="")
    raw = result.middle_json.to_dict()

    return _normalize(raw, path.name)


def _spans(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        span = dict(item)
        content = span.pop("content")
        if isinstance(content, list):
            span["children"] = _spans(content)
            span["text"] = "".join(child["text"] for child in span["children"])
        else:
            span["text"] = content
        normalized.append(span)
    return normalized


def _regions(blocks: list[dict[str, Any]], page_idx: int) -> list[Region]:
    regions: list[Region] = []
    containers = {"image", "table", "chart", "code", "list", "index"}
    names = {
        "doc_title": "title",
        "paragraph_title": "heading",
        "ref_text": "reference",
        "aside_text": "aside",
    }

    def visit(block: dict[str, Any], locator: str, parent: str | None) -> None:
        fields = dict(block)
        kind = fields.pop("type")
        content = fields.pop("content")
        region = Region(
            id=locator,
            parent_id=parent,
            type=names.get(kind, kind),
            reading_order=len(regions),
            source_index=fields.pop("index", None),
            bbox=fields.pop("bbox", None),
            heading_level=fields.pop("level", None),
            continues_previous=fields.pop("continues_prev", None),
        )
        image = {
            target: fields.pop(source)
            for source, target in (
                ("image_base64", "data_uri"),
                ("image_path", "path"),
                ("image_url", "url"),
            )
            if fields.get(source) is not None
        }
        region.image = image or None
        region.attributes = {k: v for k, v in fields.items() if v is not None} or None
        regions.append(region)

        if kind in containers:
            if not isinstance(content, list):
                raise ValueError(f"Expected child blocks at {locator}")
            for position, child in enumerate(content):
                visit(child, f"{locator}.{position}", locator)

        elif isinstance(content, list):
            region.spans = _spans(content)
            region.text = "".join(span["text"] for span in region.spans)

        elif kind == "equation":
            region.latex = content

        elif kind in {"image_body", "table_body", "chart_body"}:
            region.html = content

        else:
            region.text = content

    for block in blocks:
        visit(block, f"p{page_idx}.b{block['index']}", None)

    return regions


def _normalize(raw: dict[str, Any], filename: str) -> SpatialPageMap:
    if (raw.get("schema"), raw.get("schema_version")) != ("docvortex.middle", "2.0"):
        raise ValueError("Unsupported MinerU output: expected docvortex.middle 2.0")
    
    metadata = raw["metadata"]
    if metadata["file_suffix"] != "pdf":
        raise ValueError("MinerU did not recognize the input as a PDF")
    
    extensions = raw.get("extensions", {})
    layout = extensions.get("docvortex_layout", {})

    if layout and layout.get("version") != 1:
        raise ValueError("Unsupported MinerU page geometry extension version")
    
    geometry = {page["page_idx"]: page for page in layout.get("pages", [])}
    properties = metadata.get("document") or {}
    engine = extensions.get("mineru", {})
    
    pages = [
        Page(
            page_idx=page["page_idx"],
            width_pt=geometry.get(page["page_idx"], {}).get("width_pt"),
            height_pt=geometry.get(page["page_idx"], {}).get("height_pt"),
            regions=_regions(page.get("blocks", []), page["page_idx"]),
        )
        for page in raw["pages"]
    ]
    return SpatialPageMap(
        is_full_document=raw["is_full_document"],
        document=Document(
            filename=filename,
            page_count=properties.get("page_count"),
            properties=properties or None,
        ),
        parser=Parser(
            **metadata["producer"],
            tier=engine.get("tier"),
            mode=engine.get("parse_mode"),
            source_schema=raw["schema"],
            source_schema_version=raw["schema_version"],
            extensions=extensions or None,
        ),
        pages=pages,
    )

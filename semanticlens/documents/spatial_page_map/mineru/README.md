### MinerU Spatial Page Map

Convert PDF pages into structured, machine readable JSON that describes what appears on each page and where it is located

This module uses MinerU to parse a PDF and convert its document structure into a normalized Spatial Page Map for SemanticLens

<hr>

### Setup

You can run the Spatial Page Map module in a notebook environment such as:

- Google Colab: CPU or GPU runtime

- Marimo: preferably with a GPU powered environment when using heavier MinerU parsing tiers

Install the currently supported dependencies:

```python
%pip install "mineru==4.0.10" "docvortex==0.5.7" "pydantic>=2.12.5,<3"
```

These are the versions currently used and supported by the MinerU Spatial Page Map module.

<hr>

### Quick Start

Import the module and provide a PDF:

```python
from spatial_page_map.mineru import create_spatial_page_map

spm = create_spatial_page_map("report.pdf")
```

The function returns the generated Spatial Page Map as a Python dictionary

To save the same output as JSON:
```python
spm = create_spatial_page_map(
    "report.pdf",
    output_path="report_spm.json",
)
```
You can then inspect the parsed pages directly:
```python
print(f"Pages: {len(spm['pages'])}")
```

<hr>

### Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| pdf_path | Required | Path to the PDF you want to convert into a Spatial Page Map. Accepts a string or path like object |
| output_path | None | Optional path where the generated Spatial Page Map should be saved as a JSON file. If omitted, the result is only returned as a Python dictionary |
| pages | None | Select specific pages from the PDF using zero based page indices for example `[0, 2, 5]`. If omitted all pages are included |
| include_fields | None | Keep only selected fields in the output JSON. Useful when you need a smaller or task specific Spatial Page Map |
| exclude_fields | None | Remove selected fields from the output while keeping everything else. Useful when only a few fields are unnecessary |
| tier | "basic" | Select the MinerU parsing tier. Supported values are `"flash"`, `"basic"`, `"standard"` and `"advanced"`. Different tiers provide different levels of parsing capability and compute requirements |

> [!IMPORTANT]
> Use either `include_fields` or `exclude_fields` in a single call not both

With no optional parameters the module processes the complete PDF and returns all available normalized Spatial Page Map fields

<hr>

### Parsing Tiers

- Flash

Uses the document's native structure and text where available with little or no model inference. It is the fastest option but provides the lowest parsing quality for complex PDFs

- Basic

Uses smaller models for tasks such as OCR, formula recognition and table extraction. It can run on lower resource systems and CPU based environments

- Standard

Combines the smaller parsing models with a Vision Language Model to handle more complex layouts and visual document structure. This is generally the best default when higher quality parsing is needed

- Advanced

Uses the same general model stack as standard but spends more inference compute on difficult pages. It is the slowest tier and is intended for cases where parsing quality matters more than speed

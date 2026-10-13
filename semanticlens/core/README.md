## SemanticLens Core

### Overview

**Core is the foundation of SemanticLens.** It defines how different modules communicate and work together to perform Visual Document Retrieval (VDR)

Its main purpose is to make SemanticLens **modular, reusable and easy to extend**

### What's Inside?

| File | Purpose |
|---|---|
| `schemas.py` | Defines common data formats |
| `interfaces.py` | Defines how components interact |
| `config.py` | Manages pipeline settings |
| `exceptions.py` | Provides clear error types |
| `pipeline.py` | Connects and runs components |
| `__init__.py` | Exposes the public API |

### Example Usage

```python
from semanticlens import SemanticLens

# Components are created separately.
lens = SemanticLens(
    processor=processor,
    embedder=embedder,
    index=index,
    retriever=retriever,
    resolver=resolver,
)

lens.index("./reports")

results = lens.search("Graphs showing global warming")

for hit in results:
    print(hit.evidence.page_number, hit.score)
```

> [!IMPORTANT]
> Illustrative example and actual component implementations are still being developed

## Why Is Core Useful?

Imagine experimenting with different models:

```text
Pipeline A: ColQwen → Qdrant
Pipeline B: ColPali → Another compatible index
```

Both can use the **same Core pipeline API** by providing components that follow its interfaces

This allows developers to explore new embedding models, retrieval techniques and indexing methods without rebuilding everything

**In short:** Core provides the shared structure and coordination that make SemanticLens a flexible and extensible VDR toolkit

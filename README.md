<h2>SemanticLens 🖼️</h2>

> Making the visual knowledge in scientific PDFs searchable, understandable and verifiable

<br>

SemanticLens began with a simple question: How much scientific knowledge do we overlook when we treat a document as text alone?

It is an open source project that is being developed within `semanticClimate` community exploring how we can interact more deeply with scientific literature while preserving the context and evidence that give that knowledge meaning

<hr>

### Problem

Climate change is one of the most consequential challenges facing society and responding to it depends on our ability to understand and use the best available scientific knowledge. A huge amount of that knowledge already exists

The IPCC assessments bring together scientific, technical and socio economic knowledge about climate change so that it can support scientists, policymakers, educators, industry, communities and citizens. The reports contain thousands of pages of carefully reviewed evidence covering how the climate is changing, its impacts, future risks, adaptation and mitigation

> [!IMPORTANT]
> The problem is not that the knowledge does not exist. The problem is that using it is difficult

<p align="center"><img src="https://www.ipcc.ch/report/ar6/wg1/downloads/figures/IPCC_AR6_WGI_TS_CCBox_1_Figure_1.png" height="400px"></p> 

<p align="center"><img src="https://www.ipcc.ch/report/ar6/wg1/downloads/figures/IPCC_AR6_WGI_Figure_10_20.png" height="400px"></p>

These sample pages from IPCC reports show how much information can exist within a single scientific page

> 👉 This is the kind of information Lens aims to make searchable and interrogable as a whole
  
A reader looking for one specific piece of evidence may need to search through hundreds of pages, understand specialised terminology, locate the relevant figure or table, interpret its legend or labels and connect it back to the surrounding discussion

A text only representation can therefore preserve the words while still losing part of the scientific meaning.

The question behind the project is:

> Can we interrogate a scientific PDF as a complete visual document while still keeping every answer grounded in evidence that a human can inspect and verify?

<hr>

### What We Are Building

`SemanticLens` is a visual interrogation pipeline for scientific PDFs

A user should eventually be able to ask a question such as:

> Which maps show the regions with the highest projected drought risk?

or

> Which figures compare future warming across different climate scenarios?

It will:

1. Prepare and visually index all pages of the report
2. Maintain structured information about the meaningful regions of each page
3. Retrieve and rank the pages most relevant to the user's question
4. Re rank the strongest candidates
5. Give only the selected evidence to a Vision Language Model
6. Visually reason over charts, maps, tables, figures, legends, labels and layout
7. eturn the answer together with the exact evidence from which it was derived

<img width="1633" height="889" alt="image" src="https://github.com/user-attachments/assets/bbe964b7-dd9f-4fc1-8bc4-b9dd138c645b" />

<hr>

### Visual Retrieval 

A scientific report may contain hundreds or thousands of pages but only a few of them are likely to contain the evidence needed for a particular question. SemanticLens therefore does not send the entire document to a Vision Language Model

During document preparation every page is rendered visually and passed through a `Visual Document Encoder`. The resulting page representations are stored in a searchable `Visual Index` allowing the complete report to be searched through its visual content rather than text alone

When a user asks a question, SemanticLens performs a `Similarity Search` against this index and retrieves a small set of the most relevant pages

This first ranking is intentionally broad: Its job is to quickly reduce a large document to a manageable set of likely candidates

The Top K pages are then passed through a `Re Ranking stage`. Re Ranking examines the strongest candidates more carefully and can use additional information including the query, visual relevance and information from the Spatial Page Map to decide which pages or regions contain the strongest evidence

> [!IMPORTANT]
> Only this final smaller set of evidence is passed to the VLM for visual interrogation

> [!NOTE]
> The separation is important: retrieval finds where the answer is likely to be while re ranking decides which of those candidates are actually worth reasoning over

The result is a pipeline that can search large scientific reports efficiently while keeping the expensive visual reasoning focused on the evidence that matters most

<hr>

### Spatial Page Map 🗺️

> Giving Structure and Meaning to the Page

Visual retrieval can tell us which pages are likely to matter but reliable interrogation also benefits from knowing what actually exists on those pages

SemanticLens therefore creates a Spatial Page Map for every page

The map begins with a layout aware analysis of the page. Instead of treating the page as one flat image, it is broken into structured regions. Each region receives information such as:

- Type - text, title, figure, table, caption, equation, image, etc.
- Position — its bounding box on the original page
- Content — extracted text or associated visual content
- Structure — reading order, grouping, parent / child relationships or nearby elements
- Page identity — the exact document and page from which the region came

A simplified representation might look like:

```json
{
  "page": 42,
  "regions": [
    {
      "id": "block_1",
      "type": "text",
      "bbox": [80, 120, 500, 240],
      "text": "Projected warming increases..."
    },
    {
      "id": "figure_1",
      "type": "figure",
      "bbox": [90, 280, 920, 720]
    },
    {
      "id": "caption_1",
      "type": "caption",
      "bbox": [90, 730, 920, 790],
      "text": "Figure 4.3..."
    }
  ]
}
```

> [!NOTE]
> Its exact JSON schema, region types and semantic fields will be refined during implementation

The Spatial Page Map is intended to remain machine readable most likely as `JSON` so it can be reused during Re Ranking, passed alongside selected pages to the VLM and inspected or tested independently

The Spatial Page Map helps SemanticLens know not only what a page looks like but what its parts are and how they fit together

> [!IMPORTANT]
> During visual interrogation, the selected page image is passed to the VLM together with its Spatial Page Map. The model therefore receives both the original visual evidence and structured information about the page rather than having to reconstruct the entire layout from scratch

<hr>

### Verifiable Output

> Keeping the Answer Connected to the Evidence

The final output of SemanticLens should be more than a generated answer

For every response, the system preserves the connection between the answer and the original scientific evidence used to produce its

A result should include:

* A clear text answer
* All supporting pages that contributed to that answer
* The exact evidence highlighted on the original page
* Basic source information such as document, page number, and visual object reference and a structured machine readable result for later reuse or evaluation

> [!NOTE]
> The highlighted evidence is especially important. SemanticLens should ideally be able to point to the exact part of that page that supports the answer

This makes the output easier to inspect and verify, and also makes it possible to evaluate whether the VLM's answer is actually grounded in the retrieved evidence

Give below is a sample image of how a supporting page with highlighted evidence looks like 👇

<img height="721" alt="image" src="https://github.com/user-attachments/assets/e5f81c66-2874-4058-95d4-bb7ec3662bf7" />


<hr>

### Testing and Evaluation

SemanticLens will be tested on a small curated set of scientific report pages with realistic questions and known supporting evidence

The initial evaluation will focus on whether the system can reliably:

* Retrieve relevant visual evidence
* Produce answers that are supported by the source
* Identify the correct supporting pages and regions
* Return results consistently across repeated runs

Where possible, the benchmark will include different kinds of scientific visuals such as charts, maps, tables, figures and mixed layout pages

Evaluation will combine quantitative measures with manual review of the retrieved evidence and final answers. This will help us understand both how often the system succeeds** and whether its answers remain faithful to the original scientific document

The test suite and benchmark will grow alongside the implementation with the aim of making SemanticLens easy to compare, reproduce and improve over time

<hr>

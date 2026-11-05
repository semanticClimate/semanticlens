### Project Overview

The `IPCC Figure-Term Knowledge Graph` makes figures from the IPCC Sixth Assessment Report AR6 easier to find and explore by linking them to scientific terms mentioned in their captions

The project collects figures and captions from all three IPCC Working Groups, identifies terms from third party export of [IPCC AR6 Glossary](https://github.com/TIBHannover/ClimateKG-Data-Bench/blob/main/data/glossary-ar6.html) in their captions, and builds a bipartite knowledge graph linking figures to terms

It provides

- Reusable figure corpus
- Scientific terms vocabulary
- Interactive graph visualization

to help researchers discover and explore related figures. All connections are based on caption text not image analysis

<br>

### Data Sources


| Type | Data | Source | Description |
|---|---|---|---|
| External source | IPCC AR6 figures | [WG1](https://www.ipcc.ch/report/ar6/wg1/figures/), [WG2](https://www.ipcc.ch/report/ar6/wg2/figures/), [WG3](https://www.ipcc.ch/report/ar6/wg3/figures/) | Official IPCC figure pages used to collect captions and figure details |
| External source | IPCC AR6 glossary | [TIB Hannover glossary export](https://github.com/TIBHannover/ClimateKG-Data-Bench/blob/main/data/glossary-ar6.html) | Third party HTML export of the [official IPCC glossary](https://apps.ipcc.ch/glossary/searchlatest.php) used to build the vocabulary |
| Input | Figure captions | [wg1.json](https://github.com/semanticClimate/semanticlens/blob/main/data/raw/wg1.json), [wg2.json](https://github.com/semanticClimate/semanticlens/blob/main/data/raw/wg2.json), [wg3.json](https://github.com/semanticClimate/semanticlens/blob/main/data/raw/wg3.json) | Collected figure metadata, captions, image URLs and source links |
| Input | Glossary vocabulary | [ipcc_ar6_vocabulary.json](https://github.com/semanticClimate/semanticlens/blob/main/data/processed/ipcc_ar6_vocabulary.json) | Extracted scientific terms and aliases used for caption matching |
| Output | Figure corpus | [corpus.db](https://github.com/semanticClimate/semanticlens/blob/main/data/processed/corpus.db) | Cleaned and structured figure records in SQLite |
| Output | Term matches | [concepts.db](https://github.com/semanticClimate/semanticlens/blob/main/data/processed/concepts.db) | Identified caption mentions and accepted figure–term links in SQLite |
| Output | Knowledge graph | [ipcc_visual_knowledge_graph.ttl](https://github.com/semanticClimate/semanticlens/blob/main/ontology/ipcc_visual_knowledge_graph.ttl) | Bipartite figure–term knowledge graph in RDF / Turtle format |

<br>

### Build Process


1. Figure Collection

Extract figure metadata, captions and image URLs from IPCC AR6 WG1, WG2 and WG3 pages using Requests and Beautiful Soup, and organize the collected data into separate [json files](https://github.com/neelkumar01/IPCC-visual-knowledge-graph/tree/main/data/raw)

2. Corpus Preparation

Clean, deduplicate and validate collected figure records using Pydantic then organize figures and captions in a structured SQLite [database](https://github.com/neelkumar01/IPCC-visual-knowledge-graph/blob/main/data/processed/corpus.db)

3. Term Matching

Identify [IPCC glossary](https://github.com/neelkumar01/IPCC-visual-knowledge-graph/blob/main/data/processed/ipcc_ar6_vocabulary.json) terms and aliases in figure captions using normalized, trie based lexical matching with longest match selection and store the matches in SQLite [database](https://github.com/neelkumar01/IPCC-visual-knowledge-graph/blob/main/data/processed/concepts.db)

4. Knowledge Graph Construction

Use RDFLib to build a bipartite knowledge graph linking IPCC figures to glossary terms found in their captions and save it in RDF/Turtle (.ttl) format

5. Search and Visualization

Search IPCC figures by glossary terms or caption text and explore their connections through interactive HTML knowledge graph


<br>


### Knowledge Graph

Nodes: Figure and ScientificConcept (glossary terms)
Links: `mentionsConcept` connects figures to terms found in their captions

- Interactive Interface

<img width="1710" alt="image" src="https://github.com/user-attachments/assets/f703aa86-bd19-4bb4-b9aa-295a60a8f0c3" />

<br>

- Figure Node selection

<img width="1710"  alt="image" src="https://github.com/user-attachments/assets/227e5669-ab42-40f3-a0cf-4e13d07f4e9d" />

<br>

- Scientific term node selection

<img width="1710" alt="image" src="https://github.com/user-attachments/assets/f921d881-2326-46fa-9283-b47118f83f2c" />

<br>

- Caption box on Figure node hover

<img width="1710"  alt="image" src="https://github.com/user-attachments/assets/bef925e4-d682-4ac4-9483-cd4f9c58d853" />

<br>

### Installation and Usage

Run these three cells in **Google Colab** to install the project, load the knowledge graph and explore figure–term connections

1. Install

```python
!git clone https://github.com/semanticClimate/semanticlens.git
%cd semanticlens
!pip install -e .
```

2. Load the knowledge graph

```python
from ipcc_kg import IPCCGraph

kg = IPCCGraph.from_repo()
```

3. Search and explore

```python
query = "heatwave"

graph = kg.explore(query, limit=10)

graph.show()
```

4. Export graph

```python
html_path = graph.save("graph.html")
rdf_path = graph.save("graph.ttl")

print(f"HTML visualization: {html_path}")
print(f"RDF Turtle: {rdf_path}")
```

<br>

### Results

| Result | Count |
|---|---:|
| IPCC AR6 figures collected (WG1, WG2, WG3) | 1310 |
| Figure captions collected | 1,297 |
| Glossary terms in the vocabulary | 920 |
| Unique figure–term links identified | 6,178 |
| Figures linked to at least one term | 1,215 |

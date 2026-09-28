<p align="center"><img src="assets/semanticLens.png" width=220px></p>
<h3 align="center">SemanticLens</h2>
<p align="center">Making climate science easier to find, understand and verify</p>

<br>

SemanticLens is an open source tool which combines visual and structural understanding of scientific documents to help connect questions and answers back to the exact evidence they come from

It aims to connect every answer back to the original evidence making complex climate knowledge easier to search, inspect and use


### Problem

Climate change is already increasing the risks of extreme heat, floods, droughts, crop losses, water stress and ecosystem decline. Understanding these risks clearly matters because decisions made today will shape how well we respond to them

The IPCC reports bring together some of the most important climate science available but they are large, dense and difficult to navigate. Important evidence is often spread across text, figures, maps, tables, captions and surrounding context

The problem is not the lack of scientific knowledge. It is finding the right evidence, understanding it in context and verifying exactly where it comes from

<p align="center"><br><img src="https://www.ipcc.ch/report/ar6/wg1/downloads/figures/IPCC_AR6_WGI_Figure_10_20.png" height="400px"></p>


### What We Are Building

SemanticLens is building a new way for AI to work with scientific documents as complete sources of knowledge not just collections of text

It represents the visual and structural information inside climate reports, retrieves the most relevant evidence across pages and regions and enables multimodal models to reason over it while preserving the context, provenance and exact source behind every result

Our goal is to make scientific knowledge directly searchable, interpretable, evidence grounded and verifiable by both people and AI systems

<p align="center"><br><img width="700" alt="image" src="https://github.com/user-attachments/assets/bbe964b7-dd9f-4fc1-8bc4-b9dd138c645b" /></p>


### Grounded Evidence

SemanticLens aims to show where the evidence actually is not just which page it came from

When a result is returned, the supporting part of the original page - a figure, table, map, caption, paragraph or other relevant region can be marked with a grounding box, making the source of the answer directly visible and easy to inspect

This keeps the answer connected to the exact visual evidence used during reasoning

In the future, these grounded regions may also be linked across pages and documents to form a richer Evidence Graph

<p align="center"><br><img height="500" alt="image" src="https://github.com/user-attachments/assets/e5f81c66-2874-4058-95d4-bb7ec3662bf7" /></p>


### Evaluation and Benchmarking

Evaluation is a core part of SemanticLens

The project will be tested on realistic scientific questions with known supporting evidence to understand how reliably the system can retrieve, interpret and ground information from complex climate documents

Different retrieval, representation and reasoning approaches will be compared to understand what actually improves performance rather than assuming a particular architecture works best

The benchmark will consider both the quality of the final answer and the quality of the evidence behind it including how accurately the system finds, grounds and preserves the source information it relies on

Over time, the goal is to build a reproducible benchmark that can also be used to evaluate new multimodal models and document understanding systems as they evolve

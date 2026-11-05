"""Public interface to the IPCC figure knowledge graph."""

from .query import IPCCGraph
from .subgraph import KnowledgeSubgraph

__all__ = ["IPCCGraph", "KnowledgeSubgraph"]

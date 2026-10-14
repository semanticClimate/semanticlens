"""
Embedding models and configuration for SemanticLens
"""

from .config import ColQwenConfig
from .colqwen import ColQwenEmbedder

__all__ = [
    "ColQwenConfig",
    "ColQwenEmbedder",
]


"""Evidence-grounded answer generation for SemanticLens.

Provides the Qwen3-VL generator and citation assembly utilities.
Model weights and inference dependencies are loaded lazily.
"""

from .config import Qwen3VLConfig
from .grounding import build_answer_result
from .qwen3_vl import Qwen3VLGenerator

__all__ = [
    "Qwen3VLConfig",
    "Qwen3VLGenerator",
    "build_answer_result",
]

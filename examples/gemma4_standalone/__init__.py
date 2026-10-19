"""Standalone Gemma 4 (google/gemma-4-E2B architecture) — no `transformers` runtime.

    modeling_gemma4.py   every nn.Module, the config tree, the loss and the
                         safetensors weight loader (single file, by design)
    preprocessing.py     tokenizer / image processor / audio feature extractor

See README.md in this folder and the `hf-model-standalone` skill for the method.
"""

from .modeling_gemma4 import (  # noqa: F401
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    load_gemma4_from_safetensors,
)
from .preprocessing import (  # noqa: F401
    Gemma4FeatureExtractor,
    Gemma4ImageProcessor,
    Gemma4Processor,
    Gemma4Tokenizer,
)

__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "Gemma4AudioConfig",
    "Gemma4ForConditionalGeneration",
    "load_gemma4_from_safetensors",
    "Gemma4Tokenizer",
    "Gemma4ImageProcessor",
    "Gemma4FeatureExtractor",
    "Gemma4Processor",
]

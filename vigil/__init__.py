"""VIGIL - Vision Gating with Intermittent Listening.

Cost-bounded multimodal LLM perception over continuous video and audio
streams, built for black-box model APIs and a zero-compute edge gate.
"""
__version__ = "0.1.0"

from .config import Config
from .pipeline import Pipeline

__all__ = ["Config", "Pipeline", "__version__"]

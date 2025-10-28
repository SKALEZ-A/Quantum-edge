"""
Edge AI Runtime - Lightweight Inference Engine

This module provides optimized runtime for AI inference on edge devices,
including model compression, quantization, adaptive execution, and
resource-aware scheduling.
"""

__version__ = "1.0.0"
__author__ = "Quantum Edge AI Team"

from .inference_engine import EdgeInferenceEngine
from .model_compression import ModelCompressor
from .quantization import AdaptiveQuantizer
from .resource_manager import ResourceManager
from .adaptive_scheduler import AdaptiveScheduler
from .model_cache import ModelCache

__all__ = [
    'EdgeInferenceEngine',
    'ModelCompressor',
    'AdaptiveQuantizer',
    'ResourceManager',
    'AdaptiveScheduler',
    'ModelCache'
]

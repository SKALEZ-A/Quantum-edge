"""
Quantum Edge AI Platform - Utilities

Comprehensive utility libraries for data processing, monitoring, caching,
and common operations across the platform.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility version and metadata
__version__ = "1.0.0"
__author__ = "Quantum Edge AI Platform Team"

class UtilityManager:
    """Central manager for utility components"""

    def __init__(self):
        self.components = {}
        self.metrics = {}

    def register_component(self, name: str, component: Any):
        """Register a utility component"""
        self.components[name] = component
        logger.info(f"Registered utility component: {name}")

    def get_component(self, name: str) -> Any:
        """Get a registered component"""
        return self.components.get(name)

    def get_status(self) -> Dict[str, Any]:
        """Get status of all utility components"""
        return {
            'total_components': len(self.components),
            'components': list(self.components.keys()),
            'metrics': self.metrics,
            'timestamp': time.time()
        }

# Global utility manager instance
utility_manager = UtilityManager()

def get_utility_manager() -> UtilityManager:
    """Get the global utility manager"""
    return utility_manager

# Common utility functions
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    try:
        return a / b if b != 0 else default
    except (ZeroDivisionError, TypeError):
        return default

def calculate_percentage(part: float, total: float, decimals: int = 2) -> float:
    """Calculate percentage safely"""
    return round(safe_divide(part, total) * 100, decimals)

def format_bytes(bytes_value: int, decimals: int = 2) -> str:
    """Format bytes to human readable format"""
    if bytes_value == 0:
        return "0 B"

    k = 1024
    sizes = ["B", "KB", "MB", "GB", "TB"]
    i = 0

    while bytes_value >= k and i < len(sizes) - 1:
        bytes_value /= k
        i += 1

    return f"{bytes_value:.{decimals}f} {sizes[i]}"

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def moving_average(values: List[float], window: int = 5) -> List[float]:
    """Calculate moving average"""
    if len(values) < window:
        return values

    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        avg = sum(values[start:i+1]) / len(values[start:i+1])
        averages.append(avg)

    return averages

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics"""
    if not values:
        return {}

    import numpy as np
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': len(values)
    }

# Timing utilities
class Timer:
    """Simple timer for performance measurement"""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time"""
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        logger.debug(f"{self.name} took {elapsed:.4f} seconds")
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

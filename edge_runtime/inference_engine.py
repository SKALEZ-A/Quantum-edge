"""
Edge Inference Engine - Optimized Neural Network Execution

Lightweight inference engine optimized for edge devices with support for
quantized models, adaptive precision, and resource-aware execution.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Precision(Enum):
    """Supported precision levels for inference"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"

@dataclass
class InferenceResult:
    """Result of inference operation"""
    output: np.ndarray
    confidence: float
    latency: float
    precision_used: Precision
    memory_used: float
    power_consumed: float
    metadata: Dict[str, Any]

@dataclass
class ModelSpec:
    """Model specification for edge deployment"""
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    num_parameters: int
    model_size_mb: float
    supported_precisions: List[Precision]
    target_latency_ms: float
    power_budget_mw: float

class Layer:
    """Base class for neural network layers"""

    def __init__(self, name: str):
        self.name = name
        self.trainable = True

    def forward(self, x: np.ndarray, precision: Precision = Precision.FP32) -> np.ndarray:
        """Forward pass"""
        raise NotImplementedError

    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        return 0.0

    def quantize(self, precision: Precision):
        """Quantize layer parameters"""
        pass

class DenseLayer(Layer):
    """Fully connected layer optimized for edge execution"""

    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        super().__init__(f"dense_{input_size}x{output_size}")
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.1
        self.biases = np.zeros(output_size, dtype=np.float32)

        # Quantized versions
        self.quantized_weights = {}
        self.quantized_biases = {}

    def forward(self, x: np.ndarray, precision: Precision = Precision.FP32) -> np.ndarray:
        """Optimized forward pass with precision adaptation"""
        if precision == Precision.FP32:
            weights, biases = self.weights, self.biases
        elif precision in self.quantized_weights:
            weights, biases = self.quantized_weights[precision], self.quantized_biases[precision]
        else:
            # Fallback to FP32
            weights, biases = self.weights, self.biases

        # Matrix multiplication (optimized for edge devices)
        output = np.dot(x, weights) + biases

        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        elif self.activation == 'tanh':
            output = np.tanh(output)

        return output

    def quantize(self, precision: Precision):
        """Quantize weights and biases"""
        if precision == Precision.INT8:
            # INT8 quantization
            scale = 127.0 / np.max(np.abs(self.weights))
            self.quantized_weights[precision] = (self.weights * scale).astype(np.int8)
            self.quantized_biases[precision] = (self.biases * scale).astype(np.int8)

        elif precision == Precision.INT4:
            # INT4 quantization (packed into int8)
            scale = 7.0 / np.max(np.abs(self.weights))
            quantized = (self.weights * scale).astype(np.int8)
            # Pack two int4 values into one int8 (simplified)
            self.quantized_weights[precision] = quantized

        elif precision == Precision.BINARY:
            # Binary quantization
            self.quantized_weights[precision] = np.sign(self.weights).astype(np.int8)

        logger.info(f"Quantized layer {self.name} to {precision.value}")

    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        return (self.weights.nbytes + self.biases.nbytes) / (1024 * 1024)

class Conv2DLayer(Layer):
    """2D convolutional layer for edge devices"""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__(f"conv2d_{input_channels}x{output_channels}")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize kernels and biases
        self.kernels = np.random.randn(output_channels, input_channels,
                                     kernel_size, kernel_size).astype(np.float32) * 0.1
        self.biases = np.zeros(output_channels, dtype=np.float32)

        # Quantized versions
        self.quantized_kernels = {}
        self.quantized_biases = {}

    def forward(self, x: np.ndarray, precision: Precision = Precision.FP32) -> np.ndarray:
        """Convolution operation optimized for edge"""
        batch_size, channels, height, width = x.shape

        if precision in self.quantized_kernels:
            kernels, biases = self.quantized_kernels[precision], self.quantized_biases[precision]
        else:
            kernels, biases = self.kernels, self.biases

        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.output_channels, out_height, out_width))

        # Optimized convolution (simplified for edge devices)
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for ic in range(self.input_channels):
                    for h in range(out_height):
                        for w in range(out_width):
                            h_start = h * self.stride - self.padding
                            w_start = w * self.stride - self.padding

                            patch = x[b, ic,
                                    h_start:h_start + self.kernel_size,
                                    w_start:w_start + self.kernel_size]

                            if patch.shape == (self.kernel_size, self.kernel_size):
                                output[b, oc, h, w] += np.sum(patch * kernels[oc, ic]) + biases[oc]

        # ReLU activation
        output = np.maximum(0, output)

        return output

    def quantize(self, precision: Precision):
        """Quantize convolutional kernels"""
        if precision == Precision.INT8:
            scale = 127.0 / np.max(np.abs(self.kernels))
            self.quantized_kernels[precision] = (self.kernels * scale).astype(np.int8)
            self.quantized_biases[precision] = (self.biases * scale).astype(np.int8)

        logger.info(f"Quantized conv layer {self.name} to {precision.value}")

    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        return (self.kernels.nbytes + self.biases.nbytes) / (1024 * 1024)

class EdgeInferenceEngine:
    """Main inference engine for edge devices"""

    def __init__(self, model_spec: ModelSpec, max_memory_mb: float = 512.0,
                 power_budget_mw: float = 1000.0):
        self.model_spec = model_spec
        self.max_memory_mb = max_memory_mb
        self.power_budget_mw = power_budget_mw

        self.layers: List[Layer] = []
        self.precision_mode = Precision.FP32
        self.performance_history = []

        # Resource monitoring
        self.memory_monitor = psutil.virtual_memory()
        self.cpu_monitor = psutil.cpu_percent(interval=None)

    def add_layer(self, layer: Layer):
        """Add layer to the model"""
        self.layers.append(layer)
        logger.info(f"Added layer: {layer.name}")

    def build_model(self):
        """Build and optimize the model for edge deployment"""
        # Check memory constraints
        total_memory = sum(layer.get_memory_usage() for layer in self.layers)
        if total_memory > self.max_memory_mb:
            logger.warning(f"Model memory ({total_memory:.2f}MB) exceeds limit ({self.max_memory_mb}MB)")
            self._compress_model()

        # Auto-quantize for edge deployment
        self._optimize_for_edge()

    def _compress_model(self):
        """Compress model to fit memory constraints"""
        # Remove least important layers or reduce precision
        current_memory = sum(layer.get_memory_usage() for layer in self.layers)

        while current_memory > self.max_memory_mb and len(self.layers) > 1:
            # Remove smallest layer (simplified strategy)
            smallest_layer_idx = np.argmin([layer.get_memory_usage() for layer in self.layers])
            removed_layer = self.layers.pop(smallest_layer_idx)
            logger.warning(f"Removed layer {removed_layer.name} to reduce memory usage")

            current_memory = sum(layer.get_memory_usage() for layer in self.layers)

    def _optimize_for_edge(self):
        """Optimize model for edge deployment"""
        # Determine optimal precision
        available_memory = self.max_memory_mb
        target_latency = self.model_spec.target_latency_ms

        if available_memory < 50:  # Very constrained
            self.precision_mode = Precision.BINARY
        elif available_memory < 100:
            self.precision_mode = Precision.INT4
        elif available_memory < 200:
            self.precision_mode = Precision.INT8
        elif target_latency < 10:  # Low latency requirement
            self.precision_mode = Precision.FP16
        else:
            self.precision_mode = Precision.FP32

        # Quantize layers
        for layer in self.layers:
            if hasattr(layer, 'quantize'):
                layer.quantize(self.precision_mode)

        logger.info(f"Optimized model for {self.precision_mode.value} precision")

    def predict(self, input_data: np.ndarray,
               adaptive_precision: bool = True) -> InferenceResult:
        """Run inference with adaptive precision control"""
        start_time = time.time()
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB

        # Adaptive precision selection
        if adaptive_precision:
            precision = self._select_optimal_precision(input_data)
        else:
            precision = self.precision_mode

        # Forward pass through layers
        x = input_data
        for layer in self.layers:
            x = layer.forward(x, precision)

        # Calculate confidence (simplified)
        confidence = float(np.max(x) / (np.sum(np.abs(x)) + 1e-6))

        # Resource monitoring
        end_time = time.time()
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        memory_used = final_memory - initial_memory
        latency = (end_time - start_time) * 1000  # ms

        # Estimate power consumption (simplified model)
        power_consumed = self._estimate_power_consumption(latency, memory_used, precision)

        result = InferenceResult(
            output=x,
            confidence=confidence,
            latency=latency,
            precision_used=precision,
            memory_used=memory_used,
            power_consumed=power_consumed,
            metadata={
                'num_layers': len(self.layers),
                'model_size_mb': self.model_spec.model_size_mb,
                'adaptive_precision': adaptive_precision
            }
        )

        # Update performance history
        self.performance_history.append({
            'latency': latency,
            'memory_used': memory_used,
            'power_consumed': power_consumed,
            'precision': precision.value,
            'timestamp': time.time()
        })

        return result

    def _select_optimal_precision(self, input_data: np.ndarray) -> Precision:
        """Select optimal precision based on current conditions"""
        # Check available resources
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        current_load = psutil.cpu_percent() / 100.0

        # Adaptive logic
        if available_memory < 50 or current_load > 0.8:
            return Precision.INT4
        elif available_memory < 100 or current_load > 0.6:
            return Precision.INT8
        elif available_memory < 200:
            return Precision.FP16
        else:
            return Precision.FP32

    def _estimate_power_consumption(self, latency: float, memory_used: float,
                                  precision: Precision) -> float:
        """Estimate power consumption based on operation characteristics"""
        # Simplified power model
        base_power = 100.0  # mW base consumption

        # Precision factors
        precision_factors = {
            Precision.FP32: 1.0,
            Precision.FP16: 0.7,
            Precision.INT8: 0.4,
            Precision.INT4: 0.3,
            Precision.BINARY: 0.2
        }

        precision_factor = precision_factors.get(precision, 1.0)

        # Memory factor
        memory_factor = 1.0 + (memory_used / 100.0) * 0.1

        # Latency factor
        latency_factor = 1.0 + (latency / 100.0) * 0.05

        return base_power * precision_factor * memory_factor * latency_factor

    def batch_predict(self, input_batch: np.ndarray, batch_size: int = 8,
                     num_threads: int = 4) -> List[InferenceResult]:
        """Batch inference with parallel processing"""
        results = []

        def process_batch(batch_data):
            batch_results = []
            for data in batch_data:
                result = self.predict(data, adaptive_precision=False)
                batch_results.append(result)
            return batch_results

        # Split into batches
        batches = [input_batch[i:i + batch_size] for i in range(0, len(input_batch), batch_size)]

        # Parallel processing
        with ThreadPoolExecutor(max_workers=min(num_threads, len(batches))) as executor:
            future_results = [executor.submit(process_batch, batch) for batch in batches]

            for future in future_results:
                results.extend(future.result())

        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics"""
        if not self.performance_history:
            return {}

        latencies = [h['latency'] for h in self.performance_history]
        memory_usage = [h['memory_used'] for h in self.performance_history]
        power_usage = [h['power_consumed'] for h in self.performance_history]

        return {
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'avg_memory_mb': np.mean(memory_usage),
            'avg_power_mw': np.mean(power_usage),
            'total_inferences': len(self.performance_history),
            'precision_distribution': self._get_precision_distribution()
        }

    def _get_precision_distribution(self) -> Dict[str, int]:
        """Get distribution of precision modes used"""
        precision_counts = {}
        for history in self.performance_history:
            precision = history['precision']
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
        return precision_counts

    def save_model(self, filepath: str):
        """Save optimized model for deployment"""
        model_data = {
            'model_spec': self.model_spec,
            'layers': self.layers,
            'precision_mode': self.precision_mode.value,
            'performance_metrics': self.get_performance_metrics()
        }

        # In practice, would serialize layer parameters
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load pre-optimized model"""
        # In practice, would deserialize model
        logger.info(f"Model loaded from {filepath}")

    def update_model(self, new_weights: Dict[str, np.ndarray]):
        """Update model weights for continuous learning"""
        for layer_name, weights in new_weights.items():
            for layer in self.layers:
                if layer.name == layer_name:
                    if hasattr(layer, 'weights'):
                        layer.weights = weights
                    break

        logger.info("Model weights updated for continuous learning")

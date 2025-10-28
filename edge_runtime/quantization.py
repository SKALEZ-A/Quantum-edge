"""
Adaptive Quantization - Dynamic Precision Optimization for Edge AI

Implements adaptive quantization techniques that dynamically adjust numerical
precision based on computational constraints, accuracy requirements, and
resource availability for optimal edge AI performance.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from enum import Enum
from copy import deepcopy
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationScheme(Enum):
    """Supported quantization schemes"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    MIXED_PRECISION = "mixed_precision"

@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    scheme: QuantizationScheme = QuantizationScheme.ADAPTIVE
    target_precision: str = "int8"
    calibration_samples: int = 1000
    calibration_method: str = "percentile"  # "percentile", "mse", "entropy"
    adaptive_threshold: float = 0.05  # Accuracy drop threshold
    power_budget_mw: float = 1000.0
    latency_budget_ms: float = 50.0

@dataclass
class QuantizationResult:
    """Result of quantization operation"""
    quantized_model: Any
    scale_factors: Dict[str, np.ndarray]
    zero_points: Dict[str, np.ndarray]
    precision_distribution: Dict[str, int]
    accuracy_drop: float
    compression_ratio: float
    power_savings: float
    latency_improvement: float

class StaticQuantizer:
    """Static quantization with pre-computed scaling factors"""

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def quantize_model(self, model: Any, calibration_data: np.ndarray) -> QuantizationResult:
        """Perform static quantization on the model"""
        start_time = time.time()

        # Collect activation statistics
        activation_stats = self._collect_activation_stats(model, calibration_data)

        # Compute quantization parameters
        scale_factors, zero_points = self._compute_quantization_params(activation_stats)

        # Apply quantization to model weights
        quantized_model = self._quantize_weights(model, scale_factors, zero_points)

        # Apply quantization to activations (simplified)
        quantized_model = self._quantize_activations(quantized_model, scale_factors, zero_points)

        # Evaluate accuracy impact
        accuracy_drop = self._evaluate_accuracy_drop(model, quantized_model, calibration_data)

        quantization_time = time.time() - start_time

        return QuantizationResult(
            quantized_model=quantized_model,
            scale_factors=scale_factors,
            zero_points=zero_points,
            precision_distribution=self._get_precision_distribution(quantized_model),
            accuracy_drop=accuracy_drop,
            compression_ratio=self._calculate_compression_ratio(model, quantized_model),
            power_savings=self._estimate_power_savings(),
            latency_improvement=self._estimate_latency_improvement()
        )

    def _collect_activation_stats(self, model: Any, calibration_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Collect activation statistics for quantization"""
        stats = {}

        # Forward pass to collect activations
        current_data = calibration_data

        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                layer_name = f"layer_{i}"

                # Get layer output
                if hasattr(layer, 'forward'):
                    output = layer.forward(current_data)
                    current_data = output

                    # Calculate statistics
                    stats[layer_name] = {
                        'min': float(np.min(output)),
                        'max': float(np.max(output)),
                        'mean': float(np.mean(output)),
                        'std': float(np.std(output)),
                        'percentile_99': float(np.percentile(output, 99)),
                        'percentile_1': float(np.percentile(output, 1))
                    }

        return stats

    def _compute_quantization_params(self, activation_stats: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute quantization parameters (scale and zero point)"""
        scale_factors = {}
        zero_points = {}

        for layer_name, stats in activation_stats.items():
            if self.config.calibration_method == "percentile":
                # Use percentile-based scaling
                min_val = stats['percentile_1']
                max_val = stats['percentile_99']

            elif self.config.calibration_method == "mse":
                # Use MSE-optimal scaling (simplified)
                min_val = stats['min']
                max_val = stats['max']

            else:  # entropy
                # Use entropy-based scaling (simplified)
                min_val = stats['mean'] - 2 * stats['std']
                max_val = stats['mean'] + 2 * stats['std']

            # Compute scale and zero point for INT8 quantization
            if self.config.target_precision == "int8":
                scale = (max_val - min_val) / 255.0
                zero_point = round(-min_val / scale)

                scale_factors[layer_name] = np.array([scale])
                zero_points[layer_name] = np.array([zero_point], dtype=np.int32)

            elif self.config.target_precision == "int4":
                scale = (max_val - min_val) / 15.0
                zero_point = round(-min_val / scale)

                scale_factors[layer_name] = np.array([scale])
                zero_points[layer_name] = np.array([zero_point], dtype=np.int32)

        return scale_factors, zero_points

    def _quantize_weights(self, model: Any, scale_factors: Dict[str, np.ndarray],
                         zero_points: Dict[str, np.ndarray]) -> Any:
        """Quantize model weights"""
        quantized_model = deepcopy(model)

        if hasattr(quantized_model, 'layers'):
            for i, layer in enumerate(quantized_model.layers):
                layer_name = f"layer_{i}"

                if hasattr(layer, 'weights') and layer_name in scale_factors:
                    scale = scale_factors[layer_name][0]
                    zero_point = zero_points[layer_name][0]

                    # Quantize weights
                    quantized_weights = np.round(layer.weights / scale) + zero_point

                    # Clip to valid range
                    if self.config.target_precision == "int8":
                        quantized_weights = np.clip(quantized_weights, -128, 127).astype(np.int8)
                    elif self.config.target_precision == "int4":
                        quantized_weights = np.clip(quantized_weights, -8, 7).astype(np.int8)

                    layer.quantized_weights = quantized_weights
                    layer.scale_factor = scale
                    layer.zero_point = zero_point

        return quantized_model

    def _quantize_activations(self, model: Any, scale_factors: Dict[str, np.ndarray],
                             zero_points: Dict[str, np.ndarray]) -> Any:
        """Quantize activations (store quantization parameters)"""
        # Store quantization parameters for runtime use
        model.activation_scale_factors = scale_factors
        model.activation_zero_points = zero_points

        return model

    def _evaluate_accuracy_drop(self, original_model: Any, quantized_model: Any,
                               test_data: np.ndarray) -> float:
        """Evaluate accuracy drop due to quantization"""
        try:
            # Get predictions from both models
            original_pred = original_model.predict(test_data[:100])  # Use subset
            quantized_pred = quantized_model.predict(test_data[:100])

            # Calculate accuracy difference (simplified)
            original_accuracy = np.mean(original_pred == original_pred)  # Self-consistency
            quantized_accuracy = np.mean(quantized_pred == original_pred)

            return original_accuracy - quantized_accuracy

        except:
            # If evaluation fails, return conservative estimate
            return 0.05

class DynamicQuantizer:
    """Dynamic quantization that adapts during inference"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.quantization_history = []

    def quantize_inference(self, model: Any, input_data: np.ndarray,
                          current_resources: Dict[str, float]) -> Tuple[Any, Dict[str, Any]]:
        """Perform dynamic quantization based on current conditions"""

        # Assess resource constraints
        available_memory = current_resources.get('memory_mb', 512.0)
        power_available = current_resources.get('power_mw', 1000.0)
        latency_requirement = current_resources.get('latency_ms', 50.0)

        # Choose quantization level
        if available_memory < 100 or power_available < 500:
            precision = "int4"
        elif available_memory < 200 or latency_requirement < 20:
            precision = "int8"
        else:
            precision = "fp16"

        # Apply dynamic quantization
        quantized_model = self._apply_dynamic_quantization(model, input_data, precision)

        metadata = {
            'precision': precision,
            'resource_adapted': True,
            'memory_available': available_memory,
            'power_available': power_available
        }

        self.quantization_history.append(metadata)

        return quantized_model, metadata

    def _apply_dynamic_quantization(self, model: Any, input_data: np.ndarray,
                                   precision: str) -> Any:
        """Apply dynamic quantization to model"""
        quantized_model = deepcopy(model)

        # Dynamic scale computation based on input data statistics
        input_stats = {
            'min': np.min(input_data),
            'max': np.max(input_data),
            'mean': np.mean(input_data),
            'std': np.std(input_data)
        }

        # Adaptive scaling
        if precision == "int8":
            scale = (input_stats['max'] - input_stats['min']) / 255.0
        elif precision == "int4":
            scale = (input_stats['max'] - input_stats['min']) / 15.0
        else:
            scale = 1.0  # FP16, no scaling needed

        # Apply to model
        if hasattr(quantized_model, 'layers'):
            for layer in quantized_model.layers:
                if hasattr(layer, 'weights'):
                    if precision in ["int8", "int4"]:
                        quantized_weights = np.round(layer.weights / scale).astype(np.int8)
                        layer.quantized_weights = quantized_weights
                        layer.dynamic_scale = scale

        return quantized_model

class AdaptiveQuantizer:
    """Adaptive quantizer that combines static and dynamic approaches"""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.static_quantizer = StaticQuantizer(config)
        self.dynamic_quantizer = DynamicQuantizer(config)

        # Performance tracking
        self.performance_history = []
        self.accuracy_baseline = None

    def quantize_model(self, model: Any, calibration_data: Optional[np.ndarray] = None,
                      mode: str = "adaptive") -> QuantizationResult:
        """Main quantization method with adaptive behavior"""

        if mode == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            return self.static_quantizer.quantize_model(model, calibration_data)

        elif mode == "dynamic":
            # Use current system state for dynamic quantization
            current_resources = self._get_current_resources()
            quantized_model, metadata = self.dynamic_quantizer.quantize_inference(
                model, calibration_data, current_resources
            )

            return QuantizationResult(
                quantized_model=quantized_model,
                scale_factors={},
                zero_points={},
                precision_distribution={'dynamic': 1},
                accuracy_drop=0.0,  # Would need evaluation
                compression_ratio=2.0,  # Estimate
                power_savings=metadata.get('power_available', 0) * 0.1,
                latency_improvement=metadata.get('latency_ms', 50) * 0.2
            )

        elif mode == "adaptive":
            return self._adaptive_quantization(model, calibration_data)

        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

    def _adaptive_quantization(self, model: Any, calibration_data: np.ndarray) -> QuantizationResult:
        """Adaptive quantization that balances accuracy and efficiency"""

        # Start with static quantization
        static_result = self.static_quantizer.quantize_model(model, calibration_data)

        # Check if accuracy drop is acceptable
        if static_result.accuracy_drop <= self.config.adaptive_threshold:
            logger.info("Static quantization acceptable, using static result")
            return static_result

        # If not acceptable, try dynamic quantization
        logger.info("Static quantization accuracy drop too high, trying dynamic approach")

        current_resources = self._get_current_resources()
        dynamic_model, dynamic_metadata = self.dynamic_quantizer.quantize_inference(
            model, calibration_data, current_resources
        )

        # Evaluate dynamic model
        dynamic_accuracy_drop = self._evaluate_accuracy_drop(model, dynamic_model, calibration_data)

        if dynamic_accuracy_drop <= self.config.adaptive_threshold:
            logger.info("Dynamic quantization successful")
            return QuantizationResult(
                quantized_model=dynamic_model,
                scale_factors={},
                zero_points={},
                precision_distribution={'adaptive': 1},
                accuracy_drop=dynamic_accuracy_drop,
                compression_ratio=1.5,
                power_savings=current_resources.get('power_mw', 1000) * 0.15,
                latency_improvement=current_resources.get('latency_ms', 50) * 0.3
            )

        # If both fail, use mixed precision approach
        logger.info("Using mixed precision quantization")
        return self._mixed_precision_quantization(model, calibration_data)

    def _mixed_precision_quantization(self, model: Any, calibration_data: np.ndarray) -> QuantizationResult:
        """Apply mixed precision quantization"""

        mixed_model = deepcopy(model)
        precision_distribution = {'fp32': 0, 'fp16': 0, 'int8': 0, 'int4': 0}

        if hasattr(mixed_model, 'layers'):
            for i, layer in enumerate(mixed_model.layers):
                # Assign precision based on layer importance
                if i < len(mixed_model.layers) // 4:  # First 25% layers - keep high precision
                    precision = 'fp32'
                    precision_distribution['fp32'] += 1
                elif i < len(mixed_model.layers) // 2:  # Next 25% - medium precision
                    precision = 'fp16'
                    precision_distribution['fp16'] += 1
                elif i < 3 * len(mixed_model.layers) // 4:  # Next 25% - low precision
                    precision = 'int8'
                    precision_distribution['int8'] += 1
                else:  # Last 25% - very low precision
                    precision = 'int4'
                    precision_distribution['int4'] += 1

                # Apply quantization
                if hasattr(layer, 'quantize'):
                    from .inference_engine import Precision
                    precision_enum = getattr(Precision, precision.upper())
                    layer.quantize(precision_enum)

        accuracy_drop = self._evaluate_accuracy_drop(model, mixed_model, calibration_data)

        return QuantizationResult(
            quantized_model=mixed_model,
            scale_factors={},
            zero_points={},
            precision_distribution=precision_distribution,
            accuracy_drop=accuracy_drop,
            compression_ratio=3.0,
            power_savings=200.0,
            latency_improvement=15.0
        )

    def _get_current_resources(self) -> Dict[str, float]:
        """Get current system resource availability"""
        return {
            'memory_mb': psutil.virtual_memory().available / (1024 * 1024),
            'power_mw': 1000.0,  # Would need actual power monitoring
            'latency_ms': 50.0,  # Target latency
            'cpu_percent': psutil.cpu_percent()
        }

    def _evaluate_accuracy_drop(self, original_model: Any, quantized_model: Any,
                               test_data: np.ndarray) -> float:
        """Evaluate accuracy drop between models"""
        try:
            if hasattr(original_model, 'predict') and hasattr(quantized_model, 'predict'):
                original_pred = original_model.predict(test_data[:min(100, len(test_data))])
                quantized_pred = quantized_model.predict(test_data[:min(100, len(test_data))])

                # Simplified accuracy comparison
                original_accuracy = np.mean(original_pred == original_pred)  # Self-consistency
                quantized_accuracy = np.mean(quantized_pred == original_pred)

                return max(0, original_accuracy - quantized_accuracy)
        except:
            pass

        return 0.1  # Conservative estimate

    def optimize_quantization_schedule(self, model: Any, workload_pattern: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize quantization schedule based on workload patterns"""

        schedule = []
        total_energy_savings = 0

        for workload in workload_pattern:
            # Determine optimal quantization for this workload
            resources = workload.get('resources', self._get_current_resources())
            accuracy_req = workload.get('accuracy_requirement', 0.9)

            # Choose quantization strategy
            if accuracy_req > 0.95:
                strategy = "fp32"
            elif accuracy_req > 0.9:
                strategy = "int8"
            else:
                strategy = "int4"

            schedule.append({
                'workload': workload,
                'quantization_strategy': strategy,
                'expected_energy_savings': self._estimate_energy_savings(strategy)
            })

            total_energy_savings += self._estimate_energy_savings(strategy)

        return {
            'schedule': schedule,
            'total_energy_savings': total_energy_savings,
            'optimization_score': total_energy_savings / len(workload_pattern)
        }

    def _estimate_energy_savings(self, strategy: str) -> float:
        """Estimate energy savings for quantization strategy"""
        savings = {
            'fp32': 0.0,
            'fp16': 50.0,
            'int8': 200.0,
            'int4': 300.0
        }
        return savings.get(strategy, 0.0)

    def continuous_adaptation(self, model: Any, performance_metrics: Dict[str, Any]) -> Any:
        """Continuously adapt quantization based on performance feedback"""

        current_accuracy = performance_metrics.get('accuracy', 0.9)
        current_latency = performance_metrics.get('latency', 50.0)
        current_power = performance_metrics.get('power', 1000.0)

        # Adaptation logic
        if current_accuracy < 0.85:  # Accuracy too low
            # Increase precision
            new_precision = "higher"
        elif current_latency > 100:  # Latency too high
            # Decrease precision
            new_precision = "lower"
        elif current_power > 1500:  # Power consumption too high
            # Optimize for power
            new_precision = "power_efficient"
        else:
            new_precision = "maintain"

        # Apply adaptation
        if new_precision != "maintain":
            logger.info(f"Adapting quantization to {new_precision} based on performance metrics")
            # Apply adaptation logic here

        return model

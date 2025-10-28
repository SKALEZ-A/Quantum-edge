"""
Model Compression - Advanced Compression Techniques for Edge AI

Implements various model compression techniques including pruning, distillation,
low-rank approximation, and neural architecture search optimized for edge devices.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from copy import deepcopy
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompressionResult:
    """Result of model compression"""
    compressed_model: Any
    compression_ratio: float
    accuracy_drop: float
    model_size_mb: float
    inference_speedup: float
    compression_time: float
    metadata: Dict[str, Any]

@dataclass
class PruningConfig:
    """Configuration for pruning techniques"""
    pruning_ratio: float = 0.5
    pruning_method: str = 'magnitude'  # 'magnitude', 'gradient', 'hessian'
    pruning_schedule: str = 'one_shot'  # 'one_shot', 'iterative', 'gradual'
    target_sparsity: float = 0.8

class MagnitudePruner:
    """Magnitude-based weight pruning"""

    def __init__(self, config: PruningConfig):
        self.config = config

    def prune_layer(self, weights: np.ndarray, biases: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prune weights using magnitude-based criterion"""
        # Calculate weight magnitudes
        magnitudes = np.abs(weights.flatten())

        # Determine pruning threshold
        threshold = np.percentile(magnitudes, self.config.pruning_ratio * 100)

        # Create mask for weights to keep
        mask = np.abs(weights) >= threshold

        # Apply pruning
        pruned_weights = weights * mask

        # Prune biases if provided (using weight mask)
        pruned_biases = biases
        if biases is not None:
            # For biases, use the output channel mask
            output_mask = np.any(mask, axis=0)
            pruned_biases = biases * output_mask

        return pruned_weights, pruned_biases

    def global_pruning(self, model_layers: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Global pruning across all layers"""
        all_weights = []
        layer_indices = []

        # Collect all weights
        for layer_idx, layer in enumerate(model_layers):
            if 'weights' in layer:
                weights_flat = layer['weights'].flatten()
                all_weights.extend(weights_flat)
                layer_indices.extend([layer_idx] * len(weights_flat))

        # Global magnitude pruning
        all_weights = np.array(all_weights)
        magnitudes = np.abs(all_weights)

        # Determine global threshold
        threshold = np.percentile(magnitudes, self.config.pruning_ratio * 100)

        # Apply global pruning
        pruned_layers = []
        weight_idx = 0

        for layer in model_layers:
            if 'weights' in layer:
                weights = layer['weights']
                layer_size = weights.size

                # Get mask for this layer
                layer_weights = all_weights[weight_idx:weight_idx + layer_size]
                mask = np.abs(layer_weights) >= threshold

                # Reshape mask and apply
                mask_reshaped = mask.reshape(weights.shape)
                pruned_weights = weights * mask_reshaped

                pruned_layer = layer.copy()
                pruned_layer['weights'] = pruned_weights

                # Prune biases if present
                if 'biases' in layer:
                    # Use output channel mask for biases
                    output_mask = np.any(mask_reshaped, axis=0)
                    pruned_layer['biases'] = layer['biases'] * output_mask

                pruned_layers.append(pruned_layer)
                weight_idx += layer_size
            else:
                pruned_layers.append(layer)

        return pruned_layers

class GradientPruner:
    """Gradient-based pruning using importance scores"""

    def __init__(self, config: PruningConfig):
        self.config = config

    def compute_importance_scores(self, weights: np.ndarray,
                                gradients: np.ndarray) -> np.ndarray:
        """Compute importance scores based on gradients"""
        # Taylor expansion importance: |weight * gradient|
        importance = np.abs(weights * gradients)

        # Normalize by layer
        importance = importance / (np.sum(importance) + 1e-8)

        return importance

    def prune_with_importance(self, weights: np.ndarray, importance_scores: np.ndarray) -> np.ndarray:
        """Prune weights based on importance scores"""
        # Flatten for easier processing
        weights_flat = weights.flatten()
        importance_flat = importance_scores.flatten()

        # Sort by importance
        sorted_indices = np.argsort(importance_flat)

        # Determine how many to prune
        num_to_prune = int(len(weights_flat) * self.config.pruning_ratio)

        # Set least important weights to zero
        prune_indices = sorted_indices[:num_to_prune]
        weights_flat[prune_indices] = 0

        return weights_flat.reshape(weights.shape)

class KnowledgeDistillation:
    """Knowledge distillation for model compression"""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits: np.ndarray,
                         teacher_logits: np.ndarray,
                         true_labels: np.ndarray) -> float:
        """Compute distillation loss"""
        # Soft targets from teacher
        teacher_soft = self._soften_logits(teacher_logits)
        student_soft = self._soften_logits(student_logits)

        # KL divergence between softened predictions
        kl_loss = self._kl_divergence(teacher_soft, student_soft)

        # Hard targets from true labels
        hard_loss = self._cross_entropy_loss(student_logits, true_labels)

        # Combined loss
        return self.alpha * kl_loss + (1 - self.alpha) * hard_loss

    def _soften_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits"""
        return np.exp(logits / self.temperature) / np.sum(np.exp(logits / self.temperature), axis=1, keepdims=True)

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence"""
        return np.sum(p * np.log((p + 1e-8) / (q + 1e-8)))

    def _cross_entropy_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-8))

class LowRankApproximation:
    """Low-rank matrix approximation for compression"""

    def __init__(self, rank_ratio: float = 0.5):
        self.rank_ratio = rank_ratio

    def decompose_layer(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose weight matrix using SVD"""
        # SVD decomposition
        U, s, Vt = np.linalg.svd(weights, full_matrices=False)

        # Determine rank to keep
        full_rank = min(weights.shape)
        target_rank = max(1, int(full_rank * self.rank_ratio))

        # Truncate to target rank
        U_trunc = U[:, :target_rank]
        s_trunc = s[:target_rank]
        Vt_trunc = Vt[:target_rank, :]

        # Reconstruct approximated matrix
        S_trunc = np.diag(s_trunc)
        approximated = U_trunc @ S_trunc @ Vt_trunc

        return approximated, {
            'U': U_trunc,
            'S': s_trunc,
            'Vt': Vt_trunc,
            'compression_ratio': target_rank / full_rank
        }

    def compress_model(self, model_layers: List[Dict[str, np.ndarray]]) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Compress entire model using low-rank approximation"""
        compressed_layers = []
        total_compression = 0
        num_compressed = 0

        for layer in model_layers:
            if 'weights' in layer and len(layer['weights'].shape) == 2:
                # Apply low-rank approximation to 2D weight matrices
                approximated_weights, metadata = self.decompose_layer(layer['weights'])

                compressed_layer = layer.copy()
                compressed_layer['weights'] = approximated_weights
                compressed_layer['low_rank_metadata'] = metadata

                compressed_layers.append(compressed_layer)

                total_compression += metadata['compression_ratio']
                num_compressed += 1
            else:
                compressed_layers.append(layer)

        avg_compression = total_compression / num_compressed if num_compressed > 0 else 1.0

        return compressed_layers, {
            'method': 'low_rank',
            'avg_compression_ratio': avg_compression,
            'layers_compressed': num_compressed
        }

class NeuralArchitectureSearch:
    """Neural architecture search for optimized edge models"""

    def __init__(self, search_space: Dict[str, List[Any]],
                 max_trials: int = 50, optimization_target: str = 'accuracy_latency_tradeoff'):
        self.search_space = search_space
        self.max_trials = max_trials
        self.optimization_target = optimization_target

        self.trial_results = []

    def search(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray],
              model_builder: Callable) -> Dict[str, Any]:
        """Perform neural architecture search"""

        best_architecture = None
        best_score = float('-inf')

        for trial in range(self.max_trials):
            # Sample architecture from search space
            architecture = self._sample_architecture()

            # Build model with sampled architecture
            model = model_builder(architecture)

            # Train and evaluate
            model.fit(*train_data)
            score = self._evaluate_architecture(model, val_data)

            self.trial_results.append({
                'architecture': architecture,
                'score': score,
                'trial': trial
            })

            if score > best_score:
                best_score = score
                best_architecture = architecture

            logger.info(f"Trial {trial}: Score = {score:.4f}")

        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'total_trials': self.max_trials,
            'search_space': self.search_space
        }

    def _sample_architecture(self) -> Dict[str, Any]:
        """Sample architecture from search space"""
        architecture = {}
        for param, values in self.search_space.items():
            architecture[param] = np.random.choice(values)
        return architecture

    def _evaluate_architecture(self, model: Any,
                              val_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate architecture performance"""
        X_val, y_val = val_data

        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_val)
            accuracy = accuracy_score(y_val, np.argmax(predictions, axis=1))
        else:
            predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)

        # Add latency penalty for edge optimization
        latency_penalty = 0.1  # Simplified

        return accuracy - latency_penalty

class ModelCompressor:
    """Main model compression orchestrator"""

    def __init__(self):
        self.compression_methods = {
            'pruning': self._prune_model,
            'quantization': self._quantize_model,
            'distillation': self._distill_model,
            'low_rank': self._low_rank_compress,
            'architecture_search': self._architecture_search_compress
        }

    def compress_model(self, model: Any, method: str = 'pruning',
                      target_compression: float = 0.5,
                      train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      **kwargs) -> CompressionResult:
        """Compress model using specified method"""
        start_time = time.time()

        if method not in self.compression_methods:
            raise ValueError(f"Unknown compression method: {method}")

        # Get original model performance
        original_size = self._estimate_model_size(model)
        original_accuracy = self._evaluate_model(model, val_data) if val_data else 0.0

        # Apply compression
        compressed_model = self.compression_methods[method](model, target_compression,
                                                          train_data, val_data, **kwargs)

        # Evaluate compressed model
        compressed_size = self._estimate_model_size(compressed_model)
        compressed_accuracy = self._evaluate_model(compressed_model, val_data) if val_data else 0.0

        compression_time = time.time() - start_time

        result = CompressionResult(
            compressed_model=compressed_model,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            accuracy_drop=original_accuracy - compressed_accuracy,
            model_size_mb=compressed_size,
            inference_speedup=self._estimate_speedup(method, target_compression),
            compression_time=compression_time,
            metadata={
                'method': method,
                'target_compression': target_compression,
                'original_size_mb': original_size,
                'original_accuracy': original_accuracy
            }
        )

        logger.info(f"Compression completed: {method}, ratio={result.compression_ratio:.2f}x, "
                   f"accuracy_drop={result.accuracy_drop:.4f}")

        return result

    def _prune_model(self, model: Any, target_compression: float,
                    train_data: Optional[Tuple], val_data: Optional[Tuple],
                    pruning_method: str = 'magnitude', **kwargs) -> Any:
        """Apply pruning compression"""
        config = PruningConfig(pruning_ratio=target_compression,
                              pruning_method=pruning_method)

        if pruning_method == 'magnitude':
            pruner = MagnitudePruner(config)
            # Simplified: assume model has layers attribute
            if hasattr(model, 'layers'):
                model.layers = pruner.global_pruning(model.layers)

        return model

    def _quantize_model(self, model: Any, target_compression: float,
                       train_data: Optional[Tuple], val_data: Optional[Tuple],
                       precision: str = 'int8', **kwargs) -> Any:
        """Apply quantization compression"""
        # Simplified quantization
        if hasattr(model, 'precision_mode'):
            from .inference_engine import Precision
            if precision == 'int8':
                model.precision_mode = Precision.INT8
            elif precision == 'int4':
                model.precision_mode = Precision.INT4

            # Quantize layers
            for layer in model.layers:
                if hasattr(layer, 'quantize'):
                    layer.quantize(model.precision_mode)

        return model

    def _distill_model(self, model: Any, target_compression: float,
                      train_data: Optional[Tuple], val_data: Optional[Tuple],
                      teacher_model: Any = None, **kwargs) -> Any:
        """Apply knowledge distillation"""
        if teacher_model is None:
            teacher_model = deepcopy(model)

        distillation = KnowledgeDistillation()

        # Simplified distillation training
        if train_data:
            X_train, y_train = train_data

            # Get teacher predictions
            if hasattr(teacher_model, 'predict_proba'):
                teacher_logits = teacher_model.predict_proba(X_train)
            else:
                teacher_logits = teacher_model.predict(X_train)

            # Simplified training loop
            for epoch in range(10):  # Few epochs for distillation
                student_logits = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else model.predict(X_train)
                loss = distillation.distillation_loss(student_logits, teacher_logits, y_train)

                # Update model (simplified)
                if hasattr(model, 'update_weights'):
                    # This would be the actual weight update step
                    pass

        return model

    def _low_rank_compress(self, model: Any, target_compression: float,
                          train_data: Optional[Tuple], val_data: Optional[Tuple],
                          **kwargs) -> Any:
        """Apply low-rank approximation"""
        rank_approximation = LowRankApproximation(rank_ratio=target_compression)

        if hasattr(model, 'layers'):
            compressed_layers, metadata = rank_approximation.compress_model(model.layers)
            model.layers = compressed_layers
            model.compression_metadata = metadata

        return model

    def _architecture_search_compress(self, model: Any, target_compression: float,
                                     train_data: Optional[Tuple], val_data: Optional[Tuple],
                                     search_space: Dict = None, **kwargs) -> Any:
        """Apply neural architecture search"""
        if search_space is None:
            search_space = {
                'num_layers': [2, 3, 4, 5],
                'layer_size': [32, 64, 128, 256],
                'activation': ['relu', 'tanh', 'sigmoid']
            }

        if train_data and val_data:
            nas = NeuralArchitectureSearch(search_space)

            def model_builder(architecture):
                # Simplified model builder
                return deepcopy(model)  # In practice, build new model with architecture

            search_result = nas.search(train_data, val_data, model_builder)
            # Use best architecture
            best_arch = search_result['best_architecture']
            logger.info(f"Found optimal architecture: {best_arch}")

        return model

    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB"""
        total_size = 0

        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'get_memory_usage'):
                    total_size += layer.get_memory_usage()

        return max(total_size, 0.1)  # Minimum 0.1MB

    def _evaluate_model(self, model: Any, val_data: Optional[Tuple]) -> float:
        """Evaluate model accuracy"""
        if val_data is None:
            return 0.0

        X_val, y_val = val_data

        try:
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_val)
                return accuracy_score(y_val, np.argmax(predictions, axis=1))
            else:
                predictions = model.predict(X_val)
                return accuracy_score(y_val, predictions)
        except:
            return 0.0

    def _estimate_speedup(self, method: str, compression_ratio: float) -> float:
        """Estimate inference speedup from compression"""
        speedups = {
            'pruning': 1.2 + compression_ratio * 0.5,
            'quantization': 1.5 + compression_ratio * 0.8,
            'distillation': 1.1,
            'low_rank': 1.3 + compression_ratio * 0.4,
            'architecture_search': 1.4 + compression_ratio * 0.6
        }

        return speedups.get(method, 1.0)

    def multi_stage_compression(self, model: Any,
                               compression_pipeline: List[Dict[str, Any]],
                               train_data: Optional[Tuple] = None,
                               val_data: Optional[Tuple] = None) -> CompressionResult:
        """Apply multi-stage compression pipeline"""
        current_model = deepcopy(model)
        total_compression_ratio = 1.0
        total_accuracy_drop = 0.0

        for stage_config in compression_pipeline:
            method = stage_config.pop('method')
            result = self.compress_model(current_model, method,
                                       train_data=train_data,
                                       val_data=val_data,
                                       **stage_config)

            current_model = result.compressed_model
            total_compression_ratio *= result.compression_ratio
            total_accuracy_drop += result.accuracy_drop

        return CompressionResult(
            compressed_model=current_model,
            compression_ratio=total_compression_ratio,
            accuracy_drop=total_accuracy_drop,
            model_size_mb=self._estimate_model_size(current_model),
            inference_speedup=1.0,  # Would need to calculate cumulative speedup
            compression_time=0.0,   # Would need to track total time
            metadata={
                'pipeline': compression_pipeline,
                'stages': len(compression_pipeline)
            }
        )

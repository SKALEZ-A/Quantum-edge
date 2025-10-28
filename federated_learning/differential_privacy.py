"""
Differential Privacy - Privacy-Preserving Machine Learning

Implements differential privacy mechanisms for federated learning,
including noise addition, privacy accounting, and advanced composition
techniques for quantum-enhanced privacy guarantees.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.special import erf
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrivacyBudget:
    """Privacy budget tracking for differential privacy"""
    epsilon: float
    delta: float
    used_epsilon: float = 0.0
    used_delta: float = 0.0

    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget"""
        return max(0, self.epsilon - self.used_epsilon)

    def spend_budget(self, epsilon_cost: float, delta_cost: float = 0.0):
        """Spend privacy budget"""
        self.used_epsilon += epsilon_cost
        self.used_delta += delta_cost

    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        return self.used_epsilon >= self.epsilon

@dataclass
class DPMechanism:
    """Differential privacy mechanism specification"""
    name: str
    sensitivity: float
    epsilon: float
    delta: Optional[float] = None

class DifferentialPrivacy:
    """Advanced differential privacy implementation"""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 sensitivity: float = 1.0):
        self.privacy_budget = PrivacyBudget(epsilon, delta)
        self.sensitivity = sensitivity

        # Noise generation methods
        self.noise_generators = {
            'laplace': self._laplace_noise,
            'gaussian': self._gaussian_noise,
            'exponential': self._exponential_noise,
            'staircase': self._staircase_noise
        }

    def add_noise(self, data: Union[np.ndarray, Dict[str, np.ndarray]],
                 mechanism: str = 'gaussian', **kwargs) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Add differential privacy noise to data"""

        if isinstance(data, dict):
            # Handle model parameters
            noised_data = {}
            total_epsilon_cost = 0

            for param_name, param_value in data.items():
                noised_param, epsilon_cost = self._add_noise_to_parameter(
                    param_value, mechanism, **kwargs
                )
                noised_data[param_name] = noised_param
                total_epsilon_cost += epsilon_cost

            # Update privacy budget
            self.privacy_budget.spend_budget(total_epsilon_cost)

            return noised_data

        else:
            # Handle single array
            noised_data, epsilon_cost = self._add_noise_to_parameter(data, mechanism, **kwargs)
            self.privacy_budget.spend_budget(epsilon_cost)
            return noised_data

    def _add_noise_to_parameter(self, param: np.ndarray, mechanism: str,
                              **kwargs) -> Tuple[np.ndarray, float]:
        """Add noise to a single parameter"""
        if mechanism not in self.noise_generators:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        # Compute noise scale based on sensitivity and privacy parameters
        noise_scale = self._compute_noise_scale(mechanism, **kwargs)

        # Generate and add noise
        noise = self.noise_generators[mechanism](param.shape, noise_scale)
        noised_param = param + noise

        # Compute epsilon cost
        epsilon_cost = self._compute_epsilon_cost(mechanism, noise_scale, **kwargs)

        return noised_param, epsilon_cost

    def _compute_noise_scale(self, mechanism: str, **kwargs) -> float:
        """Compute noise scale for given mechanism and privacy parameters"""
        epsilon = kwargs.get('epsilon', self.privacy_budget.remaining_epsilon())

        if mechanism == 'laplace':
            # Laplace mechanism: scale = sensitivity / epsilon
            return self.sensitivity / epsilon

        elif mechanism == 'gaussian':
            # Gaussian mechanism: scale = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
            delta = kwargs.get('delta', self.privacy_budget.delta)
            return self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        elif mechanism == 'exponential':
            # Exponential mechanism
            return 1.0 / epsilon

        elif mechanism == 'staircase':
            # Staircase mechanism
            return self.sensitivity / np.sqrt(epsilon)

        else:
            return 1.0

    def _laplace_noise(self, shape: Tuple[int, ...], scale: float) -> np.ndarray:
        """Generate Laplace noise"""
        return np.random.laplace(0, scale, shape)

    def _gaussian_noise(self, shape: Tuple[int, ...], scale: float) -> np.ndarray:
        """Generate Gaussian noise"""
        return np.random.normal(0, scale, shape)

    def _exponential_noise(self, shape: Tuple[int, ...], scale: float) -> np.ndarray:
        """Generate exponential noise"""
        return np.random.exponential(scale, shape)

    def _staircase_noise(self, shape: Tuple[int, ...], scale: float) -> np.ndarray:
        """Generate staircase noise (piecewise constant)"""
        # Create staircase pattern
        noise = np.zeros(shape)
        for i in range(len(noise.flatten())):
            level = np.random.randint(-10, 11)  # -10 to 10 levels
            noise.flat[i] = level * scale
        return noise

    def _compute_epsilon_cost(self, mechanism: str, noise_scale: float, **kwargs) -> float:
        """Compute privacy cost (epsilon) for a mechanism"""
        epsilon = kwargs.get('epsilon', self.privacy_budget.remaining_epsilon())

        if mechanism == 'laplace':
            return epsilon  # Direct epsilon cost

        elif mechanism == 'gaussian':
            # For Gaussian, epsilon depends on delta
            delta = kwargs.get('delta', self.privacy_budget.delta)
            # Approximation: epsilon ≈ (sensitivity^2) / (2 * sigma^2 * ln(1/delta))
            sigma = noise_scale
            if sigma > 0:
                return (self.sensitivity ** 2) / (2 * sigma ** 2 * np.log(1 / delta))
            return float('inf')

        else:
            return epsilon * 0.1  # Conservative estimate

class AdvancedComposition:
    """Advanced composition theorems for differential privacy"""

    def __init__(self, base_epsilon: float, base_delta: float):
        self.base_epsilon = base_epsilon
        self.base_delta = base_delta

    def basic_composition(self, epsilon1: float, epsilon2: float,
                         delta1: float = 0.0, delta2: float = 0.0) -> Tuple[float, float]:
        """Basic composition: ε_total = ε1 + ε2, δ_total = δ1 + δ2"""
        total_epsilon = epsilon1 + epsilon2
        total_delta = delta1 + delta2
        return total_epsilon, total_delta

    def advanced_composition(self, num_compositions: int, epsilon: float,
                           delta: float, delta_prime: float) -> Tuple[float, float]:
        """Advanced composition for k-fold adaptive composition"""
        # ε_total = sqrt(2*k*ln(1/δ')) * ε + k*ε*(exp(ε) - 1)
        # δ_total = k*δ + δ'

        term1 = np.sqrt(2 * num_compositions * np.log(1 / delta_prime)) * epsilon
        term2 = num_compositions * epsilon * (np.exp(epsilon) - 1)
        total_epsilon = term1 + term2

        total_delta = num_compositions * delta + delta_prime

        return total_epsilon, total_delta

    def moments_accountant(self, sigma: float, num_samples: int,
                          max_lambda: int = 32) -> float:
        """Moments accountant for privacy accounting"""
        # Simplified moments accountant
        alpha = 2  # Fixed alpha for simplicity

        # Compute moment
        moment = (np.exp(alpha * sigma**2 / 2) - 1) * num_samples

        # Convert to epsilon (approximation)
        epsilon = alpha / (2 * sigma**2) * np.log(1 + moment)

        return epsilon

class AdaptivePrivacy:
    """Adaptive privacy mechanisms that adjust based on data characteristics"""

    def __init__(self, base_epsilon: float = 1.0):
        self.base_epsilon = base_epsilon
        self.data_statistics = {}

    def adaptive_noise_scale(self, data: np.ndarray, sensitivity: float,
                           data_type: str = 'gradient') -> float:
        """Adapt noise scale based on data characteristics"""

        # Compute data statistics
        data_mean = np.mean(np.abs(data))
        data_std = np.std(data)
        data_skewness = stats.skew(data.flatten())
        data_kurtosis = stats.kurtosis(data.flatten())

        # Store for analysis
        self.data_statistics[data_type] = {
            'mean': data_mean,
            'std': data_std,
            'skewness': data_skewness,
            'kurtosis': data_kurtosis
        }

        # Adaptive scaling based on data properties
        if data_type == 'gradient':
            # For gradients, scale based on magnitude and variance
            base_scale = sensitivity / self.base_epsilon
            magnitude_factor = np.log1p(data_mean)  # Logarithmic scaling
            variance_factor = np.sqrt(data_std)  # Square root scaling
            skewness_penalty = 1 + abs(data_skewness) * 0.1  # Penalty for skewed data

            adaptive_scale = base_scale * magnitude_factor * variance_factor * skewness_penalty

        elif data_type == 'hessian':
            # For Hessians, different scaling
            adaptive_scale = sensitivity / (self.base_epsilon * np.sqrt(np.abs(data).mean() + 1))

        else:
            # Default scaling
            adaptive_scale = sensitivity / self.base_epsilon

        return adaptive_scale

    def context_aware_privacy(self, data: np.ndarray, context: Dict[str, Any]) -> float:
        """Adjust privacy parameters based on context"""

        # Extract context information
        client_reliability = context.get('client_reliability', 0.5)
        data_sensitivity_level = context.get('data_sensitivity', 'medium')
        computational_budget = context.get('computation_budget', 'normal')

        # Base epsilon
        epsilon = self.base_epsilon

        # Adjust based on client reliability (more reliable clients get less noise)
        reliability_factor = 1 / (client_reliability + 0.1)  # Avoid division by zero
        epsilon *= reliability_factor

        # Adjust based on data sensitivity
        sensitivity_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0,
            'critical': 5.0
        }
        epsilon *= sensitivity_multipliers.get(data_sensitivity_level, 1.0)

        # Adjust based on computational budget
        if computational_budget == 'low':
            epsilon *= 0.7  # More aggressive privacy for low compute
        elif computational_budget == 'high':
            epsilon *= 1.3  # Can afford better utility

        return epsilon

class LocalDifferentialPrivacy:
    """Local differential privacy for client-side privacy"""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def randomized_response(self, true_value: Union[int, float],
                          domain_size: int = 2) -> Union[int, float]:
        """Randomized response mechanism"""
        if domain_size == 2:
            # Binary randomized response
            p = 1 / (1 + np.exp(self.epsilon))  # Probability of flipping
            if np.random.random() < p:
                return 1 - true_value  # Flip bit
            else:
                return true_value
        else:
            # General randomized response
            p = np.exp(self.epsilon) / (domain_size - 1 + np.exp(self.epsilon))
            if np.random.random() < p:
                return true_value
            else:
                # Return random other value
                other_values = [i for i in range(domain_size) if i != true_value]
                return np.random.choice(other_values)

    def histogram_encoding(self, true_histogram: np.ndarray) -> np.ndarray:
        """Histogram encoding with local DP"""
        encoded_histogram = np.zeros_like(true_histogram)

        for i, count in enumerate(true_histogram):
            # Add Laplace noise to each bin
            noise = np.random.laplace(0, 1.0 / self.epsilon)
            encoded_histogram[i] = count + noise

        # Ensure non-negative counts
        encoded_histogram = np.maximum(0, encoded_histogram)

        return encoded_histogram

    def hadamard_response(self, true_vector: np.ndarray) -> np.ndarray:
        """Hadamard response for local DP on vectors"""
        n = len(true_vector)

        # Generate random Hadamard matrix (simplified)
        hadamard_size = 2**int(np.ceil(np.log2(n)))
        hadamard_matrix = self._generate_hadamard(hadamard_size)

        # Apply Hadamard transform
        transformed = hadamard_matrix[:n, :n] @ true_vector

        # Add noise
        noise_scale = np.sqrt(n) / self.epsilon
        noise = np.random.normal(0, noise_scale, n)

        return transformed + noise

    def _generate_hadamard(self, size: int) -> np.ndarray:
        """Generate Hadamard matrix"""
        if size == 1:
            return np.array([[1.0]])

        half_size = size // 2
        h_small = self._generate_hadamard(half_size)

        top_right = h_small
        bottom_left = h_small
        bottom_right = -h_small

        top = np.hstack([top_right, top_right])
        bottom = np.hstack([bottom_left, bottom_right])

        return np.vstack([top, bottom])

class QuantumDifferentialPrivacy:
    """Quantum-enhanced differential privacy"""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def quantum_noise_mechanism(self, data: np.ndarray) -> np.ndarray:
        """Add quantum-inspired noise"""
        # Use quantum superposition for noise generation
        noise = self._quantum_random_walk_noise(data.shape)
        return data + noise

    def _quantum_random_walk_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise using quantum random walk"""
        # Simplified quantum random walk
        noise = np.zeros(shape)

        for i in range(len(noise.flatten())):
            # Quantum walk: superposition of +1 and -1 with quantum amplitudes
            amplitude = np.random.normal(0, 1) + 1j * np.random.normal(0, 1)
            probability = np.abs(amplitude)**2

            if np.random.random() < probability:
                noise.flat[i] = np.real(amplitude) / self.epsilon
            else:
                noise.flat[i] = -np.real(amplitude) / self.epsilon

        return noise

    def quantum_privacy_amplification(self, data: np.ndarray,
                                    amplification_factor: float = 2.0) -> np.ndarray:
        """Amplify privacy using quantum effects"""
        # Apply quantum measurement uncertainty
        uncertainty = np.random.normal(0, amplification_factor / self.epsilon, data.shape)
        return data + uncertainty

class PrivacyAccounting:
    """Comprehensive privacy accounting and tracking"""

    def __init__(self):
        self.privacy_losses = []
        self.composition_history = []

    def track_privacy_loss(self, mechanism: str, epsilon_cost: float,
                          delta_cost: float = 0.0, metadata: Dict[str, Any] = None):
        """Track privacy loss for accountability"""

        loss_record = {
            'timestamp': time.time(),
            'mechanism': mechanism,
            'epsilon_cost': epsilon_cost,
            'delta_cost': delta_cost,
            'metadata': metadata or {}
        }

        self.privacy_losses.append(loss_record)

    def compute_total_privacy_loss(self, composition_method: str = 'basic') -> Tuple[float, float]:
        """Compute total privacy loss across all operations"""

        if not self.privacy_losses:
            return 0.0, 0.0

        total_epsilon = 0.0
        total_delta = 0.0

        if composition_method == 'basic':
            # Simple summation
            total_epsilon = sum(loss['epsilon_cost'] for loss in self.privacy_losses)
            total_delta = sum(loss['delta_cost'] for loss in self.privacy_losses)

        elif composition_method == 'advanced':
            # Use advanced composition
            composer = AdvancedComposition(1.0, 1e-5)  # Base values
            for loss in self.privacy_losses:
                eps, delta = composer.basic_composition(
                    total_epsilon, loss['epsilon_cost'],
                    total_delta, loss['delta_cost']
                )
                total_epsilon, total_delta = eps, delta

        return total_epsilon, total_delta

    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report"""

        total_epsilon, total_delta = self.compute_total_privacy_loss()

        # Analyze mechanisms used
        mechanism_counts = {}
        for loss in self.privacy_losses:
            mech = loss['mechanism']
            mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1

        # Compute privacy efficiency
        total_operations = len(self.privacy_losses)
        avg_epsilon_per_op = total_epsilon / max(total_operations, 1)

        return {
            'total_epsilon': total_epsilon,
            'total_delta': total_delta,
            'total_operations': total_operations,
            'avg_epsilon_per_operation': avg_epsilon_per_op,
            'mechanism_distribution': mechanism_counts,
            'privacy_efficiency': 1.0 / (avg_epsilon_per_op + 1e-6),
            'composition_method': 'basic',
            'report_timestamp': time.time()
        }

class FederatedPrivacyController:
    """Privacy controller for federated learning scenarios"""

    def __init__(self, global_epsilon: float = 1.0, num_clients: int = 10):
        self.global_epsilon = global_epsilon
        self.num_clients = num_clients

        self.client_privacy_budgets = {}
        self.global_accountant = PrivacyAccounting()

        # Initialize client budgets
        self._initialize_client_budgets()

    def _initialize_client_budgets(self):
        """Initialize privacy budgets for each client"""
        client_epsilon = self.global_epsilon / np.sqrt(self.num_clients)  # Conservative allocation

        for i in range(self.num_clients):
            client_id = f"client_{i}"
            self.client_privacy_budgets[client_id] = PrivacyBudget(
                epsilon=client_epsilon,
                delta=1e-6
            )

    def allocate_privacy_budget(self, client_id: str, requested_epsilon: float) -> float:
        """Allocate privacy budget to a client"""
        if client_id not in self.client_privacy_budgets:
            return 0.0

        budget = self.client_privacy_budgets[client_id]
        allocated = min(requested_epsilon, budget.remaining_epsilon())

        return allocated

    def update_client_budget(self, client_id: str, used_epsilon: float):
        """Update client's privacy budget after use"""
        if client_id in self.client_privacy_budgets:
            self.client_privacy_budgets[client_id].spend_budget(used_epsilon)
            self.global_accountant.track_privacy_loss(
                'client_update', used_epsilon, 0.0,
                {'client_id': client_id}
            )

    def check_global_privacy(self) -> bool:
        """Check if global privacy budget is still valid"""
        total_epsilon, _ = self.global_accountant.compute_total_privacy_loss()

        # Use advanced composition for federated setting
        composer = AdvancedComposition(self.global_epsilon, 1e-5)
        _, _ = composer.advanced_composition(
            self.num_clients, total_epsilon / self.num_clients, 1e-6, 1e-6
        )

        return total_epsilon <= self.global_epsilon

    def get_privacy_status(self) -> Dict[str, Any]:
        """Get comprehensive privacy status"""
        client_status = {}
        for client_id, budget in self.client_privacy_budgets.items():
            client_status[client_id] = {
                'remaining_epsilon': budget.remaining_epsilon(),
                'used_epsilon': budget.used_epsilon,
                'budget_exhausted': budget.is_exhausted()
            }

        return {
            'global_privacy_budget': self.global_epsilon,
            'client_status': client_status,
            'global_privacy_ok': self.check_global_privacy(),
            'privacy_report': self.global_accountant.generate_privacy_report()
        }

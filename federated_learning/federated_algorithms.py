#!/usr/bin/env python3
"""
Advanced Federated Learning Algorithms for Quantum Edge AI Platform

This module implements state-of-the-art federated learning algorithms
optimized for edge devices, including privacy-preserving techniques,
communication-efficient methods, and quantum-enhanced federated learning.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FederatedAlgorithm(ABC):
    """
    Abstract base class for federated learning algorithms.

    Defines the interface that all federated learning algorithms must implement.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_model = None
        self.clients = {}
        self.round_number = 0
        self.metrics_history = []

    @abstractmethod
    def initialize_global_model(self) -> Any:
        """Initialize the global model."""
        pass

    @abstractmethod
    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for the current round."""
        pass

    @abstractmethod
    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Any:
        """Aggregate model updates from clients."""
        pass

    @abstractmethod
    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate the global model."""
        pass

    def run_federated_round(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a single federated learning round."""
        start_time = time.time()

        # Aggregate updates
        new_global_model = self.aggregate_updates(client_updates)

        # Update global model
        self.global_model = new_global_model
        self.round_number += 1

        round_time = time.time() - start_time

        # Record metrics
        round_metrics = {
            'round': self.round_number,
            'round_time': round_time,
            'n_clients': len(client_updates),
            'communication_cost': self._calculate_communication_cost(client_updates)
        }

        self.metrics_history.append(round_metrics)

        return round_metrics

    def _calculate_communication_cost(self, client_updates: List[Dict[str, Any]]) -> float:
        """Calculate communication cost for the round."""
        total_cost = 0
        for update in client_updates:
            # Estimate size of model update
            if 'model_update' in update:
                update_size = len(pickle.dumps(update['model_update']))
                total_cost += update_size
        return total_cost


class FedAvg(FederatedAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm.

    The classic federated learning algorithm that averages model updates
    from selected clients.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)

    def initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model with random weights."""
        n_features = self.config.get('n_features', 10)
        n_classes = self.config.get('n_classes', 2)

        # Simple linear model
        model = {
            'weights': np.random.randn(n_features, n_classes) * 0.01,
            'bias': np.zeros(n_classes)
        }

        self.global_model = model
        return model

    def select_clients(self, round_number: int) -> List[str]:
        """Select random fraction of clients."""
        available_clients = list(self.clients.keys())
        n_select = max(1, int(len(available_clients) * self.client_selection_fraction))

        selected = np.random.choice(available_clients, n_select, replace=False)
        return selected.tolist()

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Aggregate model updates using federated averaging."""
        if not client_updates:
            return self.global_model

        # Extract model updates and sample sizes
        updates = []
        sample_sizes = []

        for update in client_updates:
            if 'model_update' in update and 'sample_size' in update:
                updates.append(update['model_update'])
                sample_sizes.append(update['sample_size'])

        if not updates:
            return self.global_model

        total_samples = sum(sample_sizes)

        # Weighted average of model updates
        aggregated_weights = np.zeros_like(updates[0]['weights'])
        aggregated_bias = np.zeros_like(updates[0]['bias'])

        for update, n_samples in zip(updates, sample_sizes):
            weight = n_samples / total_samples
            aggregated_weights += weight * update['weights']
            aggregated_bias += weight * update['bias']

        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias
        }

    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate global model on test data."""
        X_test, y_test = test_data

        # Simple linear prediction
        logits = X_test @ self.global_model['weights'] + self.global_model['bias']
        predictions = np.argmax(logits, axis=1)

        accuracy = np.mean(predictions == y_test)

        return {
            'accuracy': accuracy,
            'loss': -np.mean(logits[np.arange(len(y_test)), y_test])  # Negative log likelihood
        }


class FedProx(FederatedAlgorithm):
    """
    Federated Proximal (FedProx) algorithm.

    Adds a proximal term to the local loss function to improve stability
    and convergence in heterogeneous environments.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mu = config.get('proximal_mu', 0.01)  # Proximal term coefficient
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)

    def initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model."""
        return FedAvg.initialize_global_model(self)

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients with preference for those with more data."""
        available_clients = list(self.clients.keys())

        # Simple data-aware selection (prefer clients with more samples)
        client_weights = []
        for client_id in available_clients:
            client_info = self.clients[client_id]
            # Weight by log of sample size to avoid extreme preferences
            weight = np.log(client_info.get('sample_size', 1) + 1)
            client_weights.append(weight)

        client_weights = np.array(client_weights)
        client_weights = client_weights / client_weights.sum()

        n_select = max(1, int(len(available_clients) * self.client_selection_fraction))
        selected = np.random.choice(available_clients, n_select, p=client_weights, replace=False)

        return selected.tolist()

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Aggregate with proximal regularization."""
        # For FedProx, the aggregation is the same as FedAvg
        # The difference is in the local training (proximal term)
        return FedAvg.aggregate_updates(self, client_updates)

    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate global model."""
        return FedAvg.evaluate_global_model(self, test_data)


class FedNova(FederatedAlgorithm):
    """
    Federated Normalized Averaging (FedNova) algorithm.

    Normalizes client updates by local training steps to handle varying
    local training intensities.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)

    def initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model."""
        return FedAvg.initialize_global_model(self)

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients."""
        return FedAvg.select_clients(self, round_number)

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Aggregate with normalization by local steps."""
        if not client_updates:
            return self.global_model

        # FedNova normalization
        updates = []
        local_steps = []
        sample_sizes = []

        for update in client_updates:
            if all(key in update for key in ['model_update', 'local_steps', 'sample_size']):
                updates.append(update['model_update'])
                local_steps.append(update['local_steps'])
                sample_sizes.append(update['sample_size'])

        if not updates:
            return self.global_model

        total_samples = sum(sample_sizes)

        # Compute normalization coefficients
        # a_i = n_i / sum(n_j) * tau * rho
        # where tau is local steps, rho is FedNova coefficient
        rho = 1.0  # FedNova coefficient, can be tuned

        normalized_updates = []
        normalization_coeffs = []

        for update, tau, n_i in zip(updates, local_steps, sample_sizes):
            coeff = (n_i / total_samples) * tau * rho
            normalization_coeffs.append(coeff)

            # Normalize the update
            normalized_update = {
                'weights': update['weights'] * coeff,
                'bias': update['bias'] * coeff
            }
            normalized_updates.append(normalized_update)

        # Aggregate normalized updates
        total_coeff = sum(normalization_coeffs)
        if total_coeff == 0:
            return self.global_model

        aggregated_weights = sum(update['weights'] for update in normalized_updates) / total_coeff
        aggregated_bias = sum(update['bias'] for update in normalized_updates) / total_coeff

        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias
        }

    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate global model."""
        return FedAvg.evaluate_global_model(self, test_data)


class Scaffold(FederatedAlgorithm):
    """
    SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) algorithm.

    Uses control variates to reduce client-drift in heterogeneous environments.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)
        self.global_control_variate = None
        self.client_control_variates = {}

    def initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model and control variate."""
        model = FedAvg.initialize_global_model(self)

        # Initialize global control variate
        n_features = self.config.get('n_features', 10)
        n_classes = self.config.get('n_classes', 2)
        self.global_control_variate = {
            'weights': np.zeros((n_features, n_classes)),
            'bias': np.zeros(n_classes)
        }

        return model

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients."""
        return FedAvg.select_clients(self, round_number)

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Aggregate using SCAFFOLD with control variates."""
        if not client_updates:
            return self.global_model

        # Extract model updates and control variates
        model_updates = []
        local_control_variates = []
        sample_sizes = []

        for update in client_updates:
            if all(key in update for key in ['model_update', 'local_control_variate', 'sample_size']):
                model_updates.append(update['model_update'])
                local_control_variates.append(update['local_control_variate'])
                sample_sizes.append(update['sample_size'])

        if not model_updates:
            return self.global_model

        total_samples = sum(sample_sizes)

        # SCAFFOLD aggregation
        aggregated_weights = np.zeros_like(model_updates[0]['weights'])
        aggregated_bias = np.zeros_like(model_updates[0]['bias'])

        # Update global control variate
        new_global_control = {
            'weights': np.zeros_like(self.global_control_variate['weights']),
            'bias': np.zeros_like(self.global_control_variate['bias'])
        }

        for update, local_cv, n_samples in zip(model_updates, local_control_variates, sample_sizes):
            weight = n_samples / total_samples

            # Model aggregation with control variates
            corrected_weights = update['weights'] - local_cv['weights'] + self.global_control_variate['weights']
            corrected_bias = update['bias'] - local_cv['bias'] + self.global_control_variate['bias']

            aggregated_weights += weight * corrected_weights
            aggregated_bias += weight * corrected_bias

            # Control variate aggregation
            new_global_control['weights'] += weight * local_cv['weights']
            new_global_control['bias'] += weight * local_cv['bias']

        # Update global control variate
        self.global_control_variate = new_global_control

        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias
        }

    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate global model."""
        return FedAvg.evaluate_global_model(self, test_data)


class PrivacyPreservingFederated(FederatedAlgorithm):
    """
    Privacy-preserving federated learning with differential privacy.

    Adds differential privacy noise to model updates before aggregation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.epsilon = config.get('differential_privacy_epsilon', 1.0)
        self.delta = config.get('differential_privacy_delta', 1e-5)
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)

    def initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model."""
        return FedAvg.initialize_global_model(self)

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients."""
        return FedAvg.select_clients(self, round_number)

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Aggregate with differential privacy."""
        if not client_updates:
            return self.global_model

        # Apply differential privacy to each update
        private_updates = []
        for update in client_updates:
            if 'model_update' in update:
                private_update = self._add_differential_privacy_noise(update['model_update'])
                private_updates.append(private_update)

        if not private_updates:
            return self.global_model

        # Standard FedAvg aggregation on private updates
        total_samples = sum(update.get('sample_size', 1) for update in client_updates)

        aggregated_weights = np.zeros_like(private_updates[0]['weights'])
        aggregated_bias = np.zeros_like(private_updates[0]['bias'])

        for update, client_update in zip(private_updates, client_updates):
            n_samples = client_update.get('sample_size', 1)
            weight = n_samples / total_samples

            aggregated_weights += weight * update['weights']
            aggregated_bias += weight * update['bias']

        return {
            'weights': aggregated_weights,
            'bias': aggregated_bias
        }

    def _add_differential_privacy_noise(self, model_update: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add differential privacy noise to model update."""
        private_update = {}

        # Calculate sensitivity (L2 norm of gradients)
        for param_name, param_value in model_update.items():
            sensitivity = np.linalg.norm(param_value.flatten())

            # Gaussian mechanism: add noise scaled by sensitivity/epsilon
            noise_scale = sensitivity / self.epsilon
            noise = np.random.normal(0, noise_scale, param_value.shape)

            private_update[param_name] = param_value + noise

        return private_update

    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate global model."""
        return FedAvg.evaluate_global_model(self, test_data)


class QuantumFederated(FederatedAlgorithm):
    """
    Quantum-enhanced federated learning.

    Uses quantum circuits for model updates and aggregation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_qubits = config.get('n_qubits', 4)
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)

        # Initialize quantum components
        self._init_quantum_components()

    def _init_quantum_components(self):
        """Initialize quantum computing components."""
        try:
            from quantum_edge_ai.quantum_algorithms.quantum_circuits import VariationalQuantumCircuit
            self.vqc = VariationalQuantumCircuit(self.n_qubits, n_layers=2)
            self.quantum_available = True
        except ImportError:
            logger.warning("Quantum components not available, using classical fallback")
            self.quantum_available = False

    def initialize_global_model(self) -> Dict[str, Any]:
        """Initialize quantum-enhanced global model."""
        if self.quantum_available:
            # Quantum model: variational circuit parameters
            model = {
                'quantum_params': np.random.randn(self.vqc.n_parameters) * 0.1,
                'classical_head': {
                    'weights': np.random.randn(self.n_qubits, 2) * 0.1,
                    'bias': np.zeros(2)
                }
            }
        else:
            # Classical fallback
            model = FedAvg.initialize_global_model(self)

        self.global_model = model
        return model

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients with quantum capability preference."""
        available_clients = list(self.clients.keys())

        # Prefer clients with quantum capabilities
        quantum_clients = []
        classical_clients = []

        for client_id in available_clients:
            client_info = self.clients[client_id]
            if client_info.get('quantum_capable', False):
                quantum_clients.append(client_id)
            else:
                classical_clients.append(client_id)

        # Select from quantum clients first, then classical
        selected = []
        n_select = max(1, int(len(available_clients) * self.client_selection_fraction))

        # Try to select some quantum clients
        n_quantum_wanted = min(len(quantum_clients), max(1, n_select // 2))
        if quantum_clients:
            selected.extend(np.random.choice(quantum_clients, n_quantum_wanted, replace=False))

        # Fill remaining slots with classical clients
        remaining_slots = n_select - len(selected)
        if remaining_slots > 0 and classical_clients:
            n_classical = min(remaining_slots, len(classical_clients))
            selected.extend(np.random.choice(classical_clients, n_classical, replace=False))

        return selected

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quantum-enhanced updates."""
        if not client_updates:
            return self.global_model

        if self.quantum_available:
            return self._aggregate_quantum_updates(client_updates)
        else:
            # Classical fallback
            classical_updates = [{'model_update': update['model_update'],
                                'sample_size': update.get('sample_size', 1)}
                               for update in client_updates]
            return FedAvg.aggregate_updates(self, classical_updates)

    def _aggregate_quantum_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quantum model updates."""
        quantum_updates = []
        classical_updates = []
        sample_sizes = []

        for update in client_updates:
            if 'quantum_update' in update:
                quantum_updates.append(update['quantum_update'])
            if 'classical_update' in update:
                classical_updates.append(update['classical_update'])
            sample_sizes.append(update.get('sample_size', 1))

        total_samples = sum(sample_sizes)

        # Aggregate quantum parameters
        if quantum_updates:
            aggregated_quantum = np.zeros_like(quantum_updates[0])
            for update, n_samples in zip(quantum_updates, sample_sizes):
                weight = n_samples / total_samples
                aggregated_quantum += weight * update
        else:
            aggregated_quantum = self.global_model['quantum_params']

        # Aggregate classical head
        if classical_updates:
            aggregated_classical = FedAvg.aggregate_updates(
                self, [{'model_update': update, 'sample_size': n}
                      for update, n in zip(classical_updates, sample_sizes)]
            )
        else:
            aggregated_classical = self.global_model['classical_head']

        return {
            'quantum_params': aggregated_quantum,
            'classical_head': aggregated_classical
        }

    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """Evaluate quantum-enhanced global model."""
        # This would run quantum inference - simplified for demo
        X_test, y_test = test_data

        if self.quantum_available:
            # Mock quantum inference
            n_test = len(X_test)
            quantum_features = np.random.randn(n_test, self.n_qubits)
            logits = quantum_features @ self.global_model['classical_head']['weights'] + self.global_model['classical_head']['bias']
        else:
            # Classical fallback
            logits = X_test @ self.global_model['weights'] + self.global_model['bias']

        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == y_test)

        return {
            'accuracy': accuracy,
            'loss': -np.mean(logits[np.arange(len(y_test)), y_test])
        }


class FederatedLearningCoordinator:
    """
    Coordinator for federated learning experiments.

    Manages multiple federated learning algorithms, client simulation,
    and experiment orchestration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithms = {}
        self.clients = {}
        self.results = {}

        # Experiment configuration
        self.n_rounds = config.get('n_rounds', 10)
        self.n_clients = config.get('n_clients', 10)
        self.test_data = None

    def add_algorithm(self, name: str, algorithm: FederatedAlgorithm):
        """Add a federated learning algorithm to test."""
        self.algorithms[name] = algorithm

    def setup_clients(self):
        """Set up simulated federated clients."""
        logger.info(f"Setting up {self.n_clients} federated clients")

        for i in range(self.n_clients):
            client_info = {
                'client_id': f'client_{i+1}',
                'sample_size': np.random.randint(100, 1000),
                'compute_capacity': np.random.choice(['low', 'medium', 'high']),
                'quantum_capable': np.random.random() < 0.3,  # 30% have quantum capability
                'data_distribution': 'iid' if np.random.random() < 0.7 else 'non_iid'
            }

            self.clients[f'client_{i+1}'] = client_info

            # Register client with all algorithms
            for algorithm in self.algorithms.values():
                algorithm.clients[f'client_{i+1}'] = client_info

    def generate_test_data(self):
        """Generate test dataset."""
        n_features = self.config.get('n_features', 10)
        n_test_samples = self.config.get('n_test_samples', 1000)

        from sklearn.datasets import make_classification
        X_test, y_test = make_classification(
            n_samples=n_test_samples,
            n_features=n_features,
            n_classes=2,
            n_informative=max(2, n_features // 2),
            n_redundant=max(1, n_features // 3),
            random_state=42
        )

        self.test_data = (X_test, y_test)

    def run_experiment(self) -> Dict[str, Any]:
        """Run federated learning experiment comparing all algorithms."""
        logger.info("Starting federated learning experiment")
        logger.info(f"Algorithms: {list(self.algorithms.keys())}")
        logger.info(f"Rounds: {self.n_rounds}, Clients: {self.n_clients}")

        # Initialize all algorithms
        for name, algorithm in self.algorithms.items():
            algorithm.initialize_global_model()

        # Run federated learning rounds
        for round_num in range(self.n_rounds):
            logger.info(f"Starting round {round_num + 1}/{self.n_rounds}")

            round_results = {}

            for alg_name, algorithm in self.algorithms.items():
                # Select clients
                selected_clients = algorithm.select_clients(round_num)

                # Simulate client training (in practice, this would be distributed)
                client_updates = self._simulate_client_training(algorithm, selected_clients)

                # Run federated round
                round_metrics = algorithm.run_federated_round(client_updates)

                # Evaluate global model
                if self.test_data:
                    eval_metrics = algorithm.evaluate_global_model(self.test_data)
                    round_metrics.update(eval_metrics)

                round_results[alg_name] = round_metrics

            # Store round results
            for alg_name, metrics in round_results.items():
                if alg_name not in self.results:
                    self.results[alg_name] = []
                self.results[alg_name].append(metrics)

            # Log progress
            self._log_round_progress(round_num + 1, round_results)

        logger.info("Federated learning experiment completed")
        return self.results

    def _simulate_client_training(self, algorithm: FederatedAlgorithm, selected_clients: List[str]) -> List[Dict[str, Any]]:
        """Simulate local training on selected clients."""
        client_updates = []

        for client_id in selected_clients:
            client_info = self.clients[client_id]

            # Simulate local training
            local_update = self._generate_client_update(algorithm, client_info)

            update_data = {
                'client_id': client_id,
                'model_update': local_update,
                'sample_size': client_info['sample_size']
            }

            # Add algorithm-specific data
            if isinstance(algorithm, FedNova):
                update_data['local_steps'] = np.random.randint(5, 20)
            elif isinstance(algorithm, Scaffold):
                update_data['local_control_variate'] = self._generate_control_variate(algorithm)
            elif isinstance(algorithm, PrivacyPreservingFederated):
                # Privacy is added during aggregation
                pass
            elif isinstance(algorithm, QuantumFederated):
                if algorithm.quantum_available:
                    update_data['quantum_update'] = self._generate_quantum_update(algorithm)
                    update_data['classical_update'] = local_update

            client_updates.append(update_data)

        return client_updates

    def _generate_client_update(self, algorithm: FederatedAlgorithm, client_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate a simulated client model update."""
        # Simulate local training gradient
        n_features = self.config.get('n_features', 10)
        n_classes = self.config.get('n_classes', 2)

        # Add some noise to simulate different client data distributions
        noise_scale = 1.0 if client_info['data_distribution'] == 'iid' else 2.0

        update = {
            'weights': np.random.randn(n_features, n_classes) * 0.01 * noise_scale,
            'bias': np.random.randn(n_classes) * 0.01 * noise_scale
        }

        return update

    def _generate_control_variate(self, algorithm: Scaffold) -> Dict[str, np.ndarray]:
        """Generate control variate for SCAFFOLD."""
        n_features = self.config.get('n_features', 10)
        n_classes = self.config.get('n_classes', 2)

        return {
            'weights': np.random.randn(n_features, n_classes) * 0.001,
            'bias': np.random.randn(n_classes) * 0.001
        }

    def _generate_quantum_update(self, algorithm: QuantumFederated) -> np.ndarray:
        """Generate quantum parameter update."""
        return np.random.randn(algorithm.vqc.n_parameters) * 0.01

    def _log_round_progress(self, round_num: int, round_results: Dict[str, Dict[str, Any]]):
        """Log progress for the current round."""
        logger.info(f"Round {round_num} completed:")

        for alg_name, metrics in round_results.items():
            accuracy = metrics.get('accuracy', 'N/A')
            round_time = metrics.get('round_time', 0)
            n_clients = metrics.get('n_clients', 0)

            logger.info(f"  {alg_name}: accuracy={accuracy}, "
                       f"time={round_time:.2f}s, clients={n_clients}")

    def plot_results(self):
        """Plot experiment results."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Accuracy over rounds
            ax = axes[0, 0]
            for alg_name, rounds in self.results.items():
                accuracies = [r.get('accuracy', 0) for r in rounds]
                ax.plot(range(1, len(rounds) + 1), accuracies, label=alg_name, marker='o')
            ax.set_title('Model Accuracy Over Rounds')
            ax.set_xlabel('Round')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Round time comparison
            ax = axes[0, 1]
            algorithms = list(self.results.keys())
            avg_times = [np.mean([r.get('round_time', 0) for r in rounds])
                        for rounds in self.results.values()]
            ax.bar(algorithms, avg_times)
            ax.set_title('Average Round Time')
            ax.set_ylabel('Time (seconds)')
            plt.xticks(rotation=45, ha='right')

            # Communication cost
            ax = axes[1, 0]
            comm_costs = [sum(r.get('communication_cost', 0) for r in rounds)
                         for rounds in self.results.values()]
            ax.bar(algorithms, comm_costs)
            ax.set_title('Total Communication Cost')
            ax.set_ylabel('Cost (MB)')
            plt.xticks(rotation=45, ha='right')

            # Final accuracy comparison
            ax = axes[1, 1]
            final_accuracies = [rounds[-1].get('accuracy', 0) if rounds else 0
                              for rounds in self.results.values()]
            ax.bar(algorithms, final_accuracies)
            ax.set_title('Final Model Accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig('federated_learning_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

            logger.info("Results plot saved as 'federated_learning_comparison.png'")

        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def main():
    """Demonstrate federated learning algorithms."""
    print("🌐 Federated Learning Algorithms Demo")
    print("=" * 50)

    # Experiment configuration
    config = {
        'n_rounds': 5,
        'n_clients': 10,
        'n_features': 8,
        'n_classes': 2,
        'n_test_samples': 500,
        'client_selection_fraction': 0.3
    }

    # Create coordinator
    coordinator = FederatedLearningCoordinator(config)

    # Add algorithms to compare
    algorithms = {
        'FedAvg': FedAvg(config),
        'FedProx': FedProx({**config, 'proximal_mu': 0.01}),
        'FedNova': FedNova(config),
        'SCAFFOLD': Scaffold(config),
        'DP-FedAvg': PrivacyPreservingFederated({**config, 'differential_privacy_epsilon': 1.0}),
        'Quantum-Fed': QuantumFederated({**config, 'n_qubits': 4})
    }

    for name, algorithm in algorithms.items():
        coordinator.add_algorithm(name, algorithm)

    # Setup experiment
    coordinator.setup_clients()
    coordinator.generate_test_data()

    # Run experiment
    print("\\n🚀 Running federated learning experiment...")
    results = coordinator.run_experiment()

    # Display final results
    print("\\n📊 Final Results Summary")
    print("=" * 40)
    print("<12")
    print("-" * 55)

    for alg_name, rounds in results.items():
        if rounds:
            final_round = rounds[-1]
            accuracy = final_round.get('accuracy', 'N/A')
            total_time = sum(r.get('round_time', 0) for r in rounds)
            total_comm = sum(r.get('communication_cost', 0) for r in rounds)

            print("<12")

    # Plot results
    try:
        coordinator.plot_results()
    except Exception as e:
        print(f"\\n⚠️  Could not generate plots: {e}")

    print("\\n✅ Federated learning algorithms demo completed!")


if __name__ == "__main__":
    main()

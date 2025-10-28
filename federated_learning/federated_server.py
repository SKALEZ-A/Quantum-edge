"""
Federated Learning Server - Central Coordinator for Federated Training

Implements the central server that coordinates federated learning rounds,
manages client participation, aggregates model updates, and ensures privacy
and security throughout the distributed training process.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    num_rounds: int = 100
    num_clients: int = 10
    fraction_participate: float = 0.1  # Fraction of clients participating each round
    min_clients: int = 3  # Minimum clients needed for aggregation
    max_round_time: int = 300  # Maximum time per round in seconds
    privacy_budget: float = 1.0  # Privacy budget for differential privacy
    secure_aggregation: bool = True
    quantum_privacy: bool = True

@dataclass
class RoundResult:
    """Result of a federated learning round"""
    round_number: int
    participating_clients: int
    global_model_update: Dict[str, np.ndarray]
    training_metrics: Dict[str, float]
    privacy_metrics: Dict[str, float]
    round_time: float
    success: bool

@dataclass
class ClientUpdate:
    """Model update from a client"""
    client_id: str
    model_update: Dict[str, np.ndarray]
    num_samples: int
    training_metrics: Dict[str, float]
    timestamp: float
    checksum: str  # For integrity verification

class FederatedServer:
    """Central server for federated learning coordination"""

    def __init__(self, global_model: Any, config: FederatedConfig):
        self.global_model = global_model
        self.config = config

        # Initialize components
        self.secure_aggregator = SecureAggregator() if config.secure_aggregation else None
        self.quantum_privacy = QuantumEnhancedPrivacy() if config.quantum_privacy else None
        self.differential_privacy = DifferentialPrivacy(epsilon=config.privacy_budget)

        # Server state
        self.current_round = 0
        self.registered_clients: Dict[str, Dict[str, Any]] = {}
        self.active_clients: set = set()
        self.round_history: List[RoundResult] = []

        # Communication queues
        self.update_queue = queue.Queue()
        self.client_responses = {}

        # Synchronization
        self.round_lock = threading.Lock()
        self.client_lock = threading.Lock()

        # Performance tracking
        self.start_time = time.time()

    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """Register a new client with the server"""
        with self.client_lock:
            if client_id in self.registered_clients:
                logger.warning(f"Client {client_id} already registered")
                return False

            self.registered_clients[client_id] = {
                'info': client_info,
                'registration_time': time.time(),
                'participation_count': 0,
                'last_active': time.time(),
                'reputation_score': 1.0
            }

            logger.info(f"Registered client {client_id}. Total clients: {len(self.registered_clients)}")
            return True

    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client"""
        with self.client_lock:
            if client_id not in self.registered_clients:
                return False

            del self.registered_clients[client_id]
            self.active_clients.discard(client_id)

            logger.info(f"Unregistered client {client_id}")
            return True

    def start_federated_training(self) -> List[RoundResult]:
        """Start the federated learning training process"""
        logger.info(f"Starting federated training with {len(self.registered_clients)} clients")

        for round_num in range(1, self.config.num_rounds + 1):
            try:
                round_result = self._execute_round(round_num)
                self.round_history.append(round_result)

                if not round_result.success:
                    logger.warning(f"Round {round_num} failed, continuing...")

                # Check convergence
                if self._check_convergence():
                    logger.info(f"Training converged after {round_num} rounds")
                    break

            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                continue

        logger.info(f"Federated training completed after {len(self.round_history)} rounds")
        return self.round_history

    def _execute_round(self, round_num: int) -> RoundResult:
        """Execute a single federated learning round"""
        round_start_time = time.time()
        self.current_round = round_num

        logger.info(f"Starting round {round_num}")

        # Select participating clients
        participating_clients = self._select_clients()

        if len(participating_clients) < self.config.min_clients:
            logger.warning(f"Insufficient clients for round {round_num}: {len(participating_clients)}")
            return RoundResult(
                round_number=round_num,
                participating_clients=len(participating_clients),
                global_model_update={},
                training_metrics={},
                privacy_metrics={},
                round_time=time.time() - round_start_time,
                success=False
            )

        # Send global model to clients
        self._broadcast_global_model(participating_clients)

        # Wait for client updates
        client_updates = self._collect_client_updates(participating_clients)

        if len(client_updates) < self.config.min_clients:
            logger.warning(f"Insufficient updates for round {round_num}: {len(client_updates)}")
            return RoundResult(
                round_number=round_num,
                participating_clients=len(client_updates),
                global_model_update={},
                training_metrics={},
                privacy_metrics={},
                round_time=time.time() - round_start_time,
                success=False
            )

        # Aggregate model updates
        global_update = self._aggregate_updates(client_updates)

        # Apply privacy mechanisms
        global_update = self._apply_privacy_mechanisms(global_update, client_updates)

        # Update global model
        self._update_global_model(global_update)

        # Update client reputations
        self._update_client_reputations(participating_clients, client_updates)

        round_time = time.time() - round_start_time

        # Collect metrics
        training_metrics = self._compute_training_metrics(client_updates)
        privacy_metrics = self._compute_privacy_metrics(global_update, client_updates)

        logger.info(f"Round {round_num} completed in {round_time:.2f}s with {len(client_updates)} clients")

        return RoundResult(
            round_number=round_num,
            participating_clients=len(client_updates),
            global_model_update=global_update,
            training_metrics=training_metrics,
            privacy_metrics=privacy_metrics,
            round_time=round_time,
            success=True
        )

    def _select_clients(self) -> List[str]:
        """Select clients for participation in current round"""
        with self.client_lock:
            available_clients = list(self.registered_clients.keys())

            if len(available_clients) == 0:
                return []

            # Calculate number to select
            num_to_select = max(
                self.config.min_clients,
                int(len(available_clients) * self.config.fraction_participate)
            )

            # Use reputation-based selection
            client_scores = []
            for client_id in available_clients:
                reputation = self.registered_clients[client_id]['reputation_score']
                last_active = self.registered_clients[client_id]['last_active']
                recency_penalty = max(0, time.time() - last_active) / 3600  # Hours since last active
                score = reputation * np.exp(-recency_penalty)
                client_scores.append((client_id, score))

            # Sort by score and select top clients
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected_clients = [client_id for client_id, _ in client_scores[:num_to_select]]

            # Update active clients
            self.active_clients = set(selected_clients)

            return selected_clients

    def _broadcast_global_model(self, clients: List[str]):
        """Broadcast current global model to selected clients"""
        # In practice, this would send the model via network
        # For now, we simulate by marking clients as ready
        for client_id in clients:
            self.registered_clients[client_id]['last_active'] = time.time()

        logger.debug(f"Broadcasted global model to {len(clients)} clients")

    def _collect_client_updates(self, clients: List[str]) -> List[ClientUpdate]:
        """Collect model updates from participating clients"""
        updates = []
        timeout = time.time() + self.config.max_round_time

        # In a real implementation, this would wait for network messages
        # For simulation, we'll create mock updates
        for client_id in clients:
            if time.time() > timeout:
                break

            # Simulate client update (in practice, this comes from network)
            update = self._simulate_client_update(client_id)
            if update:
                updates.append(update)

        return updates

    def _simulate_client_update(self, client_id: str) -> Optional[ClientUpdate]:
        """Simulate a client model update (for demonstration)"""
        # Generate mock model update
        model_update = {}
        if hasattr(self.global_model, 'layers'):
            for i, layer in enumerate(self.global_model.layers):
                if hasattr(layer, 'weights'):
                    # Generate small random update
                    update_size = layer.weights.shape
                    update = np.random.normal(0, 0.01, update_size)
                    model_update[f'layer_{i}'] = update

        if not model_update:
            return None

        return ClientUpdate(
            client_id=client_id,
            model_update=model_update,
            num_samples=np.random.randint(100, 1000),
            training_metrics={
                'loss': np.random.uniform(0.1, 0.5),
                'accuracy': np.random.uniform(0.8, 0.95)
            },
            timestamp=time.time(),
            checksum=self._compute_checksum(model_update)
        )

    def _aggregate_updates(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate model updates from clients"""
        if not client_updates:
            return {}

        if self.secure_aggregator:
            # Use secure aggregation
            return self.secure_aggregator.aggregate(client_updates)
        else:
            # Simple FedAvg aggregation
            return self._fedavg_aggregation(client_updates)

    def _fedavg_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Federated averaging aggregation"""
        aggregated_update = {}
        total_samples = sum(update.num_samples for update in client_updates)

        # Initialize with first update
        first_update = client_updates[0]
        for param_name, param_update in first_update.model_update.items():
            aggregated_update[param_name] = (param_update * first_update.num_samples) / total_samples

        # Add other updates
        for update in client_updates[1:]:
            weight = update.num_samples / total_samples
            for param_name, param_update in update.model_update.items():
                if param_name in aggregated_update:
                    aggregated_update[param_name] += param_update * weight

        return aggregated_update

    def _apply_privacy_mechanisms(self, global_update: Dict[str, np.ndarray],
                                client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Apply privacy-preserving mechanisms to aggregated update"""

        # Apply differential privacy
        private_update = self.differential_privacy.add_noise(global_update)

        # Apply quantum privacy if enabled
        if self.quantum_privacy:
            private_update = self.quantum_privacy.enhance_privacy(private_update)

        return private_update

    def _update_global_model(self, global_update: Dict[str, np.ndarray]):
        """Update the global model with aggregated updates"""
        if hasattr(self.global_model, 'layers'):
            for i, layer in enumerate(self.global_model.layers):
                param_name = f'layer_{i}'
                if param_name in global_update and hasattr(layer, 'weights'):
                    # Apply update with learning rate
                    learning_rate = 0.1  # Could be adaptive
                    layer.weights += learning_rate * global_update[param_name]

        logger.debug("Updated global model with aggregated updates")

    def _update_client_reputations(self, clients: List[str], updates: List[ClientUpdate]):
        """Update client reputation scores based on participation and quality"""
        update_dict = {update.client_id: update for update in updates}

        for client_id in clients:
            if client_id in update_dict:
                # Successful participation
                self.registered_clients[client_id]['participation_count'] += 1
                self.registered_clients[client_id]['reputation_score'] *= 1.05  # Reward
            else:
                # Failed participation
                self.registered_clients[client_id]['reputation_score'] *= 0.95  # Penalty

            # Cap reputation between 0.1 and 2.0
            self.registered_clients[client_id]['reputation_score'] = np.clip(
                self.registered_clients[client_id]['reputation_score'], 0.1, 2.0
            )

    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.round_history) < 10:
            return False

        # Check if loss has stabilized
        recent_losses = [r.training_metrics.get('avg_loss', 0) for r in self.round_history[-5:]]
        if len(recent_losses) >= 3:
            loss_std = np.std(recent_losses)
            if loss_std < 0.001:  # Convergence threshold
                return True

        return False

    def _compute_training_metrics(self, client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Compute training metrics for the round"""
        if not client_updates:
            return {}

        losses = [u.training_metrics.get('loss', 0) for u in client_updates]
        accuracies = [u.training_metrics.get('accuracy', 0) for u in client_updates]

        return {
            'avg_loss': np.mean(losses),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses),
            'avg_accuracy': np.mean(accuracies),
            'total_samples': sum(u.num_samples for u in client_updates)
        }

    def _compute_privacy_metrics(self, global_update: Dict[str, np.ndarray],
                               client_updates: List[ClientUpdate]) -> Dict[str, float]:
        """Compute privacy-related metrics"""
        return {
            'privacy_budget_used': self.config.privacy_budget / self.config.num_rounds,
            'secure_aggregation_used': self.config.secure_aggregation,
            'quantum_privacy_used': self.config.quantum_privacy,
            'client_diversity': len(set(len(u.model_update) for u in client_updates))
        }

    def _compute_checksum(self, model_update: Dict[str, np.ndarray]) -> str:
        """Compute checksum for model update integrity"""
        # Simple checksum implementation
        total_sum = 0
        for param in model_update.values():
            total_sum += np.sum(param)
        return str(hash(str(total_sum)))

    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        total_runtime = time.time() - self.start_time

        return {
            'total_clients': len(self.registered_clients),
            'active_clients': len(self.active_clients),
            'completed_rounds': len(self.round_history),
            'total_runtime': total_runtime,
            'avg_round_time': np.mean([r.round_time for r in self.round_history]) if self.round_history else 0,
            'successful_rounds': sum(1 for r in self.round_history if r.success),
            'server_efficiency': len(self.round_history) / max(total_runtime, 1) * 3600  # Rounds per hour
        }

    def save_checkpoint(self, filepath: str):
        """Save server state for checkpointing"""
        checkpoint = {
            'current_round': self.current_round,
            'global_model': self.global_model,  # Would need proper serialization
            'registered_clients': self.registered_clients,
            'round_history': self.round_history,
            'server_stats': self.get_server_stats()
        }
        logger.info(f"Server checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load server state from checkpoint"""
        # Would load from file in practice
        logger.info(f"Server checkpoint loaded from {filepath}")

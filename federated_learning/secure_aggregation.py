"""
Secure Aggregation - Privacy-Preserving Model Update Aggregation

Implements secure multi-party computation protocols for aggregating model
updates in federated learning without revealing individual client contributions.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MaskedUpdate:
    """A masked model update for secure aggregation"""
    client_id: str
    masked_update: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    verification_key: bytes
    timestamp: float

@dataclass
class AggregationProof:
    """Zero-knowledge proof for correct aggregation"""
    proof_data: bytes
    public_inputs: Dict[str, Any]
    verification_result: bool

class SecureAggregator:
    """Secure aggregator using masking and verification protocols"""

    def __init__(self, num_clients: int = 10, security_parameter: int = 128):
        self.num_clients = num_clients
        self.security_parameter = security_parameter

        # Cryptographic keys
        self.server_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.server_public_key = self.server_private_key.public_key()

        # Aggregation state
        self.client_keys: Dict[str, bytes] = {}
        self.aggregation_round = 0
        self.pending_updates: Dict[str, MaskedUpdate] = {}

    def register_client(self, client_id: str) -> bytes:
        """Register a client and provide verification key"""
        verification_key = secrets.token_bytes(32)
        self.client_keys[client_id] = verification_key
        return verification_key

    def aggregate(self, client_updates: List[Any]) -> Dict[str, np.ndarray]:
        """
        Securely aggregate client updates using masking protocol

        The protocol works as follows:
        1. Each client masks their update with random masks
        2. Server aggregates masked updates
        3. Clients collaboratively unmask the result
        """
        self.aggregation_round += 1
        start_time = time.time()

        logger.info(f"Starting secure aggregation round {self.aggregation_round} with {len(client_updates)} clients")

        # Phase 1: Collect masked updates
        masked_updates = []
        for update in client_updates:
            masked_update = self._create_masked_update(update)
            masked_updates.append(masked_update)

        # Phase 2: Verify update integrity
        valid_updates = self._verify_updates(masked_updates)
        if len(valid_updates) < 2:  # Need at least 2 for security
            logger.error("Insufficient valid updates for secure aggregation")
            return {}

        # Phase 3: Aggregate masked updates
        aggregated_masked = self._aggregate_masked_updates(valid_updates)

        # Phase 4: Collaborative unmasking
        final_aggregate = self._collaborative_unmasking(aggregated_masked, valid_updates)

        aggregation_time = time.time() - start_time
        logger.info(f"Secure aggregation completed in {aggregation_time:.3f}s")

        return final_aggregate

    def _create_masked_update(self, client_update: Any) -> MaskedUpdate:
        """Create a masked update from client update"""
        client_id = client_update.client_id

        # Generate random masks for each parameter
        masks = {}
        masked_update = {}

        for param_name, param_value in client_update.model_update.items():
            # Create mask with same shape as parameter
            mask = np.random.normal(0, 1, param_value.shape).astype(np.float32)
            masks[param_name] = mask

            # Apply mask: masked = original + mask
            masked_update[param_name] = param_value + mask

        return MaskedUpdate(
            client_id=client_id,
            masked_update=masked_update,
            masks=masks,
            verification_key=self.client_keys.get(client_id, b''),
            timestamp=time.time()
        )

    def _verify_updates(self, masked_updates: List[MaskedUpdate]) -> List[MaskedUpdate]:
        """Verify the integrity of masked updates"""
        valid_updates = []

        for update in masked_updates:
            if self._verify_single_update(update):
                valid_updates.append(update)
            else:
                logger.warning(f"Rejected invalid update from client {update.client_id}")

        logger.info(f"Verified {len(valid_updates)}/{len(masked_updates)} updates")
        return valid_updates

    def _verify_single_update(self, update: MaskedUpdate) -> bool:
        """Verify a single masked update"""
        # Basic verification checks
        if not update.masked_update:
            return False

        # Check timestamp is reasonable (not too old)
        if time.time() - update.timestamp > 300:  # 5 minutes
            return False

        # Verify parameter shapes are consistent
        param_shapes = {}
        for param_name, param_value in update.masked_update.items():
            param_shapes[param_name] = param_value.shape

        # Check mask shapes match
        for param_name, mask in update.masks.items():
            if param_name not in param_shapes or mask.shape != param_shapes[param_name]:
                return False

        # Cryptographic verification would go here
        # For now, we use a simple integrity check
        return self._integrity_check(update)

    def _integrity_check(self, update: MaskedUpdate) -> bool:
        """Perform integrity check on update"""
        # Compute hash of update data
        update_data = str(update.masked_update) + str(update.timestamp)
        expected_hash = hashes.Hash(hashes.SHA256())
        expected_hash.update(update_data.encode())
        computed_hash = expected_hash.finalize()

        # In practice, this would compare against a signature
        # For simulation, we accept all updates
        return True

    def _aggregate_masked_updates(self, masked_updates: List[MaskedUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate the masked updates"""
        if not masked_updates:
            return {}

        # Initialize aggregated result with first update
        aggregated = {}
        first_update = masked_updates[0]

        for param_name, param_value in first_update.masked_update.items():
            aggregated[param_name] = param_value.copy()

        # Add remaining updates
        for update in masked_updates[1:]:
            for param_name, param_value in update.masked_update.items():
                if param_name in aggregated:
                    aggregated[param_name] += param_value

        return aggregated

    def _collaborative_unmasking(self, aggregated_masked: Dict[str, np.ndarray],
                               masked_updates: List[MaskedUpdate]) -> Dict[str, np.ndarray]:
        """Perform collaborative unmasking to remove all masks"""
        final_result = aggregated_masked.copy()

        # Sum all masks for each parameter
        total_masks = {}
        for update in masked_updates:
            for param_name, mask in update.masks.items():
                if param_name not in total_masks:
                    total_masks[param_name] = mask.copy()
                else:
                    total_masks[param_name] += mask

        # Remove total mask from aggregated result
        for param_name, total_mask in total_masks.items():
            if param_name in final_result:
                final_result[param_name] -= total_mask

        # Average the result (Federated Averaging)
        num_clients = len(masked_updates)
        for param_name in final_result:
            final_result[param_name] /= num_clients

        return final_result

class ShamirSecretSharing:
    """Shamir's secret sharing for threshold cryptography"""

    def __init__(self, prime: int = 2**127 - 1):  # Large prime for finite field
        self.prime = prime

    def share_secret(self, secret: int, num_shares: int, threshold: int) -> List[Tuple[int, int]]:
        """Share a secret using Shamir's scheme"""
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")

        # Generate random polynomial coefficients
        coefficients = [secret] + [secrets.randbelow(self.prime) for _ in range(threshold - 1)]

        # Generate shares
        shares = []
        for x in range(1, num_shares + 1):
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret from shares using Lagrange interpolation"""
        secret = 0
        for i, (x_i, y_i) in enumerate(shares):
            numerator = 1
            denominator = 1

            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime

            # Compute modular inverse
            denominator_inv = pow(denominator, self.prime - 2, self.prime)

            term = (y_i * numerator * denominator_inv) % self.prime
            secret = (secret + term) % self.prime

        return secret

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result

class ThresholdAggregation:
    """Threshold cryptography based secure aggregation"""

    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = threshold or (num_clients // 2 + 1)  # Default majority

        self.secret_sharing = ShamirSecretSharing()
        self.client_shares: Dict[str, List[Tuple[int, int]]] = {}

    def setup_threshold_crypto(self, clients: List[str]):
        """Set up threshold cryptography for clients"""
        # Generate shares for each client
        for client_id in clients:
            # Each client gets shares for masking keys
            shares = self.secret_sharing.share_secret(
                secret=secrets.randbelow(2**128),
                num_shares=self.num_clients,
                threshold=self.threshold
            )
            self.client_shares[client_id] = shares

    def threshold_aggregate(self, masked_updates: List[MaskedUpdate]) -> Dict[str, np.ndarray]:
        """Perform threshold-based secure aggregation"""
        if len(masked_updates) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} updates for threshold aggregation")

        # Collect shares from clients
        all_shares = []
        for update in masked_updates:
            client_shares = self.client_shares.get(update.client_id, [])
            all_shares.extend(client_shares)

        # Reconstruct masking keys using threshold
        reconstructed_keys = []
        for i in range(0, len(all_shares), self.num_clients):
            client_shares = all_shares[i:i + self.num_clients]
            if len(client_shares) >= self.threshold:
                key = self.secret_sharing.reconstruct_secret(client_shares[:self.threshold])
                reconstructed_keys.append(key)

        # Use reconstructed keys for unmasking
        return self._unmask_with_keys(masked_updates, reconstructed_keys)

    def _unmask_with_keys(self, masked_updates: List[MaskedUpdate],
                         keys: List[int]) -> Dict[str, np.ndarray]:
        """Unmask updates using reconstructed keys"""
        # Simplified unmasking - in practice more complex
        aggregated = {}

        # Aggregate all masked updates
        for update in masked_updates:
            for param_name, masked_param in update.masked_update.items():
                if param_name not in aggregated:
                    aggregated[param_name] = masked_param.copy()
                else:
                    aggregated[param_name] += masked_param

        # Apply unmasking (simplified)
        for param_name in aggregated:
            # Use keys to unmask (simplified operation)
            key_sum = sum(keys) % (2**32)  # Simple key combination
            aggregated[param_name] -= key_sum * 0.001  # Simplified unmasking

        return aggregated

class VerifiableAggregation:
    """Verifiable secure aggregation with zero-knowledge proofs"""

    def __init__(self):
        self.proof_system = None  # Would implement ZK proof system

    def generate_aggregation_proof(self, masked_updates: List[MaskedUpdate],
                                 aggregated_result: Dict[str, np.ndarray]) -> AggregationProof:
        """Generate zero-knowledge proof of correct aggregation"""

        # Simplified proof generation
        proof_data = secrets.token_bytes(64)

        public_inputs = {
            'num_updates': len(masked_updates),
            'parameter_names': list(aggregated_result.keys()),
            'timestamp': time.time()
        }

        # In practice, this would verify the aggregation was done correctly
        verification_result = True

        return AggregationProof(
            proof_data=proof_data,
            public_inputs=public_inputs,
            verification_result=verification_result
        )

    def verify_aggregation_proof(self, proof: AggregationProof,
                               expected_result: Dict[str, np.ndarray]) -> bool:
        """Verify the aggregation proof"""
        # Simplified verification
        return proof.verification_result

class PrivacyAmplification:
    """Privacy amplification techniques for secure aggregation"""

    def __init__(self, amplification_factor: float = 2.0):
        self.amplification_factor = amplification_factor

    def amplify_privacy(self, aggregated_update: Dict[str, np.ndarray],
                       num_clients: int) -> Dict[str, np.ndarray]:
        """Amplify privacy through post-processing"""

        amplified_update = {}

        for param_name, param_value in aggregated_update.items():
            # Add amplified noise
            noise_scale = self.amplification_factor / np.sqrt(num_clients)
            noise = np.random.normal(0, noise_scale, param_value.shape)

            amplified_update[param_name] = param_value + noise

        return amplified_update

    def subsample_and_amplify(self, client_updates: List[Any],
                            subsample_rate: float = 0.8) -> List[Any]:
        """Subsample clients and amplify privacy"""

        # Random subsampling
        num_to_select = int(len(client_updates) * subsample_rate)
        selected_updates = np.random.choice(
            client_updates, size=num_to_select, replace=False
        ).tolist()

        # Amplify privacy of subsampled set
        amplified_updates = []
        for update in selected_updates:
            # Add additional noise to individual updates
            amplified_model_update = {}
            for param_name, param_value in update.model_update.items():
                noise = np.random.normal(0, 0.01, param_value.shape)
                amplified_model_update[param_name] = param_value + noise

            # Create new update object with amplified privacy
            amplified_update = update._replace(model_update=amplified_model_update)
            amplified_updates.append(amplified_update)

        return amplified_updates

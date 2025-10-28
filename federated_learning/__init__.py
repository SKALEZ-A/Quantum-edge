"""
Federated Learning Framework - Privacy-Preserving Distributed AI

This module implements federated learning protocols with quantum-enhanced
privacy, secure aggregation, differential privacy, and decentralized model
synchronization for edge AI applications.
"""

__version__ = "1.0.0"
__author__ = "Quantum Edge AI Team"

from .federated_server import FederatedServer
from .federated_client import FederatedClient
from .secure_aggregation import SecureAggregator
from .differential_privacy import DifferentialPrivacy
from .quantum_privacy import QuantumEnhancedPrivacy
from .model_sync import DecentralizedModelSync

__all__ = [
    'FederatedServer',
    'FederatedClient',
    'SecureAggregator',
    'DifferentialPrivacy',
    'QuantumEnhancedPrivacy',
    'DecentralizedModelSync'
]

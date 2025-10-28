"""
Quantum-Enhanced Edge AI Platform
Quantum Algorithms Module

This module implements quantum-inspired algorithms and quantum computing
techniques for optimization, machine learning, and edge computing applications.
"""

__version__ = "1.0.0"
__author__ = "Quantum Edge AI Team"

from .quantum_optimizer import QuantumOptimizer
from .vqe_solver import VariationalQuantumEigensolver
from .qaoa_solver import QuantumApproximateOptimizationAlgorithm
from .quantum_ml import QuantumMachineLearning
from .quantum_feature_maps import QuantumFeatureMaps
from .hybrid_algorithms import HybridClassicalQuantum

__all__ = [
    'QuantumOptimizer',
    'VariationalQuantumEigensolver',
    'QuantumApproximateOptimizationAlgorithm',
    'QuantumMachineLearning',
    'QuantumFeatureMaps',
    'HybridClassicalQuantum'
]

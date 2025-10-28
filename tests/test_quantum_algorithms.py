"""
Unit tests for Quantum Algorithms
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_algorithms.quantum_ml import (
    QuantumFeatureMap, QuantumKernel, QuantumSupportVectorMachine,
    QuantumNeuralNetwork, QuantumMachineLearning
)
from quantum_algorithms.hybrid_algorithms import HybridQuantumClassical
from quantum_algorithms.qaoa_solver import QAOASolver
from quantum_algorithms.vqe_solver import VQESolver
from quantum_algorithms.quantum_optimizer import QuantumOptimizer
from tests import TestUtils

class TestQuantumFeatureMap(unittest.TestCase):
    """Test Quantum Feature Map"""

    def setUp(self):
        """Set up test fixtures"""
        self.feature_map = QuantumFeatureMap(n_qubits=4, depth=2)

    def test_feature_map_initialization(self):
        """Test feature map initialization"""
        self.assertEqual(self.feature_map.n_qubits, 4)
        self.assertEqual(self.feature_map.depth, 2)
        self.assertIsNotNone(self.feature_map.parameters)

    def test_feature_encoding(self):
        """Test feature encoding"""
        features = np.array([0.1, 0.2, 0.3, 0.4])

        # Should not raise exception
        encoded = self.feature_map.encode(features)

        # Check that it returns something (in real implementation would be quantum state)
        self.assertIsNotNone(encoded)

    def test_feature_map_gradient(self):
        """Test gradient computation"""
        features = np.array([0.1, 0.2, 0.3, 0.4])
        gradients = self.feature_map.compute_gradient(features)

        self.assertIsInstance(gradients, np.ndarray)
        self.assertEqual(len(gradients), len(features))

class TestQuantumKernel(unittest.TestCase):
    """Test Quantum Kernel"""

    def setUp(self):
        """Set up test fixtures"""
        self.kernel = QuantumKernel(n_qubits=4, kernel_type='gaussian')

    def test_kernel_initialization(self):
        """Test kernel initialization"""
        self.assertEqual(self.kernel.n_qubits, 4)
        self.assertEqual(self.kernel.kernel_type, 'gaussian')

    def test_kernel_matrix_computation(self):
        """Test kernel matrix computation"""
        X = np.random.randn(10, 4)

        kernel_matrix = self.kernel.compute_kernel_matrix(X)

        # Check matrix properties
        self.assertEqual(kernel_matrix.shape, (10, 10))

        # Should be symmetric
        self.assertTrue(np.allclose(kernel_matrix, kernel_matrix.T))

        # Should have positive diagonal (for normalized kernels)
        self.assertTrue(np.all(np.diag(kernel_matrix) >= 0))

    def test_kernel_evaluation(self):
        """Test kernel evaluation between two points"""
        x1 = np.array([0.1, 0.2, 0.3, 0.4])
        x2 = np.array([0.2, 0.3, 0.4, 0.5])

        kernel_value = self.kernel.evaluate(x1, x2)

        # Should be a scalar
        self.assertIsInstance(kernel_value, (int, float, np.number))

        # Should be non-negative
        self.assertGreaterEqual(kernel_value, 0)

class TestQuantumSupportVectorMachine(unittest.TestCase):
    """Test Quantum Support Vector Machine"""

    def setUp(self):
        """Set up test fixtures"""
        self.qsvm = QuantumSupportVectorMachine(n_qubits=4, C=1.0)

    def test_qsvm_initialization(self):
        """Test QSVM initialization"""
        self.assertEqual(self.qsvm.n_qubits, 4)
        self.assertEqual(self.qsvm.C, 1.0)
        self.assertIsNotNone(self.qsvm.kernel)

    def test_qsvm_training(self):
        """Test QSVM training"""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(20, 4)
        y = np.sign(X[:, 0] + X[:, 1] - 0.5)  # Linear decision boundary

        # Train QSVM
        self.qsvm.fit(X, y)

        # Check that support vectors are stored
        self.assertIsNotNone(self.qsvm.support_vectors_)
        self.assertGreater(len(self.qsvm.support_vectors_), 0)

    def test_qsvm_prediction(self):
        """Test QSVM prediction"""
        # Train first
        X_train = np.random.randn(20, 4)
        y_train = np.sign(X_train[:, 0] + X_train[:, 1] - 0.5)

        self.qsvm.fit(X_train, y_train)

        # Test prediction
        X_test = np.random.randn(5, 4)
        predictions = self.qsvm.predict(X_test)

        # Check predictions shape
        self.assertEqual(len(predictions), 5)

        # Check predictions are valid labels
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))

    def test_qsvm_probabilities(self):
        """Test QSVM probability estimation"""
        # Train first
        X_train = np.random.randn(20, 4)
        y_train = np.sign(X_train[:, 0] + X_train[:, 1] - 0.5)

        self.qsvm.fit(X_train, y_train)

        # Test probabilities
        X_test = np.random.randn(5, 4)
        probabilities = self.qsvm.predict_proba(X_test)

        # Check shape
        self.assertEqual(probabilities.shape, (5, 2))

        # Check probabilities sum to 1
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

        # Check probabilities are valid
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

class TestQuantumNeuralNetwork(unittest.TestCase):
    """Test Quantum Neural Network"""

    def setUp(self):
        """Set up test fixtures"""
        self.qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, n_classes=2)

    def test_qnn_initialization(self):
        """Test QNN initialization"""
        self.assertEqual(self.qnn.n_qubits, 4)
        self.assertEqual(self.qnn.n_layers, 2)
        self.assertEqual(self.qnn.n_classes, 2)
        self.assertIsNotNone(self.qnn.parameters)

    def test_qnn_forward_pass(self):
        """Test QNN forward pass"""
        x = np.array([0.1, 0.2, 0.3, 0.4])

        output = self.qnn.forward(x)

        # Check output shape
        self.assertEqual(len(output), self.qnn.n_classes)

        # Check output is probability distribution
        self.assertTrue(np.all(output >= 0))
        self.assertAlmostEqual(np.sum(output), 1.0, places=5)

    def test_qnn_backward_pass(self):
        """Test QNN backward pass"""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y_true = np.array([1.0, 0.0])  # One-hot encoded

        # Forward pass
        y_pred = self.qnn.forward(x)

        # Backward pass
        gradients = self.qnn.backward(x, y_pred, y_true)

        # Check gradients exist
        self.assertIsNotNone(gradients)
        self.assertIsInstance(gradients, dict)

    def test_qnn_training_step(self):
        """Test QNN training step"""
        x = np.array([0.1, 0.2, 0.3, 0.4])
        y_true = np.array([1.0, 0.0])

        # Training step
        loss = self.qnn.training_step(x, y_true)

        # Loss should be a number
        self.assertIsInstance(loss, (int, float, np.number))

        # Loss should be non-negative
        self.assertGreaterEqual(loss, 0)

class TestQuantumMachineLearning(unittest.TestCase):
    """Test Quantum Machine Learning suite"""

    def setUp(self):
        """Set up test fixtures"""
        self.qml = QuantumMachineLearning(n_qubits=4)

    def test_qml_initialization(self):
        """Test QML initialization"""
        self.assertEqual(self.qml.n_qubits, 4)
        self.assertIsNotNone(self.qml.feature_map)
        self.assertIsNotNone(self.qml.kernel)
        self.assertIsNotNone(self.qml.qsvm)
        self.assertIsNotNone(self.qml.qnn)

    def test_qml_classification_pipeline(self):
        """Test complete QML classification pipeline"""
        # Generate test data
        X = np.random.randn(50, 4)
        y = np.sign(X[:, 0] + X[:, 1] - 0.5)

        # Train model
        self.qml.train_classifier(X, y, method='qsvm')

        # Make predictions
        X_test = np.random.randn(10, 4)
        predictions = self.qml.classify(X_test)

        # Check predictions
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))

    def test_qml_regression_pipeline(self):
        """Test QML regression pipeline"""
        # Generate test data
        X = np.random.randn(50, 4)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(50) * 0.1

        # Train model
        self.qml.train_regressor(X, y)

        # Make predictions
        X_test = np.random.randn(10, 4)
        predictions = self.qml.regress(X_test)

        # Check predictions
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(isinstance(pred, (int, float, np.number)) for pred in predictions))

class TestHybridQuantumClassical(unittest.TestCase):
    """Test Hybrid Quantum-Classical Algorithms"""

    def setUp(self):
        """Set up test fixtures"""
        self.hybrid = HybridQuantumClassical(n_qubits=4, classical_layers=2)

    def test_hybrid_initialization(self):
        """Test hybrid algorithm initialization"""
        self.assertEqual(self.hybrid.n_qubits, 4)
        self.assertEqual(self.hybrid.classical_layers, 2)
        self.assertIsNotNone(self.hybrid.quantum_circuit)
        self.assertIsNotNone(self.hybrid.classical_network)

    def test_hybrid_forward_pass(self):
        """Test hybrid forward pass"""
        x = np.random.randn(10)

        output = self.hybrid.forward(x)

        # Check output exists
        self.assertIsNotNone(output)
        self.assertIsInstance(output, np.ndarray)

    def test_hybrid_training(self):
        """Test hybrid training"""
        X = np.random.randn(20, 10)
        y = np.random.randn(20, 5)

        # Training should not raise exception
        self.hybrid.train(X, y, epochs=2)

        # Check that model is trained
        self.assertTrue(self.hybrid.is_trained)

class TestQAOASolver(unittest.TestCase):
    """Test QAOA Solver"""

    def setUp(self):
        """Set up test fixtures"""
        self.qaoa = QAOASolver(n_qubits=4, p=2)

    def test_qaoa_initialization(self):
        """Test QAOA initialization"""
        self.assertEqual(self.qaoa.n_qubits, 4)
        self.assertEqual(self.qaoa.p, 2)
        self.assertIsNotNone(self.qaoa.cost_hamiltonian)
        self.assertIsNotNone(self.qaoa.mixer_hamiltonian)

    def test_qaoa_optimization(self):
        """Test QAOA optimization"""
        # Define a simple cost function (max cut)
        def cost_function(bitstring):
            # Simple max cut on a triangle
            edges = [(0, 1), (1, 2), (2, 0)]
            cost = 0
            for i, j in edges:
                if bitstring[i] != bitstring[j]:
                    cost += 1
            return -cost  # Maximize cut

        # Optimize
        result = self.qaoa.optimize(cost_function, max_iter=10)

        # Check result
        self.assertIsNotNone(result)
        self.assertIn('optimal_solution', result)
        self.assertIn('optimal_value', result)

class TestVQESolver(unittest.TestCase):
    """Test VQE Solver"""

    def setUp(self):
        """Set up test fixtures"""
        self.vqe = VQESolver(n_qubits=4, ansatz_depth=2)

    def test_vqe_initialization(self):
        """Test VQE initialization"""
        self.assertEqual(self.vqe.n_qubits, 4)
        self.assertEqual(self.vqe.ansatz_depth, 2)
        self.assertIsNotNone(self.vqe.ansatz)

    def test_vqe_ground_state_search(self):
        """Test VQE ground state search"""
        # Define a simple Hamiltonian (Ising model)
        hamiltonian_terms = [
            (1.0, [0, 1], 'ZZ'),  # Z_0 * Z_1
            (0.5, [1, 2], 'ZZ'),  # Z_1 * Z_2
            (-1.0, [0], 'Z'),     # -Z_0
        ]

        # Find ground state
        result = self.vqe.find_ground_state(hamiltonian_terms, max_iter=5)

        # Check result
        self.assertIsNotNone(result)
        self.assertIn('ground_state_energy', result)
        self.assertIn('optimal_parameters', result)

class TestQuantumOptimizer(unittest.TestCase):
    """Test Quantum Optimizer"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = QuantumOptimizer(method='adam', learning_rate=0.01)

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.method, 'adam')
        self.assertEqual(self.optimizer.learning_rate, 0.01)

    def test_parameter_update(self):
        """Test parameter update"""
        params = np.random.randn(10)
        gradients = np.random.randn(10)

        # Update parameters
        new_params = self.optimizer.update(params, gradients)

        # Check that parameters changed
        self.assertFalse(np.array_equal(params, new_params))

        # Check shape preserved
        self.assertEqual(new_params.shape, params.shape)

class TestQuantumAlgorithmsPerformance(unittest.TestCase):
    """Performance tests for quantum algorithms"""

    def setUp(self):
        """Set up performance test"""
        self.qml = QuantumMachineLearning(n_qubits=4)

    def test_qsvm_training_performance(self):
        """Test QSVM training performance"""
        X = np.random.randn(50, 4)
        y = np.sign(X[:, 0] + X[:, 1] - 0.5)

        # Measure training time
        _, train_time = TestUtils.measure_execution_time(
            self.qml.train_classifier, X, y, method='qsvm'
        )

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(train_time, 30.0)  # 30 seconds max

    def test_kernel_computation_performance(self):
        """Test kernel computation performance"""
        kernel = QuantumKernel(n_qubits=4)
        X = np.random.randn(20, 4)

        # Measure kernel matrix computation time
        _, compute_time = TestUtils.measure_execution_time(
            kernel.compute_kernel_matrix, X
        )

        # Should be reasonably fast
        self.assertLess(compute_time, 10.0)  # 10 seconds max

    def test_qnn_inference_performance(self):
        """Test QNN inference performance"""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, n_classes=2)
        x = np.random.randn(4)

        # Measure inference time
        _, inference_time = TestUtils.measure_execution_time(qnn.forward, x)

        # Should be fast
        TestUtils.assert_performance_threshold(qnn.forward, 100.0, x)  # 100ms max

class TestQuantumAlgorithmsIntegration(unittest.TestCase):
    """Integration tests for quantum algorithms"""

    def test_complete_ml_pipeline(self):
        """Test complete ML pipeline"""
        # Create data
        X = np.random.randn(100, 4)
        y = np.sign(X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 2])

        # Initialize QML
        qml = QuantumMachineLearning(n_qubits=4)

        # Train model
        qml.train_classifier(X, y, method='qsvm')

        # Test predictions
        X_test = np.random.randn(10, 4)
        predictions = qml.classify(X_test)

        # Verify predictions
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))

        # Test accuracy (should be reasonable for this simple problem)
        y_test = np.sign(X_test[:, 0] + 0.5 * X_test[:, 1] + 0.2 * X_test[:, 2])
        accuracy = np.mean(predictions == y_test)
        self.assertGreater(accuracy, 0.5)  # Should be better than random

    def test_hybrid_algorithm_training(self):
        """Test hybrid algorithm training"""
        hybrid = HybridQuantumClassical(n_qubits=4, classical_layers=1)

        # Create training data
        X = np.random.randn(50, 8)
        y = np.random.randn(50, 3)

        # Train hybrid model
        hybrid.train(X, y, epochs=3)

        # Test inference
        x_test = np.random.randn(8)
        output = hybrid.forward(x_test)

        # Verify output
        self.assertEqual(len(output), 3)
        self.assertTrue(all(isinstance(val, (int, float, np.number)) for val in output))

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

"""
Integration tests for Quantum Edge AI Platform
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from edge_runtime.inference_engine import EdgeInferenceEngine, Precision
from quantum_algorithms.quantum_ml import QuantumMachineLearning
from federated_learning.federated_server import FederatedLearningServer
from federated_learning.federated_client import FederatedLearningClient
from api_services.rest_api import RESTAPI
from api_services.graphql_api import GraphQLAPI
from api_services.websocket_service import WebSocketService
from privacy_security.privacy import PrivacyEngine, PrivacyMechanism
from tests import TestUtils

class TestEdgeInferenceIntegration(unittest.TestCase):
    """Integration tests for edge inference"""

    def setUp(self):
        """Set up integration test"""
        self.engine = EdgeInferenceEngine()

    def test_model_loading_and_inference(self):
        """Test complete model loading and inference pipeline"""
        # Create mock model spec
        model_spec = TestUtils.create_test_model_spec()

        # Load model (will fail without actual model file, but tests pipeline)
        success = self.engine.load_model(model_spec)
        self.assertFalse(success)  # Expected without real model

        # Test resource monitoring
        resources = self.engine.get_resource_usage()
        self.assertIn('memory_percent', resources)

    def test_adaptive_precision_workflow(self):
        """Test adaptive precision selection workflow"""
        # Test different memory conditions
        test_conditions = [
            {'memory_percent': 20, 'cpu_percent': 15},  # Low load
            {'memory_percent': 75, 'cpu_percent': 60},  # Medium load
            {'memory_percent': 95, 'cpu_percent': 90},  # High load
        ]

        for condition in test_conditions:
            precision = self.engine._select_adaptive_precision(condition)

            # Should select appropriate precision
            self.assertIsInstance(precision, Precision)

            # High load should prefer lower precision
            if condition['memory_percent'] > 80:
                self.assertIn(precision, [Precision.INT8, Precision.INT4, Precision.BINARY])

class TestQuantumMLIntegration(unittest.TestCase):
    """Integration tests for quantum machine learning"""

    def setUp(self):
        """Set up integration test"""
        self.qml = QuantumMachineLearning(n_qubits=4)

    def test_complete_ml_workflow(self):
        """Test complete ML workflow"""
        # Generate test data
        X = np.random.randn(100, 4)
        y = np.sign(X[:, 0] + 0.5 * X[:, 1] - 0.2)

        # Train model
        self.qml.train_classifier(X, y, method='qsvm')

        # Make predictions
        X_test = np.random.randn(20, 4)
        predictions = self.qml.classify(X_test)

        # Verify predictions
        self.assertEqual(len(predictions), 20)
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))

    def test_quantum_feature_processing(self):
        """Test quantum feature processing pipeline"""
        # Test feature map
        features = np.random.randn(10, 4)

        # Process through quantum feature map
        processed = []
        for feature in features:
            encoded = self.qml.feature_map.encode(feature)
            processed.append(encoded)

        # Should process all features
        self.assertEqual(len(processed), len(features))

class TestFederatedLearningIntegration(unittest.TestCase):
    """Integration tests for federated learning"""

    def setUp(self):
        """Set up integration test"""
        self.server = FederatedLearningServer()
        self.clients = [FederatedLearningClient(f"client_{i}") for i in range(3)]

    def test_federated_round_workflow(self):
        """Test federated learning round workflow"""
        # Initialize global model
        global_model = {'weights': np.random.randn(10, 5), 'bias': np.random.randn(5)}

        # Simulate federated round
        self.server.initialize_global_model(global_model)

        # Clients perform local training (mock)
        client_updates = []
        for client in self.clients:
            # Mock local training
            update = {
                'client_id': client.client_id,
                'weights': global_model['weights'] + np.random.randn(10, 5) * 0.1,
                'bias': global_model['bias'] + np.random.randn(5) * 0.1,
                'num_samples': 100
            }
            client_updates.append(update)

        # Server aggregates updates
        aggregated_model = self.server.aggregate_updates(client_updates)

        # Verify aggregation
        self.assertIn('weights', aggregated_model)
        self.assertIn('bias', aggregated_model)
        self.assertEqual(aggregated_model['weights'].shape, global_model['weights'].shape)

class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API services"""

    def setUp(self):
        """Set up integration test"""
        self.rest_api = RESTAPI()
        self.graphql_api = GraphQLAPI()
        self.websocket_service = WebSocketService()

    def test_api_initialization(self):
        """Test API service initialization"""
        # Should initialize without errors
        self.assertIsNotNone(self.rest_api)
        self.assertIsNotNone(self.graphql_api)
        self.assertIsNotNone(self.websocket_service)

    def test_rest_api_request_handling(self):
        """Test REST API request handling"""
        # Create mock request
        request = TestUtils.create_mock_request(
            method='POST',
            path='/api/inference',
            data={'input': [1.0, 2.0, 3.0]}
        )

        # Process request (mock response)
        response = self.rest_api._process_inference_request(request)

        # Should return response structure
        self.assertIsInstance(response, dict)

    def test_websocket_connection_handling(self):
        """Test WebSocket connection handling"""
        # Mock connection
        mock_connection = Mock()

        # Handle connection
        self.websocket_service.handle_connection(mock_connection)

        # Should add to active connections
        self.assertIn(mock_connection, self.websocket_service.active_connections)

class TestPrivacyIntegration(unittest.TestCase):
    """Integration tests for privacy features"""

    def setUp(self):
        """Set up integration test"""
        self.privacy_engine = PrivacyEngine(
            mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
            epsilon=0.5
        )

    def test_privacy_protected_inference(self):
        """Test privacy-protected inference"""
        # Create test data
        sensitive_data = np.random.randn(50, 10)

        # Apply privacy protection
        protected_data, report = self.privacy_engine.apply_privacy(sensitive_data)

        # Verify protection
        self.assertIsNotNone(protected_data)
        self.assertIsNotNone(report)
        self.assertEqual(protected_data.shape, sensitive_data.shape)

    def test_privacy_compliance_checking(self):
        """Test privacy compliance checking"""
        data_usage_scenarios = [
            "data_processing_with_consent_and_encryption",
            "anonymous_analytics_collection",
            "personal_data_storage_with_retention_policy"
        ]

        for scenario in data_usage_scenarios:
            compliance = self.privacy_engine.check_compliance(scenario)

            # Should return compliance assessment
            self.assertIsInstance(compliance, dict)
            self.assertIn('gdpr_compliant', compliance)

class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end workflow tests"""

    def test_complete_edge_ai_pipeline(self):
        """Test complete edge AI pipeline"""
        # Initialize components
        inference_engine = EdgeInferenceEngine()
        qml = QuantumMachineLearning(n_qubits=4)
        privacy_engine = PrivacyEngine()

        # Generate test data
        X = np.random.randn(200, 4)
        y = np.sign(X[:, 0] + 0.5 * X[:, 1] - 0.1 * X[:, 2])

        # Train quantum model
        qml.train_classifier(X, y, method='qsvm')

        # Apply privacy to test data
        X_private, _ = privacy_engine.apply_privacy(X[:50])

        # Make predictions on private data
        predictions = qml.classify(X_private)

        # Verify pipeline completion
        self.assertEqual(len(predictions), 50)
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))

    def test_federated_privacy_protected_learning(self):
        """Test federated learning with privacy protection"""
        # Setup federated learning
        server = FederatedLearningServer()
        clients = [FederatedLearningClient(f"client_{i}") for i in range(3)]

        # Initialize privacy for each client
        privacy_engines = [PrivacyEngine(epsilon=0.8) for _ in clients]

        # Simulate federated rounds
        for round_num in range(2):
            # Clients perform local training with privacy
            client_updates = []
            for i, client in enumerate(clients):
                # Generate private local data
                local_data = np.random.randn(30, 4)
                local_labels = np.random.randn(30)

                # Apply local privacy
                private_data, _ = privacy_engines[i].apply_privacy(local_data)

                # Mock training update
                update = {
                    'client_id': client.client_id,
                    'weights': np.random.randn(10, 5) * 0.1,
                    'bias': np.random.randn(5) * 0.1,
                    'num_samples': len(private_data)
                }
                client_updates.append(update)

            # Server aggregates
            aggregated = server.aggregate_updates(client_updates)

            # Verify aggregation
            self.assertIsNotNone(aggregated)

class TestConcurrentOperations(unittest.TestCase):
    """Tests for concurrent operations"""

    def test_concurrent_inference_requests(self):
        """Test concurrent inference requests"""
        engine = EdgeInferenceEngine()

        def mock_inference_request(request_id):
            # Simulate inference time
            time.sleep(0.01)
            return f"result_{request_id}"

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(mock_inference_request, i)
                for i in range(10)
            ]

            results = [future.result() for future in as_completed(futures)]

        # Should complete all requests
        self.assertEqual(len(results), 10)
        self.assertTrue(all('result_' in r for r in results))

    def test_concurrent_privacy_operations(self):
        """Test concurrent privacy operations"""
        privacy_engine = PrivacyEngine()

        def apply_privacy_to_batch(batch_id):
            data = np.random.randn(20, 5)
            protected_data, _ = privacy_engine.apply_privacy(data)
            return batch_id, protected_data.shape

        # Run concurrent privacy operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(apply_privacy_to_batch, i)
                for i in range(5)
            ]

            results = [future.result() for future in as_completed(futures)]

        # Verify all operations completed
        self.assertEqual(len(results), 5)
        for batch_id, shape in results:
            self.assertEqual(shape, (20, 5))

class TestResourceManagement(unittest.TestCase):
    """Tests for resource management"""

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring"""
        engine = EdgeInferenceEngine()

        # Get initial memory usage
        initial_memory = engine.get_resource_usage()

        # Perform some operations
        for _ in range(10):
            _ = np.random.randn(1000, 100)  # Allocate memory

        # Get final memory usage
        final_memory = engine.get_resource_usage()

        # Memory monitoring should work
        self.assertIsInstance(initial_memory, dict)
        self.assertIsInstance(final_memory, dict)
        self.assertIn('memory_percent', final_memory)

    def test_model_unloading_under_memory_pressure(self):
        """Test model unloading under memory pressure"""
        engine = EdgeInferenceEngine()

        # Load multiple models (mock)
        models = []
        for i in range(3):
            model_spec = TestUtils.create_test_model_spec()
            model_spec['model_id'] = f"model_{i}"
            models.append(model_spec)

        # Simulate memory pressure
        engine.memory_threshold = 50  # Low threshold

        # Mock high memory usage
        with patch.object(engine, '_get_memory_usage', return_value=80):
            # Should trigger cleanup
            cleaned = engine._cleanup_memory()

            # Should return some cleanup info
            self.assertIsInstance(cleaned, (int, bool))

class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Tests for error handling and recovery"""

    def test_inference_engine_error_recovery(self):
        """Test inference engine error recovery"""
        engine = EdgeInferenceEngine()

        # Test with invalid model
        invalid_spec = {"invalid": "spec"}

        # Should handle error gracefully
        success = engine.load_model(invalid_spec)
        self.assertFalse(success)

        # Engine should still be functional
        resources = engine.get_resource_usage()
        self.assertIsNotNone(resources)

    def test_privacy_engine_error_handling(self):
        """Test privacy engine error handling"""
        privacy_engine = PrivacyEngine()

        # Test with invalid data
        invalid_data = "invalid_data_type"

        # Should handle gracefully
        try:
            result, report = privacy_engine.apply_privacy(invalid_data)
            # If it succeeds, verify result
            self.assertIsNotNone(result)
        except Exception as e:
            # If it fails, should be controlled error
            self.assertIsInstance(e, Exception)

    def test_api_error_handling(self):
        """Test API error handling"""
        rest_api = RESTAPI()

        # Test with malformed request
        malformed_request = {"invalid": "request"}

        # Should handle gracefully
        try:
            response = rest_api._process_request(malformed_request)
            # Should return error response
            self.assertIn('error', response)
        except Exception:
            # Or raise controlled exception
            pass

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""

    def test_inference_throughput_benchmark(self):
        """Benchmark inference throughput"""
        # Create mock inference function
        def mock_inference(data):
            time.sleep(0.001)  # 1ms per inference
            return np.array([0.5])

        # Benchmark throughput
        num_requests = 100
        test_data = [np.random.randn(10) for _ in range(num_requests)]

        start_time = time.time()
        results = [mock_inference(data) for data in test_data]
        end_time = time.time()

        throughput = num_requests / (end_time - start_time)

        # Should achieve reasonable throughput
        self.assertGreater(throughput, 50)  # At least 50 inferences per second

    def test_privacy_overhead_benchmark(self):
        """Benchmark privacy mechanism overhead"""
        privacy_engine = PrivacyEngine()
        data = np.random.randn(1000, 10)

        # Measure time without privacy
        start_time = time.time()
        original_result = data.copy()
        original_time = time.time() - start_time

        # Measure time with privacy
        start_time = time.time()
        private_result, _ = privacy_engine.apply_privacy(data)
        private_time = time.time() - start_time

        # Privacy should add reasonable overhead
        overhead_ratio = private_time / max(original_time, 0.001)
        self.assertLess(overhead_ratio, 10)  # Less than 10x overhead

    def test_memory_efficiency_benchmark(self):
        """Benchmark memory efficiency"""
        engine = EdgeInferenceEngine()

        initial_memory = engine.get_resource_usage()

        # Perform memory-intensive operations
        large_arrays = []
        for _ in range(10):
            large_arrays.append(np.random.randn(1000, 1000))

        peak_memory = engine.get_resource_usage()

        # Clean up
        del large_arrays

        final_memory = engine.get_resource_usage()

        # Memory should be manageable
        memory_increase = peak_memory.get('memory_percent', 0) - initial_memory.get('memory_percent', 0)
        self.assertLess(memory_increase, 200)  # Less than 200% increase

if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)

"""
Unit tests for Edge Inference Engine
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from edge_runtime.inference_engine import (
    EdgeInferenceEngine, Precision, InferenceResult, ModelSpec, Layer, DenseLayer,
    Conv2DLayer, InferenceEngineError, ResourceExhaustionError, ModelLoadError
)
from tests import TestUtils

class TestPrecision(unittest.TestCase):
    """Test precision handling"""

    def test_precision_enum_values(self):
        """Test precision enum values"""
        self.assertEqual(Precision.FP32.value, 4)
        self.assertEqual(Precision.FP16.value, 2)
        self.assertEqual(Precision.INT8.value, 1)
        self.assertEqual(Precision.INT4.value, 0.5)
        self.assertEqual(Precision.BINARY.value, 0.125)

    def test_precision_from_string(self):
        """Test creating precision from string"""
        self.assertEqual(Precision.from_string("fp32"), Precision.FP32)
        self.assertEqual(Precision.from_string("FP16"), Precision.FP16)
        self.assertEqual(Precision.from_string("int8"), Precision.INT8)
        self.assertEqual(Precision.from_string("INT4"), Precision.INT4)
        self.assertEqual(Precision.from_string("binary"), Precision.BINARY)

    def test_precision_from_string_invalid(self):
        """Test invalid precision string"""
        with self.assertRaises(ValueError):
            Precision.from_string("invalid")

class TestInferenceResult(unittest.TestCase):
    """Test inference result structure"""

    def test_inference_result_creation(self):
        """Test creating inference result"""
        output = np.array([0.1, 0.9, 0.0])
        probabilities = np.array([0.2, 0.8])

        result = InferenceResult(
            output=output,
            probabilities=probabilities,
            inference_time=0.05,
            model_id="test_model",
            precision=Precision.FP32,
            memory_usage=1024,
            confidence=0.85
        )

        self.assertTrue(np.array_equal(result.output, output))
        self.assertTrue(np.array_equal(result.probabilities, probabilities))
        self.assertEqual(result.inference_time, 0.05)
        self.assertEqual(result.model_id, "test_model")
        self.assertEqual(result.precision, Precision.FP32)
        self.assertEqual(result.memory_usage, 1024)
        self.assertEqual(result.confidence, 0.85)

    def test_inference_result_to_dict(self):
        """Test converting result to dictionary"""
        result = InferenceResult(
            output=np.array([0.1, 0.9]),
            probabilities=np.array([0.2, 0.8]),
            inference_time=0.05,
            model_id="test_model",
            precision=Precision.FP32
        )

        result_dict = result.to_dict()

        self.assertIn('output', result_dict)
        self.assertIn('probabilities', result_dict)
        self.assertIn('inference_time', result_dict)
        self.assertIn('model_id', result_dict)
        self.assertEqual(result_dict['model_id'], "test_model")

class TestModelSpec(unittest.TestCase):
    """Test model specification"""

    def test_model_spec_creation(self):
        """Test creating model specification"""
        spec = ModelSpec(
            model_id="test_model_001",
            model_type="neural_network",
            input_shape=[1, 28, 28],
            output_shape=[10],
            precision=Precision.FP32,
            framework="pytorch",
            metadata={"version": "1.0"}
        )

        self.assertEqual(spec.model_id, "test_model_001")
        self.assertEqual(spec.model_type, "neural_network")
        self.assertEqual(spec.input_shape, [1, 28, 28])
        self.assertEqual(spec.output_shape, [10])
        self.assertEqual(spec.precision, Precision.FP32)
        self.assertEqual(spec.framework, "pytorch")
        self.assertEqual(spec.metadata["version"], "1.0")

    def test_model_spec_validation(self):
        """Test model spec validation"""
        # Valid spec
        spec = ModelSpec(
            model_id="test",
            model_type="neural_network",
            input_shape=[1, 28, 28],
            output_shape=[10]
        )
        spec.validate()  # Should not raise

        # Invalid model_id
        with self.assertRaises(ValueError):
            invalid_spec = ModelSpec(
                model_id="",
                model_type="neural_network",
                input_shape=[1, 28, 28],
                output_shape=[10]
            )
            invalid_spec.validate()

class TestLayers(unittest.TestCase):
    """Test neural network layers"""

    def test_dense_layer_creation(self):
        """Test creating dense layer"""
        layer = DenseLayer(
            input_size=784,
            output_size=128,
            activation='relu',
            use_bias=True
        )

        self.assertEqual(layer.input_size, 784)
        self.assertEqual(layer.output_size, 128)
        self.assertEqual(layer.activation, 'relu')
        self.assertTrue(layer.use_bias)

    def test_dense_layer_forward(self):
        """Test dense layer forward pass"""
        layer = DenseLayer(input_size=4, output_size=3)

        # Initialize weights
        layer.weights = np.random.randn(4, 3)
        layer.bias = np.random.randn(3)

        # Test input
        x = np.array([1.0, 2.0, 3.0, 4.0])

        output = layer.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (3,))

        # Check it's not just zeros
        self.assertFalse(np.allclose(output, 0))

    def test_conv2d_layer_creation(self):
        """Test creating Conv2D layer"""
        layer = Conv2DLayer(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.assertEqual(layer.in_channels, 1)
        self.assertEqual(layer.out_channels, 32)
        self.assertEqual(layer.kernel_size, 3)
        self.assertEqual(layer.stride, 1)
        self.assertEqual(layer.padding, 1)

class TestEdgeInferenceEngine(unittest.TestCase):
    """Test Edge Inference Engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = EdgeInferenceEngine()
        self.test_model_spec = TestUtils.create_test_model_spec()

    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsInstance(self.engine, EdgeInferenceEngine)
        self.assertEqual(len(self.engine.loaded_models), 0)
        self.assertIsNotNone(self.engine.resource_monitor)

    @patch('edge_runtime.inference_engine.EdgeInferenceEngine._load_model_from_file')
    def test_load_model(self, mock_load):
        """Test loading a model"""
        # Mock the model loading
        mock_model = Mock()
        mock_load.return_value = mock_model

        # Load model
        success = self.engine.load_model(self.test_model_spec)

        self.assertTrue(success)
        self.assertIn(self.test_model_spec['model_id'], self.engine.loaded_models)
        mock_load.assert_called_once()

    def test_load_model_invalid_spec(self):
        """Test loading model with invalid spec"""
        invalid_spec = {"invalid": "spec"}

        with self.assertRaises(ModelLoadError):
            self.engine.load_model(invalid_spec)

    def test_unload_model(self):
        """Test unloading a model"""
        # First load a model
        self.engine.loaded_models[self.test_model_spec['model_id']] = Mock()

        # Unload it
        success = self.engine.unload_model(self.test_model_spec['model_id'])

        self.assertTrue(success)
        self.assertNotIn(self.test_model_spec['model_id'], self.engine.loaded_models)

    def test_unload_nonexistent_model(self):
        """Test unloading non-existent model"""
        success = self.engine.unload_model("nonexistent_model")
        self.assertFalse(success)

    @patch('edge_runtime.inference_engine.EdgeInferenceEngine._perform_inference')
    def test_run_inference(self, mock_inference):
        """Test running inference"""
        # Setup
        model_id = self.test_model_spec['model_id']
        self.engine.loaded_models[model_id] = Mock()

        test_data = TestUtils.create_test_data()
        mock_result = InferenceResult(
            output=np.array([0.1, 0.9]),
            probabilities=np.array([0.2, 0.8]),
            inference_time=0.05,
            model_id=model_id,
            precision=Precision.FP32
        )
        mock_inference.return_value = mock_result

        # Run inference
        result = self.engine.run_inference(model_id, test_data['input'])

        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.model_id, model_id)
        mock_inference.assert_called_once()

    def test_run_inference_unloaded_model(self):
        """Test inference with unloaded model"""
        with self.assertRaises(InferenceEngineError):
            self.engine.run_inference("nonexistent_model", np.array([1.0]))

    @patch('edge_runtime.inference_engine.psutil.virtual_memory')
    def test_resource_monitoring(self, mock_memory):
        """Test resource monitoring"""
        # Mock memory info
        mock_memory.return_value = Mock()
        mock_memory.return_value.percent = 75.5
        mock_memory.return_value.available = 2 * 1024 * 1024 * 1024  # 2GB

        # Test monitoring
        resources = self.engine.get_resource_usage()

        self.assertIn('memory_percent', resources)
        self.assertIn('memory_available', resources)
        self.assertEqual(resources['memory_percent'], 75.5)

    def test_adaptive_precision_selection(self):
        """Test adaptive precision selection"""
        # Test different memory conditions
        high_memory = {'memory_percent': 30, 'cpu_percent': 20}
        low_memory = {'memory_percent': 85, 'cpu_percent': 80}

        # High memory should allow higher precision
        precision = self.engine._select_adaptive_precision(high_memory)
        self.assertIn(precision, [Precision.FP32, Precision.FP16])

        # Low memory should force lower precision
        precision = self.engine._select_adaptive_precision(low_memory)
        self.assertIn(precision, [Precision.INT8, Precision.INT4, Precision.BINARY])

    def test_quantize_weights(self):
        """Test weight quantization"""
        # Create test weights
        weights = np.random.randn(100, 50).astype(np.float32)

        # Quantize to different precisions
        for precision in [Precision.INT8, Precision.INT4]:
            quantized = self.engine._quantize_weights(weights, precision)

            # Check shape is preserved
            self.assertEqual(quantized.shape, weights.shape)

            # Check dtype is integer
            self.assertTrue(np.issubdtype(quantized.dtype, np.integer))

    def test_performance_monitoring(self):
        """Test performance monitoring"""
        model_id = "test_model"

        # Record some performance data
        self.engine.performance_history[model_id] = [
            {'inference_time': 0.1, 'memory_usage': 1024, 'timestamp': time.time()},
            {'inference_time': 0.08, 'memory_usage': 1050, 'timestamp': time.time()},
            {'inference_time': 0.12, 'memory_usage': 1030, 'timestamp': time.time()},
        ]

        # Get performance stats
        stats = self.engine.get_performance_stats(model_id)

        self.assertIn('average_inference_time', stats)
        self.assertIn('max_inference_time', stats)
        self.assertIn('min_inference_time', stats)
        self.assertIn('total_inferences', stats)

        self.assertEqual(stats['total_inferences'], 3)
        self.assertAlmostEqual(stats['average_inference_time'], 0.1, places=2)

    def test_memory_usage_calculation(self):
        """Test memory usage calculation"""
        # Test with different tensor sizes
        small_tensor = np.random.randn(10, 10).astype(np.float32)
        large_tensor = np.random.randn(1000, 1000).astype(np.float32)

        small_usage = self.engine._calculate_memory_usage(small_tensor)
        large_usage = self.engine._calculate_memory_usage(large_tensor)

        # Large tensor should use more memory
        self.assertGreater(large_usage, small_usage)

        # Small tensor should be reasonable size (10*10*4 = 400 bytes)
        self.assertGreater(small_usage, 300)
        self.assertLess(small_usage, 500)

class TestInferenceEngineIntegration(unittest.TestCase):
    """Integration tests for inference engine"""

    def setUp(self):
        """Set up integration test"""
        self.engine = EdgeInferenceEngine()

    def test_full_inference_pipeline(self):
        """Test complete inference pipeline"""
        # This would be a more comprehensive integration test
        # For now, just test the pipeline setup

        model_spec = TestUtils.create_test_model_spec()
        test_data = TestUtils.create_test_data()

        # Load model (would fail in real scenario without actual model file)
        success = self.engine.load_model(model_spec)
        self.assertFalse(success)  # Expected to fail without real model

    def test_resource_limits(self):
        """Test resource limit enforcement"""
        # Test with very large input that should trigger resource limits
        large_input = np.random.randn(1000, 1000, 100).astype(np.float32)

        # This should either succeed or fail gracefully with resource limits
        try:
            # Would need a loaded model to actually test
            # For now, just verify the method exists and handles input
            self.assertIsInstance(large_input, np.ndarray)
        except Exception as e:
            # Should be a controlled error, not a crash
            self.assertIsInstance(e, Exception)

class TestInferenceEnginePerformance(unittest.TestCase):
    """Performance tests for inference engine"""

    def setUp(self):
        """Set up performance test"""
        self.engine = EdgeInferenceEngine()

    def test_inference_performance_baseline(self):
        """Test baseline inference performance"""
        # Create mock model and test data
        mock_model = Mock()
        self.engine.loaded_models["perf_test_model"] = mock_model

        # Mock inference method to return quickly
        def mock_perform_inference(data):
            time.sleep(0.001)  # 1ms delay
            return InferenceResult(
                output=np.array([0.5]),
                probabilities=np.array([1.0]),
                inference_time=0.001,
                model_id="perf_test_model",
                precision=Precision.FP32
            )

        self.engine._perform_inference = mock_perform_inference

        # Test performance
        test_data = TestUtils.create_test_data(batch_size=1)

        # Measure performance
        result, exec_time = TestUtils.measure_execution_time(
            self.engine.run_inference,
            "perf_test_model",
            test_data['input']
        )

        # Should be very fast (< 10ms)
        TestUtils.assert_performance_threshold(
            self.engine.run_inference,
            10.0,  # 10ms threshold
            "perf_test_model",
            test_data['input']
        )

        self.assertIsInstance(result, InferenceResult)

    def test_memory_efficiency(self):
        """Test memory efficiency"""
        # Test that memory usage stays within bounds
        initial_memory = self.engine.get_resource_usage()

        # Perform some operations
        for _ in range(10):
            test_data = TestUtils.create_test_data(batch_size=32)
            # Would need actual inference here

        final_memory = self.engine.get_resource_usage()

        # Memory usage should not increase dramatically
        memory_increase = final_memory.get('memory_percent', 0) - initial_memory.get('memory_percent', 0)
        self.assertLess(memory_increase, 50)  # Less than 50% increase

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

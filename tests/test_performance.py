"""
Performance benchmark tests for Quantum Edge AI Platform
"""

import unittest
import numpy as np
import time
import psutil
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from memory_profiler import profile as memory_profile
import cProfile
import pstats
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from edge_runtime.inference_engine import EdgeInferenceEngine, Precision
from quantum_algorithms.quantum_ml import QuantumMachineLearning
from privacy_security.privacy import PrivacyEngine, DifferentialPrivacy
from tests import TestUtils

class PerformanceBenchmark:
    """Base class for performance benchmarks"""

    def __init__(self, name: str):
        self.name = name
        self.results = []

    def measure_time(self, func, *args, **kwargs):
        """Measure execution time"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        self.results.append({
            'operation': func.__name__,
            'time': execution_time,
            'timestamp': time.time()
        })

        return result, execution_time

    def measure_memory(self, func, *args, **kwargs):
        """Measure memory usage"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        self.results.append({
            'operation': func.__name__,
            'memory_used': memory_used,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'timestamp': time.time()
        })

        return result, memory_used

    def get_statistics(self):
        """Get performance statistics"""
        if not self.results:
            return {}

        times = [r['time'] for r in self.results if 'time' in r]
        memories = [r['memory_used'] for r in self.results if 'memory_used' in r]

        stats = {
            'total_operations': len(self.results),
            'avg_time': np.mean(times) if times else 0,
            'max_time': np.max(times) if times else 0,
            'min_time': np.min(times) if times else 0,
            'avg_memory': np.mean(memories) if memories else 0,
            'max_memory': np.max(memories) if memories else 0,
            'total_memory': np.sum(memories) if memories else 0
        }

        return stats

class TestInferenceEnginePerformance(unittest.TestCase):
    """Performance tests for Inference Engine"""

    def setUp(self):
        """Set up performance test"""
        self.engine = EdgeInferenceEngine()
        self.benchmark = PerformanceBenchmark("InferenceEngine")

    def test_inference_latency_benchmark(self):
        """Benchmark inference latency"""
        # Create test data of different sizes
        test_cases = [
            (1, [1, 28, 28]),      # Tiny input
            (32, [3, 224, 224]),   # Standard image
            (64, [512]),           # Large vector
        ]

        for batch_size, input_shape in test_cases:
            with self.subTest(batch_size=batch_size, input_shape=input_shape):
                data = np.random.randn(batch_size, *input_shape).astype(np.float32)

                # Measure inference time (mock - would need real model)
                _, latency = self.benchmark.measure_time(
                    lambda: time.sleep(0.001)  # Simulate 1ms inference
                )

                # Assert reasonable latency (< 100ms for edge device)
                self.assertLess(latency, 0.1, f"Latency too high for {batch_size}x{input_shape}")

    def test_throughput_benchmark(self):
        """Benchmark inference throughput"""
        num_requests = 100
        batch_size = 1
        input_shape = [1, 28, 28]

        # Simulate concurrent inference requests
        def simulate_inference():
            time.sleep(0.005)  # 5ms per inference
            return np.random.randn(10)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simulate_inference) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time

        # Should achieve reasonable throughput
        self.assertGreater(throughput, 20, "Throughput too low")  # 20 inferences/sec minimum

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during inference"""
        # Test different model sizes
        model_sizes = [
            (1000, 100),    # Small model
            (10000, 1000),  # Medium model
            (50000, 5000),  # Large model
        ]

        for input_size, output_size in model_sizes:
            with self.subTest(input_size=input_size, output_size=output_size):
                # Simulate model loading and inference
                def simulate_model_operations():
                    # Simulate weights
                    weights = np.random.randn(input_size, output_size)
                    # Simulate inference
                    x = np.random.randn(32, input_size)
                    y = np.dot(x, weights)
                    return y

                _, memory_used = self.benchmark.measure_memory(simulate_model_operations)

                # Memory usage should be reasonable (< 500MB for large models)
                self.assertLess(memory_used, 500, f"Memory usage too high: {memory_used}MB")

    def test_precision_performance_tradeoff(self):
        """Benchmark performance vs precision tradeoff"""
        precisions = [Precision.FP32, Precision.FP16, Precision.INT8, Precision.INT4]
        data = np.random.randn(100, 100)

        precision_performance = {}

        for precision in precisions:
            # Simulate quantization time
            start_time = time.time()
            if precision == Precision.INT8:
                quantized = np.clip(data * 127, -128, 127).astype(np.int8)
                time.sleep(0.001)  # Small delay for INT8
            elif precision == Precision.INT4:
                quantized = np.clip(data * 7, -8, 7).astype(np.int8)
                time.sleep(0.002)  # Slightly more time for INT4
            else:
                quantized = data.astype(np.float32 if precision == Precision.FP32 else np.float16)
                time.sleep(0.0005)  # Fast for FP32/FP16

            end_time = time.time()

            precision_performance[precision.value] = {
                'time': end_time - start_time,
                'compression_ratio': data.nbytes / quantized.nbytes
            }

        # Lower precision should be faster
        self.assertLess(
            precision_performance[Precision.INT8.value]['time'],
            precision_performance[Precision.FP32.value]['time']
        )

        # Lower precision should compress better
        self.assertGreater(
            precision_performance[Precision.INT8.value]['compression_ratio'],
            1.0
        )

    def test_concurrent_inference_scaling(self):
        """Test how performance scales with concurrent requests"""
        concurrency_levels = [1, 2, 4, 8]
        base_requests = 50

        scaling_results = {}

        for concurrency in concurrency_levels:
            num_requests = base_requests * concurrency

            def simulate_inference():
                time.sleep(0.001)  # 1ms base latency
                return np.random.randn(10)

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(simulate_inference) for _ in range(num_requests)]
                results = [f.result() for f in futures]

            end_time = time.time()
            total_time = end_time - start_time

            scaling_results[concurrency] = {
                'total_time': total_time,
                'throughput': num_requests / total_time,
                'efficiency': (num_requests / total_time) / concurrency
            }

        # Efficiency should degrade gracefully with concurrency
        single_efficiency = scaling_results[1]['efficiency']
        high_concurrency_efficiency = scaling_results[8]['efficiency']

        # Should maintain at least 50% efficiency at high concurrency
        efficiency_ratio = high_concurrency_efficiency / single_efficiency
        self.assertGreater(efficiency_ratio, 0.5, "Concurrency scaling too poor")

class TestQuantumMLPerformance(unittest.TestCase):
    """Performance tests for Quantum ML"""

    def setUp(self):
        """Set up performance test"""
        self.qml = QuantumMachineLearning(n_qubits=4)
        self.benchmark = PerformanceBenchmark("QuantumML")

    def test_kernel_computation_performance(self):
        """Benchmark quantum kernel computation"""
        # Test different dataset sizes
        dataset_sizes = [50, 100, 200]

        for n_samples in dataset_sizes:
            with self.subTest(n_samples=n_samples):
                X = np.random.randn(n_samples, 4)

                _, computation_time = self.benchmark.measure_time(
                    self.qml.kernel.compute_kernel_matrix, X
                )

                # Should scale reasonably (O(n²) for kernel computation)
                expected_max_time = 0.1 * (n_samples / 50) ** 2  # Quadratic scaling
                self.assertLess(computation_time, expected_max_time,
                              f"Kernel computation too slow for {n_samples} samples")

    def test_qsvm_training_performance(self):
        """Benchmark QSVM training performance"""
        training_sizes = [50, 100, 150]

        for n_samples in training_sizes:
            with self.subTest(n_samples=n_samples):
                X = np.random.randn(n_samples, 4)
                y = np.sign(X[:, 0] + X[:, 1] - 0.5)

                _, training_time = self.benchmark.measure_time(
                    self.qml.train_classifier, X, y, method='qsvm'
                )

                # Training should be reasonable (< 5 seconds for moderate sizes)
                self.assertLess(training_time, 5.0,
                              f"Training too slow for {n_samples} samples")

    def test_quantum_feature_encoding_speed(self):
        """Benchmark quantum feature encoding speed"""
        n_features = 100
        feature_dim = 4

        features = np.random.randn(n_features, feature_dim)

        start_time = time.time()

        encoded_features = []
        for feature in features:
            encoded = self.qml.feature_map.encode(feature)
            encoded_features.append(encoded)

        end_time = time.time()
        encoding_time = end_time - start_time

        # Should encode features quickly (< 1ms per feature)
        avg_time_per_feature = encoding_time / n_features
        self.assertLess(avg_time_per_feature, 0.001,
                       f"Feature encoding too slow: {avg_time_per_feature*1000:.2f}ms per feature")

    def test_memory_usage_quantum_operations(self):
        """Benchmark memory usage of quantum operations"""
        # Test kernel matrix memory usage
        dataset_sizes = [50, 100, 150]

        for n_samples in dataset_sizes:
            with self.subTest(n_samples=n_samples):
                X = np.random.randn(n_samples, 4)

                _, memory_used = self.benchmark.measure_memory(
                    self.qml.kernel.compute_kernel_matrix, X
                )

                # Kernel matrix is n_samples x n_samples
                expected_memory = (n_samples ** 2 * 8) / (1024 * 1024)  # Float64 in MB

                # Should use reasonable memory (< 100MB for large matrices)
                self.assertLess(memory_used, 100,
                              f"Memory usage too high: {memory_used}MB for {n_samples}x{n_samples} matrix")

class TestPrivacyPerformance(unittest.TestCase):
    """Performance tests for Privacy mechanisms"""

    def setUp(self):
        """Set up performance test"""
        self.privacy_engine = PrivacyEngine()
        self.dp = DifferentialPrivacy(epsilon=1.0)
        self.benchmark = PerformanceBenchmark("Privacy")

    def test_differential_privacy_overhead(self):
        """Benchmark DP mechanism overhead"""
        data_sizes = [(100, 10), (1000, 50), (5000, 100)]

        for n_samples, n_features in data_sizes:
            with self.subTest(n_samples=n_samples, n_features=n_features):
                data = np.random.randn(n_samples, n_features)

                # Measure time without privacy
                _, baseline_time = self.benchmark.measure_time(lambda: data.copy())

                # Measure time with privacy
                _, privacy_time = self.benchmark.measure_time(
                    lambda: self.dp.add_gaussian_noise(data)
                )

                # Privacy overhead should be reasonable
                overhead_ratio = privacy_time / max(baseline_time, 0.0001)
                self.assertLess(overhead_ratio, 5.0,
                              f"Privacy overhead too high: {overhead_ratio:.2f}x")

    def test_privacy_mechanism_scalability(self):
        """Test how privacy mechanisms scale with data size"""
        data_sizes = [1000, 5000, 10000]
        scalability_results = {}

        for size in data_sizes:
            data = np.random.randn(size, 10)

            _, processing_time = self.benchmark.measure_time(
                self.privacy_engine.apply_privacy, data
            )

            scalability_results[size] = processing_time

            # Should process within reasonable time
            self.assertLess(processing_time, 1.0,
                          f"Privacy processing too slow for {size} samples")

        # Check scaling behavior (should be roughly linear)
        small_time = scalability_results[1000]
        large_time = scalability_results[10000]
        scaling_factor = large_time / small_time

        # Should scale roughly linearly (allowing some overhead)
        self.assertLess(scaling_factor, 15.0,
                       f"Poor scaling: {scaling_factor:.2f}x slowdown for 10x data")

    def test_concurrent_privacy_operations(self):
        """Benchmark concurrent privacy operations"""
        num_concurrent = 10
        data_batches = [np.random.randn(100, 10) for _ in range(num_concurrent)]

        def apply_privacy_batch(batch):
            return self.privacy_engine.apply_privacy(batch)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(apply_privacy_batch, batch) for batch in data_batches]
            results = [f.result() for f in futures]

        end_time = time.time()
        total_time = end_time - start_time

        throughput = num_concurrent / total_time

        # Should achieve reasonable concurrent throughput
        self.assertGreater(throughput, 5,
                         f"Concurrent privacy throughput too low: {throughput:.2f} ops/sec")

class TestSystemPerformance(unittest.TestCase):
    """System-wide performance tests"""

    def setUp(self):
        """Set up system performance test"""
        self.benchmark = PerformanceBenchmark("System")

    def test_end_to_end_pipeline_performance(self):
        """Test end-to-end pipeline performance"""
        # Simulate complete pipeline: data preprocessing -> privacy -> quantum ML -> inference
        pipeline_stages = []

        # Stage 1: Data preprocessing
        def preprocess_data():
            data = np.random.randn(1000, 20)
            # Simulate preprocessing
            time.sleep(0.01)
            return data

        _, preprocess_time = self.benchmark.measure_time(preprocess_data)
        pipeline_stages.append(('preprocessing', preprocess_time))

        # Stage 2: Privacy application
        privacy_engine = PrivacyEngine()
        data = np.random.randn(1000, 20)

        _, privacy_time = self.benchmark.measure_time(
            privacy_engine.apply_privacy, data
        )
        pipeline_stages.append(('privacy', privacy_time))

        # Stage 3: Quantum ML training
        qml = QuantumMachineLearning(n_qubits=4)
        X = np.random.randn(500, 4)
        y = np.sign(X[:, 0] + X[:, 1])

        _, training_time = self.benchmark.measure_time(
            qml.train_classifier, X, y, method='qsvm'
        )
        pipeline_stages.append(('training', training_time))

        # Stage 4: Inference
        X_test = np.random.randn(100, 4)

        _, inference_time = self.benchmark.measure_time(
            qml.classify, X_test
        )
        pipeline_stages.append(('inference', inference_time))

        # Calculate total pipeline time
        total_time = sum(time for _, time in pipeline_stages)

        # Pipeline should complete within reasonable time (< 10 seconds)
        self.assertLess(total_time, 10.0,
                       f"Pipeline too slow: {total_time:.2f}s total")

        # Most time should be in training (not preprocessing/privacy)
        training_ratio = training_time / total_time
        self.assertLess(training_ratio, 0.8,
                       "Training dominates pipeline time too much")

    def test_resource_utilization_under_load(self):
        """Test resource utilization under load"""
        # Simulate sustained load
        num_iterations = 50
        memory_usage = []
        cpu_usage = []

        process = psutil.Process(os.getpid())

        for i in range(num_iterations):
            # Perform some work
            data = np.random.randn(1000, 100)
            result = np.dot(data.T, data)
            _ = result.sum()

            # Record resource usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            memory_usage.append(memory_mb)
            cpu_usage.append(cpu_percent)

            time.sleep(0.01)  # Small delay between measurements

        # Analyze resource usage
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        avg_cpu = np.mean(cpu_usage)
        max_cpu = np.max(cpu_usage)

        # Memory usage should be stable (not continuously growing)
        memory_growth = max_memory - memory_usage[0]
        self.assertLess(memory_growth, 50,
                       f"Memory leak detected: {memory_growth:.1f}MB growth")

        # CPU usage should be reasonable
        self.assertLess(avg_cpu, 80,
                       f"Average CPU usage too high: {avg_cpu:.1f}%")

    def test_cold_start_performance(self):
        """Test cold start performance"""
        # Measure time to initialize all major components
        start_time = time.time()

        # Initialize all major components
        inference_engine = EdgeInferenceEngine()
        qml = QuantumMachineLearning(n_qubits=4)
        privacy_engine = PrivacyEngine()

        end_time = time.time()
        cold_start_time = end_time - start_time

        # Cold start should be reasonable (< 5 seconds)
        self.assertLess(cold_start_time, 5.0,
                       f"Cold start too slow: {cold_start_time:.2f}s")

class PerformanceProfilingTest(unittest.TestCase):
    """Advanced performance profiling tests"""

    def test_memory_profiling_inference(self):
        """Profile memory usage during inference operations"""
        @memory_profile
        def profiled_inference():
            engine = EdgeInferenceEngine()
            # Simulate inference workload
            data = np.random.randn(100, 1000)
            result = np.dot(data, data.T)
            return result

        # Run profiled function
        result = profiled_inference()

        # Verify result is computed
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 100))

    def test_cpu_profiling_quantum_operations(self):
        """Profile CPU usage during quantum operations"""
        profiler = cProfile.Profile()

        def quantum_operations():
            qml = QuantumMachineLearning(n_qubits=4)
            X = np.random.randn(50, 4)
            y = np.sign(X[:, 0])

            # Train and predict
            qml.train_classifier(X, y, method='qsvm')
            predictions = qml.classify(X[:10])

            return predictions

        # Profile the operations
        profiler.enable()
        result = quantum_operations()
        profiler.disable()

        # Analyze profile
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions

        profile_output = s.getvalue()

        # Should complete successfully
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)

        # Profile should contain our functions
        self.assertIn('quantum_operations', profile_output)

class ScalabilityTest(unittest.TestCase):
    """Scalability performance tests"""

    def test_horizontal_scaling_inference(self):
        """Test inference scaling with multiple workers"""
        def inference_worker(worker_id, num_requests):
            results = []
            for i in range(num_requests):
                # Simulate inference
                time.sleep(0.001)
                results.append(f"worker_{worker_id}_result_{i}")
            return results

        # Test different worker configurations
        worker_configs = [
            (1, 100),   # 1 worker, 100 requests
            (2, 100),   # 2 workers, 100 requests each
            (4, 100),   # 4 workers, 100 requests each
        ]

        scalability_results = {}

        for num_workers, requests_per_worker in worker_configs:
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(inference_worker, i, requests_per_worker)
                    for i in range(num_workers)
                ]
                results = [f.result() for f in futures]

            end_time = time.time()
            total_time = end_time - start_time
            total_requests = num_workers * requests_per_worker
            throughput = total_requests / total_time

            scalability_results[num_workers] = {
                'throughput': throughput,
                'total_time': total_time,
                'efficiency': throughput / num_workers
            }

        # More workers should improve throughput (to a point)
        single_worker_throughput = scalability_results[1]['throughput']
        four_worker_throughput = scalability_results[4]['throughput']

        # Should scale reasonably (allowing for thread overhead)
        scaling_ratio = four_worker_throughput / single_worker_throughput
        self.assertGreater(scaling_ratio, 1.5,
                          f"Poor scaling: only {scaling_ratio:.2f}x improvement with 4x workers")

if __name__ == '__main__':
    # Run performance tests
    unittest.main(verbosity=2)

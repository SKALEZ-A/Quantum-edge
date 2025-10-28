"""
Unit tests for Privacy and Security modules
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time
import json
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from privacy_security.privacy import (
    DifferentialPrivacy, LocalDifferentialPrivacy, FederatedPrivacy,
    QuantumPrivacy, PrivacyEngine, PrivacyMechanism, PrivacyLevel, PrivacyBudget
)
from privacy_security.encryption import (
    QuantumSafeEncryption, HomomorphicEncryption, SecureKeyManagement
)
from privacy_security.authentication import (
    MultiFactorAuthentication, BiometricAuthentication, QuantumAuthentication
)
from privacy_security.access_control import (
    RoleBasedAccessControl, AttributeBasedAccessControl,
    QuantumAccessControl, AccessControlEngine
)
from tests import TestUtils

class TestDifferentialPrivacy(unittest.TestCase):
    """Test Differential Privacy implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

    def test_dp_initialization(self):
        """Test DP initialization"""
        self.assertEqual(self.dp.epsilon, 1.0)
        self.assertEqual(self.dp.delta, 1e-5)
        self.assertIsNotNone(self.dp.privacy_budget)

    def test_gaussian_noise_addition(self):
        """Test Gaussian noise addition"""
        value = 5.0
        noisy_value = self.dp.add_gaussian_noise(value)

        # Value should be close to original but different
        self.assertNotEqual(noisy_value, value)
        self.assertIsInstance(noisy_value, (int, float, np.number))

    def test_laplace_noise_addition(self):
        """Test Laplace noise addition"""
        value = 10.0
        noisy_value = self.dp.add_laplace_noise(value)

        # Value should be close to original but different
        self.assertNotEqual(noisy_value, value)
        self.assertIsInstance(noisy_value, (int, float, np.number))

    def test_privacy_histogram(self):
        """Test private histogram"""
        counts = np.array([10, 20, 15, 5])

        private_histogram = self.dp.privatize_histogram(counts)

        # Shape should be preserved
        self.assertEqual(private_histogram.shape, counts.shape)

        # Values should be close but not exact
        self.assertFalse(np.array_equal(private_histogram, counts))

    def test_private_mean(self):
        """Test private mean computation"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        private_mean = self.dp.privatize_mean(values)

        # Should be close to true mean (3.0)
        self.assertAlmostEqual(private_mean, 3.0, delta=2.0)

    def test_exponential_mechanism(self):
        """Test exponential mechanism"""
        candidates = ['A', 'B', 'C']
        scores = [1.0, 2.0, 0.5]

        result = self.dp.exponential_mechanism(candidates, scores)

        # Result should be one of the candidates
        self.assertIn(result, candidates)

class TestLocalDifferentialPrivacy(unittest.TestCase):
    """Test Local Differential Privacy"""

    def setUp(self):
        """Set up test fixtures"""
        self.ldp = LocalDifferentialPrivacy(epsilon=1.0)

    def test_randomized_response(self):
        """Test randomized response"""
        true_value = True

        # Test multiple times to see randomization
        responses = [self.ldp.randomized_response(true_value) for _ in range(100)]

        # Should get some true and some false responses
        true_count = sum(responses)
        false_count = len(responses) - true_count

        # Both should be non-zero (randomization working)
        self.assertGreater(true_count, 0)
        self.assertGreater(false_count, 0)

    def test_hadamard_response(self):
        """Test Hadamard response"""
        true_value = 1
        domain_size = 4

        response = self.ldp.hadamard_response(true_value, domain_size)

        # Response should be in valid range
        self.assertGreaterEqual(response, 0)
        self.assertLess(response, domain_size)

class TestFederatedPrivacy(unittest.TestCase):
    """Test Federated Privacy"""

    def setUp(self):
        """Set up test fixtures"""
        self.fp = FederatedPrivacy(num_clients=5, epsilon=1.0)

    def test_fp_initialization(self):
        """Test FP initialization"""
        self.assertEqual(self.fp.num_clients, 5)
        self.assertEqual(self.fp.epsilon, 1.0)

    def test_client_privacy_budget(self):
        """Test client privacy budget initialization"""
        client_id = "client_001"

        self.fp.initialize_client_privacy(client_id)

        self.assertIn(client_id, self.fp.client_privacy_budgets)
        budget = self.fp.client_privacy_budgets[client_id]
        self.assertEqual(budget.epsilon, self.fp.epsilon / self.fp.num_clients)

    def test_noise_to_gradients(self):
        """Test adding noise to gradients"""
        client_id = "client_001"
        gradients = np.random.randn(100, 50)

        self.fp.initialize_client_privacy(client_id)
        noisy_gradients = self.fp.add_noise_to_gradients(gradients, client_id)

        # Shape should be preserved
        self.assertEqual(noisy_gradients.shape, gradients.shape)

        # Gradients should be different
        self.assertFalse(np.array_equal(noisy_gradients, gradients))

    def test_secure_aggregation(self):
        """Test secure aggregation"""
        client_updates = [np.random.randn(10, 5) for _ in range(5)]

        aggregated = self.fp.secure_aggregation(client_updates)

        # Shape should match individual updates
        self.assertEqual(aggregated.shape, client_updates[0].shape)

class TestQuantumPrivacy(unittest.TestCase):
    """Test Quantum Privacy"""

    def setUp(self):
        """Set up test fixtures"""
        self.qp = QuantumPrivacy(n_qubits=4)

    def test_quantum_obfuscation(self):
        """Test quantum obfuscation"""
        data = np.random.randn(10, 5)

        obfuscated = self.qp.quantum_obfuscation(data)

        # Shape should be preserved
        self.assertEqual(obfuscated.shape, data.shape)

        # Data should be different
        self.assertFalse(np.array_equal(obfuscated, data))

    def test_quantum_random_vector(self):
        """Test quantum random vector generation"""
        size = 10
        vector = self.qp.quantum_random_vector(size)

        # Check size
        self.assertEqual(len(vector), size)

        # Check it's numeric
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in vector))

    def test_quantum_anonymization(self):
        """Test quantum anonymization"""
        dataset = np.random.randn(20, 5)

        anonymized = self.qp.quantum_anonymization(dataset)

        # Shape should be preserved
        self.assertEqual(anonymized.shape, dataset.shape)

        # Data should be different
        self.assertFalse(np.array_equal(anonymized, dataset))

class TestPrivacyEngine(unittest.TestCase):
    """Test Privacy Engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = PrivacyEngine(
            mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
            epsilon=1.0
        )

    def test_privacy_engine_initialization(self):
        """Test privacy engine initialization"""
        self.assertEqual(self.engine.mechanism, PrivacyMechanism.DIFFERENTIAL_PRIVACY)
        self.assertEqual(self.engine.epsilon, 1.0)
        self.assertIsNotNone(self.engine.dp)

    def test_privacy_application(self):
        """Test applying privacy mechanism"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        privatized_data, report = self.engine.apply_privacy(data)

        # Check data is privatized
        self.assertIsNotNone(privatized_data)
        self.assertFalse(np.array_equal(privatized_data, data))

        # Check report
        self.assertIsInstance(report, object)  # PrivacyReport
        self.assertEqual(report.mechanism, PrivacyMechanism.DIFFERENTIAL_PRIVACY)

    def test_compliance_check(self):
        """Test compliance checking"""
        data_usage = "data_processing_with_consent_and_encryption"

        compliance = self.engine.check_compliance(data_usage)

        # Should return compliance dictionary
        self.assertIsInstance(compliance, dict)
        self.assertIn('gdpr_compliant', compliance)
        self.assertIn('ccpa_compliant', compliance)
        self.assertIn('hipaa_compliant', compliance)

class TestQuantumSafeEncryption(unittest.TestCase):
    """Test Quantum-Safe Encryption"""

    def setUp(self):
        """Set up test fixtures"""
        self.qse = QuantumSafeEncryption()

    def test_key_generation(self):
        """Test key generation"""
        private_key, public_key = self.qse.generate_keypair()

        # Keys should be bytes
        self.assertIsInstance(private_key, bytes)
        self.assertIsInstance(public_key, bytes)

        # Keys should be different
        self.assertNotEqual(private_key, public_key)

    def test_encryption_decryption(self):
        """Test encryption and decryption"""
        # Generate keys
        private_key, public_key = self.qse.generate_keypair()

        # Test data
        plaintext = b"Hello, Quantum World!"

        # Encrypt
        ciphertext = self.qse.encrypt(plaintext, public_key)

        # Should be different from plaintext
        self.assertNotEqual(ciphertext, plaintext)

        # Decrypt
        decrypted = self.qse.decrypt(ciphertext, private_key)

        # Should match original
        self.assertEqual(decrypted, plaintext)

class TestHomomorphicEncryption(unittest.TestCase):
    """Test Homomorphic Encryption"""

    def setUp(self):
        """Set up test fixtures"""
        self.he = HomomorphicEncryption()

    def test_homomorphic_addition(self):
        """Test homomorphic addition"""
        a = 5
        b = 3

        # Encrypt values
        enc_a = self.he.encrypt(a)
        enc_b = self.he.encrypt(b)

        # Add encrypted values
        enc_sum = self.he.add(enc_a, enc_b)

        # Decrypt result
        result = self.he.decrypt(enc_sum)

        # Should equal a + b
        self.assertEqual(result, a + b)

    def test_homomorphic_multiplication(self):
        """Test homomorphic multiplication"""
        a = 4
        b = 3

        # Encrypt values
        enc_a = self.he.encrypt(a)
        enc_b = self.he.encrypt(b)

        # Multiply encrypted values
        enc_product = self.he.multiply(enc_a, enc_b)

        # Decrypt result
        result = self.he.decrypt(enc_product)

        # Should equal a * b
        self.assertEqual(result, a * b)

class TestMultiFactorAuthentication(unittest.TestCase):
    """Test Multi-Factor Authentication"""

    def setUp(self):
        """Set up test fixtures"""
        self.mfa = MultiFactorAuthentication()

    def test_mfa_setup(self):
        """Test MFA setup"""
        user_id = "user_001"

        # Setup MFA
        secret = self.mfa.setup_mfa(user_id)

        # Should return a secret
        self.assertIsInstance(secret, str)
        self.assertGreater(len(secret), 0)

    def test_mfa_verification(self):
        """Test MFA verification"""
        user_id = "user_001"

        # Setup MFA
        self.mfa.setup_mfa(user_id)

        # Generate valid code (in real scenario, this would come from authenticator app)
        code = self.mfa.generate_totp_code(user_id)

        # Verify code
        is_valid = self.mfa.verify_totp(user_id, code)

        # Should be valid immediately after generation
        self.assertTrue(is_valid)

class TestRoleBasedAccessControl(unittest.TestCase):
    """Test Role-Based Access Control"""

    def setUp(self):
        """Set up test fixtures"""
        self.rbac = RoleBasedAccessControl()

    def test_role_creation(self):
        """Test role creation"""
        role_name = "admin"
        permissions = ["read", "write", "delete"]

        self.rbac.create_role(role_name, permissions)

        # Check role exists
        self.assertIn(role_name, self.rbac.roles)
        self.assertEqual(self.rbac.roles[role_name], permissions)

    def test_user_assignment(self):
        """Test user role assignment"""
        user_id = "user_001"
        role_name = "admin"

        self.rbac.create_role(role_name, ["read", "write"])
        self.rbac.assign_role(user_id, role_name)

        # Check assignment
        self.assertIn(user_id, self.rbac.user_roles)
        self.assertIn(role_name, self.rbac.user_roles[user_id])

    def test_permission_check(self):
        """Test permission checking"""
        user_id = "user_001"
        role_name = "admin"

        self.rbac.create_role(role_name, ["read", "write"])
        self.rbac.assign_role(user_id, role_name)

        # Check permissions
        self.assertTrue(self.rbac.check_permission(user_id, "read"))
        self.assertTrue(self.rbac.check_permission(user_id, "write"))
        self.assertFalse(self.rbac.check_permission(user_id, "delete"))

class TestAttributeBasedAccessControl(unittest.TestCase):
    """Test Attribute-Based Access Control"""

    def setUp(self):
        """Set up test fixtures"""
        self.abac = AttributeBasedAccessControl()

    def test_policy_creation(self):
        """Test policy creation"""
        policy_name = "data_access_policy"
        conditions = {
            'department': 'engineering',
            'clearance_level': 'high'
        }
        permissions = ["read", "write"]

        self.abac.create_policy(policy_name, conditions, permissions)

        # Check policy exists
        self.assertIn(policy_name, self.abac.policies)

    def test_attribute_evaluation(self):
        """Test attribute evaluation"""
        policy_name = "data_access_policy"

        # Create policy
        conditions = {'department': 'engineering', 'clearance_level': 'high'}
        self.abac.create_policy(policy_name, conditions, ["read"])

        # Test attributes
        user_attrs = {'department': 'engineering', 'clearance_level': 'high'}
        resource_attrs = {'sensitivity': 'high'}

        has_access = self.abac.evaluate_access(user_attrs, resource_attrs, policy_name)

        # Should have access
        self.assertTrue(has_access)

class TestAccessControlEngine(unittest.TestCase):
    """Test Access Control Engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.engine = AccessControlEngine()

    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.rbac)
        self.assertIsNotNone(self.engine.abac)
        self.assertIsNotNone(self.engine.qac)

    def test_access_decision(self):
        """Test access decision making"""
        user_id = "user_001"
        resource = "sensitive_data"
        action = "read"

        # Setup RBAC
        self.engine.rbac.create_role("analyst", ["read"])
        self.engine.rbac.assign_role(user_id, "analyst")

        # Make access decision
        decision = self.engine.make_access_decision(user_id, resource, action)

        # Should allow access
        self.assertTrue(decision['allowed'])
        self.assertEqual(decision['method'], 'rbac')

class TestPrivacySecurityIntegration(unittest.TestCase):
    """Integration tests for privacy and security"""

    def test_complete_privacy_pipeline(self):
        """Test complete privacy protection pipeline"""
        # Create test data
        sensitive_data = np.random.randn(100, 10)

        # Initialize privacy engine
        privacy_engine = PrivacyEngine(epsilon=0.5)

        # Apply privacy protection
        privatized_data, report = privacy_engine.apply_privacy(sensitive_data)

        # Check that data is protected
        self.assertIsNotNone(privatized_data)
        self.assertIsNotNone(report)

        # Check utility preservation (data should still be usable)
        self.assertEqual(privatized_data.shape, sensitive_data.shape)

    def test_end_to_end_security(self):
        """Test end-to-end security workflow"""
        # Setup encryption
        qse = QuantumSafeEncryption()
        private_key, public_key = qse.generate_keypair()

        # Setup authentication
        mfa = MultiFactorAuthentication()
        user_id = "test_user"
        mfa.setup_mfa(user_id)

        # Setup access control
        rbac = RoleBasedAccessControl()
        rbac.create_role("user", ["read"])
        rbac.assign_role(user_id, "user")

        # Test complete workflow
        data = b"Sensitive information"

        # Encrypt data
        encrypted = qse.encrypt(data, public_key)

        # Verify access control
        has_access = rbac.check_permission(user_id, "read")
        self.assertTrue(has_access)

        # Decrypt data
        decrypted = qse.decrypt(encrypted, private_key)
        self.assertEqual(decrypted, data)

class TestPrivacySecurityPerformance(unittest.TestCase):
    """Performance tests for privacy and security"""

    def test_encryption_performance(self):
        """Test encryption performance"""
        qse = QuantumSafeEncryption()
        _, public_key = qse.generate_keypair()

        data = b"A" * 1000  # 1KB of data

        # Measure encryption time
        _, encrypt_time = TestUtils.measure_execution_time(
            qse.encrypt, data, public_key
        )

        # Should be reasonably fast
        TestUtils.assert_performance_threshold(
            qse.encrypt, 100.0, data, public_key  # 100ms max
        )

    def test_privacy_mechanism_performance(self):
        """Test privacy mechanism performance"""
        dp = DifferentialPrivacy(epsilon=1.0)
        data = np.random.randn(1000)

        # Measure noise addition time
        _, noise_time = TestUtils.measure_execution_time(
            dp.add_gaussian_noise, data
        )

        # Should be fast
        TestUtils.assert_performance_threshold(
            dp.add_gaussian_noise, 10.0, data  # 10ms max
        )

    def test_access_control_performance(self):
        """Test access control performance"""
        rbac = RoleBasedAccessControl()

        # Setup test data
        rbac.create_role("test_role", ["read", "write", "delete"])
        for i in range(100):
            rbac.assign_role(f"user_{i}", "test_role")

        # Measure permission check time
        _, check_time = TestUtils.measure_execution_time(
            rbac.check_permission, "user_50", "read"
        )

        # Should be very fast
        TestUtils.assert_performance_threshold(
            rbac.check_permission, 1.0, "user_50", "read"  # 1ms max
        )

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

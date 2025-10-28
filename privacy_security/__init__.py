"""
Quantum Edge AI Platform - Privacy & Security Module

Comprehensive privacy-preserving and security frameworks for quantum edge AI,
including encryption, access control, compliance, and threat detection.
"""

__version__ = "1.0.0"
__author__ = "Quantum Edge AI Platform Team"

from .encryption import QuantumEncryption, HomomorphicEncryption, SecureMultiParty
from .authentication import MultiFactorAuth, BiometricAuth, ZeroTrustAuth
from .access_control import AttributeBasedAccess, PolicyBasedAccess, ContextAwareAccess
from .compliance import GDPRCompliance, HIPAACompliance, SOC2Compliance
from .threat_detection import IntrusionDetection, AnomalyDetection, BehavioralAnalysis
from .audit import SecurityAudit, ComplianceAudit, AccessAudit
from .privacy import DifferentialPrivacy, FederatedPrivacy, QuantumPrivacy
from .key_management import QuantumKeyDistribution, KeyRotation, SecureKeyStorage

__all__ = [
    'QuantumEncryption', 'HomomorphicEncryption', 'SecureMultiParty',
    'MultiFactorAuth', 'BiometricAuth', 'ZeroTrustAuth',
    'AttributeBasedAccess', 'PolicyBasedAccess', 'ContextAwareAccess',
    'GDPRCompliance', 'HIPAACompliance', 'SOC2Compliance',
    'IntrusionDetection', 'AnomalyDetection', 'BehavioralAnalysis',
    'SecurityAudit', 'ComplianceAudit', 'AccessAudit',
    'DifferentialPrivacy', 'FederatedPrivacy', 'QuantumPrivacy',
    'QuantumKeyDistribution', 'KeyRotation', 'SecureKeyStorage'
]

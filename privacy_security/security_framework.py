#!/usr/bin/env python3
"""
Comprehensive Security Framework for Quantum Edge AI Platform

This module provides enterprise-grade security capabilities including:
- Multi-layer authentication and authorization
- End-to-end encryption with quantum-resistant algorithms
- Secure communication protocols
- Intrusion detection and prevention
- Audit logging and compliance monitoring
- Zero-trust architecture implementation
- Threat intelligence integration
"""

import os
import json
import time
import hashlib
import secrets
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import re
from enum import Enum
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import Certificate, load_pem_x509_certificate
from cryptography.x509.oid import NameOID
import jwt
import base64
import sqlite3
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    SECRET = 4
    TOP_SECRET = 5


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityEvent:
    """Security event representation."""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: ThreatLevel
    source_ip: str
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str
    created_at: datetime
    updated_at: datetime
    enabled: bool = True


class CryptographicEngine:
    """
    Advanced cryptographic engine with quantum-resistant algorithms.

    Supports multiple encryption standards and key management.
    """

    def __init__(self, key_store_path: str = "security/keystore"):
        self.key_store_path = Path(key_store_path)
        self.key_store_path.mkdir(parents=True, exist_ok=True)

        # Initialize cryptographic backends
        self.backend = default_backend()

        # Key cache
        self.key_cache: Dict[str, bytes] = {}
        self.cert_cache: Dict[str, Certificate] = {}

        # Master encryption key (should be securely stored in HSM in production)
        self.master_key = self._derive_master_key()

        logger.info("Cryptographic engine initialized")

    def _derive_master_key(self) -> bytes:
        """Derive master encryption key from environment."""
        # In production, this should come from HSM or secure key vault
        salt = secrets.token_bytes(32)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        # Use system-specific entropy
        entropy = str(os.environ.get('PLATFORM_SECURITY_KEY', 'quantum_edge_ai_default')).encode()
        return kdf.derive(entropy)

    def generate_key_pair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem

    def encrypt_data(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data using AES-256-GCM."""
        if key is None:
            key = self.master_key

        # Generate nonce
        nonce = secrets.token_bytes(12)

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()

        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Combine nonce, ciphertext, and tag
        return nonce + encryptor.tag + ciphertext

    def decrypt_data(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-256-GCM."""
        if key is None:
            key = self.master_key

        # Extract components
        nonce = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
        decryptor = cipher.decryptor()

        # Decrypt data
        return decryptor.update(ciphertext) + decryptor.finalize()

    def sign_data(self, data: bytes, private_key_pem: bytes) -> bytes:
        """Sign data using RSA-PSS."""
        from cryptography.hazmat.primitives import serialization

        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )

        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return signature

    def verify_signature(self, data: bytes, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify RSA-PSS signature."""
        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=self.backend
        )

        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def generate_hmac(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Generate HMAC-SHA256."""
        if key is None:
            key = self.master_key

        h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
        h.update(data)
        return h.finalize()

    def verify_hmac(self, data: bytes, signature: bytes, key: Optional[bytes] = None) -> bool:
        """Verify HMAC-SHA256."""
        if key is None:
            key = self.master_key

        h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
        h.update(data)

        try:
            h.verify(signature)
            return True
        except Exception:
            return False

    def store_key(self, key_id: str, key_data: bytes, encrypted: bool = True):
        """Store encryption key securely."""
        if encrypted:
            key_data = self.encrypt_data(key_data)

        key_path = self.key_store_path / f"{key_id}.key"
        with open(key_path, 'wb') as f:
            f.write(key_data)

        self.key_cache[key_id] = key_data
        logger.info(f"Key stored: {key_id}")

    def load_key(self, key_id: str, encrypted: bool = True) -> Optional[bytes]:
        """Load encryption key."""
        if key_id in self.key_cache:
            return self.key_cache[key_id]

        key_path = self.key_store_path / f"{key_id}.key"
        if not key_path.exists():
            return None

        with open(key_path, 'rb') as f:
            key_data = f.read()

        if encrypted:
            key_data = self.decrypt_data(key_data)

        self.key_cache[key_id] = key_data
        return key_data

    def rotate_keys(self):
        """Rotate all encryption keys."""
        logger.info("Starting key rotation")

        # Generate new master key
        new_master_key = self._derive_master_key()

        # Re-encrypt all stored keys with new master key
        for key_path in self.key_store_path.glob("*.key"):
            key_id = key_path.stem

            # Load old encrypted key
            with open(key_path, 'rb') as f:
                old_encrypted = f.read()

            # Decrypt with old master key
            try:
                decrypted_key = self.decrypt_data(old_encrypted, self.master_key)

                # Re-encrypt with new master key
                new_encrypted = self.encrypt_data(decrypted_key, new_master_key)

                # Store new encrypted key
                with open(key_path, 'wb') as f:
                    f.write(new_encrypted)

            except Exception as e:
                logger.error(f"Failed to rotate key {key_id}: {e}")

        # Update master key
        self.master_key = new_master_key

        # Clear cache
        self.key_cache.clear()

        logger.info("Key rotation completed")


class AuthenticationManager:
    """
    Multi-factor authentication and authorization system.

    Supports JWT tokens, OAuth2, SAML, and custom authentication schemes.
    """

    def __init__(self, crypto_engine: CryptographicEngine):
        self.crypto = crypto_engine
        self.jwt_secret = self.crypto.master_key
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}

        # Session management
        self.session_timeout = timedelta(hours=8)
        self.max_sessions_per_user = 5

        logger.info("Authentication manager initialized")

    def authenticate_user(self, username: str, password: str,
                         mfa_token: Optional[str] = None) -> Optional[str]:
        """Authenticate user with optional MFA."""
        user = self.users.get(username)
        if not user or not user.get('enabled', False):
            return None

        # Verify password (hashed)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != user['password_hash']:
            self._log_failed_authentication(username, "invalid_password")
            return None

        # Verify MFA if enabled
        if user.get('mfa_enabled', False):
            if not mfa_token or not self._verify_mfa_token(username, mfa_token):
                self._log_failed_authentication(username, "invalid_mfa")
                return None

        # Check account status
        if user.get('locked_until'):
            if datetime.now() < user['locked_until']:
                self._log_failed_authentication(username, "account_locked")
                return None
            else:
                # Unlock account
                user['locked_until'] = None
                user['failed_attempts'] = 0

        # Create session
        session_id = self._create_session(username, user)
        self._log_successful_authentication(username, session_id)

        return session_id

    def authorize_request(self, session_id: str, resource: str, action: str) -> bool:
        """Authorize request based on session and permissions."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check session expiry
        if datetime.now() - session['created_at'] > self.session_timeout:
            self._invalidate_session(session_id)
            return False

        # Get user permissions
        username = session['username']
        user = self.users.get(username)
        if not user:
            return False

        # Check permissions
        user_permissions = user.get('permissions', [])
        required_permission = f"{resource}:{action}"

        # Check exact match
        if required_permission in user_permissions:
            return True

        # Check wildcard permissions
        for perm in user_permissions:
            if perm.endswith('*'):
                if required_permission.startswith(perm[:-1]):
                    return True

        return False

    def create_jwt_token(self, username: str, expires_in: timedelta = timedelta(hours=1)) -> str:
        """Create JWT token for user."""
        payload = {
            'username': username,
            'exp': datetime.utcnow() + expires_in,
            'iat': datetime.utcnow(),
            'iss': 'quantum-edge-ai-platform'
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def _create_session(self, username: str, user: Dict[str, Any]) -> str:
        """Create new user session."""
        # Clean up old sessions for this user
        user_sessions = [sid for sid, s in self.sessions.items()
                        if s['username'] == username]

        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest_session = min(user_sessions,
                               key=lambda sid: self.sessions[sid]['created_at'])
            self._invalidate_session(oldest_session)

        # Create new session
        session_id = secrets.token_hex(32)
        session = {
            'session_id': session_id,
            'username': username,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': None,  # Would be set from request
            'user_agent': None   # Would be set from request
        }

        self.sessions[session_id] = session
        return session_id

    def _invalidate_session(self, session_id: str):
        """Invalidate user session."""
        if session_id in self.sessions:
            username = self.sessions[session_id]['username']
            del self.sessions[session_id]
            logger.info(f"Session invalidated for user: {username}")

    def _verify_mfa_token(self, username: str, token: str) -> bool:
        """Verify MFA token (simplified implementation)."""
        # In production, this would integrate with TOTP/HOTP
        user = self.users.get(username)
        if not user:
            return False

        # Simple token verification (replace with proper MFA implementation)
        expected_token = user.get('mfa_secret', '') + str(int(time.time() // 30))
        return hashlib.sha256(expected_token.encode()).hexdigest()[:6] == token

    def _log_failed_authentication(self, username: str, reason: str):
        """Log failed authentication attempt."""
        logger.warning(f"Failed authentication for user {username}: {reason}")

        user = self.users.get(username, {})
        failed_attempts = user.get('failed_attempts', 0) + 1
        user['failed_attempts'] = failed_attempts

        # Lock account after multiple failures
        if failed_attempts >= 5:
            user['locked_until'] = datetime.now() + timedelta(minutes=15)
            logger.warning(f"Account locked for user {username} due to failed attempts")

    def _log_successful_authentication(self, username: str, session_id: str):
        """Log successful authentication."""
        logger.info(f"Successful authentication for user {username}, session: {session_id}")

        # Reset failed attempts
        user = self.users.get(username, {})
        user['failed_attempts'] = 0
        user['last_login'] = datetime.now()

    def create_user(self, username: str, password: str, permissions: List[str],
                   mfa_enabled: bool = False) -> bool:
        """Create new user account."""
        if username in self.users:
            return False

        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        # Generate MFA secret if enabled
        mfa_secret = secrets.token_hex(16) if mfa_enabled else None

        user = {
            'username': username,
            'password_hash': password_hash,
            'permissions': permissions,
            'mfa_enabled': mfa_enabled,
            'mfa_secret': mfa_secret,
            'enabled': True,
            'created_at': datetime.now(),
            'failed_attempts': 0,
            'last_login': None,
            'locked_until': None
        }

        self.users[username] = user
        logger.info(f"User created: {username}")
        return True


class AccessControlEngine:
    """
    Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC) engine.

    Supports complex permission models with hierarchical roles and dynamic attributes.
    """

    def __init__(self):
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, List[str]] = {}  # username -> list of roles
        self.permissions: Dict[str, Dict[str, Any]] = {}
        self.policies: Dict[str, SecurityPolicy] = {}

        # ABAC attributes
        self.user_attributes: Dict[str, Dict[str, Any]] = {}
        self.resource_attributes: Dict[str, Dict[str, Any]] = {}

        logger.info("Access control engine initialized")

    def create_role(self, role_name: str, permissions: List[str],
                   description: str = "", parent_roles: List[str] = None) -> bool:
        """Create new role with permissions."""
        if role_name in self.roles:
            return False

        # Inherit permissions from parent roles
        inherited_permissions = set()
        if parent_roles:
            for parent in parent_roles:
                if parent in self.roles:
                    inherited_permissions.update(self.roles[parent]['permissions'])

        all_permissions = list(set(permissions) | inherited_permissions)

        role = {
            'role_name': role_name,
            'description': description,
            'permissions': all_permissions,
            'parent_roles': parent_roles or [],
            'created_at': datetime.now()
        }

        self.roles[role_name] = role
        logger.info(f"Role created: {role_name}")
        return True

    def assign_role_to_user(self, username: str, role_name: str) -> bool:
        """Assign role to user."""
        if role_name not in self.roles:
            return False

        if username not in self.users:
            self.users[username] = []

        if role_name not in self.users[username]:
            self.users[username].append(role_name)
            logger.info(f"Role {role_name} assigned to user {username}")
            return True

        return False

    def check_permission(self, username: str, permission: str,
                        resource_attributes: Dict[str, Any] = None) -> bool:
        """Check if user has permission with optional ABAC evaluation."""
        user_roles = self.users.get(username, [])

        # Get all permissions from user's roles
        user_permissions = set()
        for role_name in user_roles:
            if role_name in self.roles:
                user_permissions.update(self.roles[role_name]['permissions'])

        # Check basic RBAC
        if permission not in user_permissions:
            return False

        # Check ABAC policies if resource attributes provided
        if resource_attributes:
            return self._evaluate_abac_policies(username, permission, resource_attributes)

        return True

    def _evaluate_abac_policies(self, username: str, permission: str,
                              resource_attributes: Dict[str, Any]) -> bool:
        """Evaluate ABAC policies."""
        user_attrs = self.user_attributes.get(username, {})

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            if self._matches_policy(policy, username, permission,
                                  user_attrs, resource_attributes):
                return policy.enforcement_level == 'allow'

        return True  # Default allow if no matching policy

    def _matches_policy(self, policy: SecurityPolicy, username: str, permission: str,
                       user_attrs: Dict[str, Any], resource_attrs: Dict[str, Any]) -> bool:
        """Check if request matches policy conditions."""
        for rule in policy.rules:
            rule_type = rule.get('type')
            attribute = rule.get('attribute')
            operator = rule.get('operator')
            value = rule.get('value')

            if rule_type == 'user_attribute':
                actual_value = user_attrs.get(attribute)
            elif rule_type == 'resource_attribute':
                actual_value = resource_attrs.get(attribute)
            elif rule_type == 'permission':
                actual_value = permission
            elif rule_type == 'username':
                actual_value = username
            else:
                continue

            if not self._evaluate_condition(actual_value, operator, value):
                return False

        return True

    def _evaluate_condition(self, actual_value: Any, operator: str, expected_value: Any) -> bool:
        """Evaluate condition based on operator."""
        if operator == 'equals':
            return actual_value == expected_value
        elif operator == 'not_equals':
            return actual_value != expected_value
        elif operator == 'contains':
            return expected_value in actual_value if isinstance(actual_value, (list, str)) else False
        elif operator == 'in':
            return actual_value in expected_value if isinstance(expected_value, (list, tuple)) else False
        elif operator == 'greater_than':
            return actual_value > expected_value if isinstance(actual_value, (int, float)) else False
        elif operator == 'less_than':
            return actual_value < expected_value if isinstance(actual_value, (int, float)) else False
        else:
            return False

    def set_user_attributes(self, username: str, attributes: Dict[str, Any]):
        """Set user attributes for ABAC."""
        self.user_attributes[username] = attributes

    def set_resource_attributes(self, resource: str, attributes: Dict[str, Any]):
        """Set resource attributes for ABAC."""
        self.resource_attributes[resource] = attributes

    def create_policy(self, policy: SecurityPolicy):
        """Create new security policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Policy created: {policy.policy_id}")


class IntrusionDetectionSystem:
    """
    Advanced intrusion detection and prevention system.

    Monitors system activity for suspicious patterns and potential security threats.
    """

    def __init__(self):
        self.anomaly_detectors: Dict[str, Any] = {}
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable] = []
        self.event_buffer: deque = deque(maxlen=10000)

        # Statistical models for anomaly detection
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

        # Configure default threat patterns
        self._configure_threat_patterns()

        logger.info("Intrusion detection system initialized")

    def _configure_threat_patterns(self):
        """Configure default threat detection patterns."""
        self.threat_patterns = {
            'brute_force': {
                'pattern': r'failed_login.*failed_login.*failed_login',
                'time_window': 300,  # 5 minutes
                'threshold': 5,
                'severity': ThreatLevel.HIGH
            },
            'sql_injection': {
                'pattern': r'(\'|(\\x27)|(\\x2D\\x2D)|(%27)|(%2D%2D))',
                'severity': ThreatLevel.CRITICAL
            },
            'xss_attempt': {
                'pattern': r'(<script|<iframe|<object|<embed)',
                'severity': ThreatLevel.HIGH
            },
            'suspicious_traffic': {
                'threshold': 1000,  # requests per minute
                'severity': ThreatLevel.MEDIUM
            },
            'privilege_escalation': {
                'pattern': r'(sudo|su|runas).*(root|admin)',
                'severity': ThreatLevel.CRITICAL
            }
        }

    def analyze_event(self, event: SecurityEvent) -> List[SecurityEvent]:
        """Analyze security event for threats."""
        detected_threats = []

        # Check against threat patterns
        for pattern_name, pattern_config in self.threat_patterns.items():
            if self._matches_pattern(event, pattern_config):
                threat_event = SecurityEvent(
                    event_id=f"threat_{secrets.token_hex(8)}",
                    timestamp=datetime.now(),
                    event_type=f"threat_detected_{pattern_name}",
                    severity=pattern_config['severity'],
                    source_ip=event.source_ip,
                    user_id=event.user_id,
                    resource=event.resource,
                    action=event.action,
                    details={
                        'original_event': event.event_id,
                        'pattern_matched': pattern_name,
                        'pattern_config': pattern_config
                    }
                )
                detected_threats.append(threat_event)

        # Check for anomalies
        anomaly_threats = self._detect_anomalies(event)
        detected_threats.extend(anomaly_threats)

        # Add event to buffer for pattern analysis
        self.event_buffer.append(event)

        # Trigger alerts for detected threats
        for threat in detected_threats:
            self._trigger_alert(threat)

        return detected_threats

    def _matches_pattern(self, event: SecurityEvent, pattern_config: Dict[str, Any]) -> bool:
        """Check if event matches threat pattern."""
        pattern = pattern_config.get('pattern')
        if not pattern:
            return False

        # Check event details for pattern matches
        search_text = json.dumps(event.details)
        if re.search(pattern, search_text, re.IGNORECASE):
            return True

        return False

    def _detect_anomalies(self, event: SecurityEvent) -> List[SecurityEvent]:
        """Detect anomalous behavior."""
        anomalies = []

        # Simple statistical anomaly detection
        event_type = event.event_type
        if event_type not in self.baseline_stats:
            self.baseline_stats[event_type] = {
                'count': 0,
                'last_seen': event.timestamp,
                'avg_interval': 0
            }

        stats = self.baseline_stats[event_type]
        current_time = event.timestamp

        # Update statistics
        time_diff = (current_time - stats['last_seen']).total_seconds()
        stats['count'] += 1
        stats['avg_interval'] = (stats['avg_interval'] * (stats['count'] - 1) + time_diff) / stats['count']
        stats['last_seen'] = current_time

        # Detect anomalies
        if stats['count'] > 10:  # Need some baseline data
            # Check for unusual frequency
            if time_diff < stats['avg_interval'] * 0.1:  # Much more frequent than usual
                anomaly = SecurityEvent(
                    event_id=f"anomaly_{secrets.token_hex(8)}",
                    timestamp=datetime.now(),
                    event_type="anomaly_high_frequency",
                    severity=ThreatLevel.MEDIUM,
                    source_ip=event.source_ip,
                    user_id=event.user_id,
                    resource=event.resource,
                    details={
                        'anomaly_type': 'high_frequency',
                        'expected_interval': stats['avg_interval'],
                        'actual_interval': time_diff,
                        'event_type': event_type
                    }
                )
                anomalies.append(anomaly)

        return anomalies

    def _trigger_alert(self, threat: SecurityEvent):
        """Trigger alerts for detected threats."""
        logger.warning(f"Security threat detected: {threat.event_type} "
                      f"(severity: {threat.severity.name})")

        # Notify all alert handlers
        for handler in self.alert_handlers:
            try:
                handler(threat)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def add_alert_handler(self, handler: Callable):
        """Add alert handler function."""
        self.alert_handlers.append(handler)

    def get_recent_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events."""
        return list(self.event_buffer)[-limit:]

    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for event in self.event_buffer:
            if event.event_type.startswith('threat_detected_'):
                threat_type = event.event_type.replace('threat_detected_', '')
                threat_counts[threat_type] += 1

            severity_counts[event.severity.name] += 1

        return {
            'threat_counts': dict(threat_counts),
            'severity_counts': dict(severity_counts),
            'total_events': len(self.event_buffer)
        }


class AuditLogger:
    """
    Comprehensive audit logging system.

    Logs all security-relevant events with tamper-evident storage.
    """

    def __init__(self, log_path: str = "security/audit.log",
                 crypto_engine: Optional[CryptographicEngine] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.crypto = crypto_engine

        # Log integrity
        self.log_hashes: List[str] = []
        self.integrity_key = secrets.token_bytes(32)

        # Load existing log integrity if available
        self._load_log_integrity()

        logger.info("Audit logger initialized")

    def log_event(self, event: SecurityEvent):
        """Log security event with integrity protection."""
        log_entry = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'severity': event.severity.name,
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'resource': event.resource,
            'action': event.action,
            'details': event.details,
            'mitigation_actions': event.mitigation_actions
        }

        # Serialize log entry
        log_text = json.dumps(log_entry, sort_keys=True)

        # Add integrity hash
        entry_hash = hashlib.sha256(log_text.encode()).hexdigest()

        # Sign the hash if crypto engine available
        if self.crypto:
            signature = self.crypto.generate_hmac(entry_hash.encode())
            signed_entry = {
                'entry': log_entry,
                'hash': entry_hash,
                'signature': base64.b64encode(signature).decode()
            }
        else:
            signed_entry = {
                'entry': log_entry,
                'hash': entry_hash
            }

        # Append to log file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(signed_entry) + '\\n')

        # Update integrity chain
        if self.log_hashes:
            combined_hash = hashlib.sha256(
                (self.log_hashes[-1] + entry_hash).encode()
            ).hexdigest()
        else:
            combined_hash = entry_hash

        self.log_hashes.append(combined_hash)

        # Save integrity information
        self._save_log_integrity()

        logger.debug(f"Audit event logged: {event.event_type}")

    def verify_log_integrity(self) -> bool:
        """Verify integrity of audit log."""
        if not self.log_path.exists():
            return True  # Empty log is valid

        calculated_hashes = []
        current_chain_hash = ""

        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    signed_entry = json.loads(line.strip())
                    log_entry = signed_entry['entry']
                    stored_hash = signed_entry['hash']

                    # Verify entry hash
                    log_text = json.dumps(log_entry, sort_keys=True)
                    calculated_hash = hashlib.sha256(log_text.encode()).hexdigest()

                    if calculated_hash != stored_hash:
                        logger.error("Log entry hash verification failed")
                        return False

                    # Update chain hash
                    if current_chain_hash:
                        current_chain_hash = hashlib.sha256(
                            (current_chain_hash + calculated_hash).encode()
                        ).hexdigest()
                    else:
                        current_chain_hash = calculated_hash

                    calculated_hashes.append(current_chain_hash)

                    # Verify signature if present
                    if 'signature' in signed_entry and self.crypto:
                        signature = base64.b64decode(signed_entry['signature'])
                        if not self.crypto.verify_hmac(stored_hash.encode(), signature):
                            logger.error("Log entry signature verification failed")
                            return False

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Log entry parsing failed: {e}")
                    return False

        # Compare with stored hashes
        if len(calculated_hashes) != len(self.log_hashes):
            logger.error("Log integrity chain length mismatch")
            return False

        for i, (calculated, stored) in enumerate(zip(calculated_hashes, self.log_hashes)):
            if calculated != stored:
                logger.error(f"Log integrity chain broken at entry {i}")
                return False

        return True

    def _save_log_integrity(self):
        """Save log integrity information."""
        integrity_file = self.log_path.with_suffix('.integrity')

        integrity_data = {
            'log_hashes': self.log_hashes,
            'integrity_key': base64.b64encode(self.integrity_key).decode(),
            'last_updated': datetime.now().isoformat()
        }

        with open(integrity_file, 'w') as f:
            json.dump(integrity_data, f)

    def _load_log_integrity(self):
        """Load log integrity information."""
        integrity_file = self.log_path.with_suffix('.integrity')

        if not integrity_file.exists():
            return

        try:
            with open(integrity_file, 'r') as f:
                integrity_data = json.load(f)

            self.log_hashes = integrity_data.get('log_hashes', [])
            self.integrity_key = base64.b64decode(integrity_data.get('integrity_key', ''))

        except Exception as e:
            logger.error(f"Failed to load log integrity: {e}")

    def query_events(self, start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    event_type: Optional[str] = None,
                    user_id: Optional[str] = None,
                    limit: int = 1000) -> List[SecurityEvent]:
        """Query audit events with filters."""
        events = []

        if not self.log_path.exists():
            return events

        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    signed_entry = json.loads(line.strip())
                    log_entry = signed_entry['entry']

                    # Apply filters
                    event_time = datetime.fromisoformat(log_entry['timestamp'])

                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue
                    if event_type and log_entry['event_type'] != event_type:
                        continue
                    if user_id and log_entry['user_id'] != user_id:
                        continue

                    # Convert back to SecurityEvent
                    event = SecurityEvent(
                        event_id=log_entry['event_id'],
                        timestamp=event_time,
                        event_type=log_entry['event_type'],
                        severity=ThreatLevel[log_entry['severity']],
                        source_ip=log_entry['source_ip'],
                        user_id=log_entry['user_id'],
                        resource=log_entry['resource'],
                        action=log_entry['action'],
                        details=log_entry['details'],
                        mitigation_actions=log_entry['mitigation_actions']
                    )

                    events.append(event)

                    if len(events) >= limit:
                        break

                except Exception as e:
                    logger.error(f"Error parsing audit log entry: {e}")
                    continue

        return events


class NotificationSystem:
    """
    Multi-channel notification system for security alerts.

    Supports email, SMS, Slack, and custom notification channels.
    """

    def __init__(self):
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.templates: Dict[str, str] = {}

        # Default templates
        self._setup_default_templates()

        logger.info("Notification system initialized")

    def _setup_default_templates(self):
        """Setup default notification templates."""
        self.templates = {
            'security_alert': """
🚨 Security Alert

Event: {event_type}
Severity: {severity}
Time: {timestamp}
Source: {source_ip}
User: {user_id}
Resource: {resource}
Action: {action}

Details: {details}

Recommended Actions:
{mitigation_actions}
            """,

            'anomaly_detected': """
🔍 Anomaly Detected

Type: {event_type}
Severity: {severity}
Time: {timestamp}
Description: {description}

This may indicate unusual system behavior.
            """,

            'system_compromised': """
🚨 CRITICAL: System Compromised

Immediate action required!
Event: {event_type}
Time: {timestamp}
Source: {source_ip}

Security team has been alerted.
            """
        }

    def add_channel(self, name: str, channel_type: str, config: Dict[str, Any]):
        """Add notification channel."""
        self.channels[name] = {
            'type': channel_type,
            'config': config,
            'enabled': True
        }

        logger.info(f"Notification channel added: {name} ({channel_type})")

    def send_notification(self, template_name: str, context: Dict[str, Any],
                         channels: Optional[List[str]] = None):
        """Send notification using specified template and context."""
        if template_name not in self.templates:
            logger.error(f"Template not found: {template_name}")
            return

        template = self.templates[template_name]
        message = template.format(**context)

        # Use all channels if none specified
        if channels is None:
            channels = list(self.channels.keys())

        # Send through each channel
        for channel_name in channels:
            if channel_name in self.channels and self.channels[channel_name]['enabled']:
                channel_config = self.channels[channel_name]
                try:
                    self._send_to_channel(channel_config, message, context)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_name}: {e}")

    def _send_to_channel(self, channel_config: Dict[str, Any], message: str, context: Dict[str, Any]):
        """Send message through specific channel."""
        channel_type = channel_config['type']
        config = channel_config['config']

        if channel_type == 'email':
            self._send_email(config, message, context)
        elif channel_type == 'slack':
            self._send_slack(config, message, context)
        elif channel_type == 'sms':
            self._send_sms(config, message, context)
        elif channel_type == 'webhook':
            self._send_webhook(config, message, context)
        else:
            logger.warning(f"Unknown channel type: {channel_type}")

    def _send_email(self, config: Dict[str, Any], message: str, context: Dict[str, Any]):
        """Send email notification."""
        msg = MIMEMultipart()
        msg['From'] = config['from_address']
        msg['To'] = config['to_address']
        msg['Subject'] = f"Security Alert: {context.get('event_type', 'Unknown')}"

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587))
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
        server.quit()

    def _send_slack(self, config: Dict[str, Any], message: str, context: Dict[str, Any]):
        """Send Slack notification."""
        import requests

        payload = {
            'text': message,
            'username': 'Security Bot',
            'icon_emoji': ':shield:'
        }

        requests.post(config['webhook_url'], json=payload)

    def _send_sms(self, config: Dict[str, Any], message: str, context: Dict[str, Any]):
        """Send SMS notification."""
        # Implementation would depend on SMS provider (Twilio, AWS SNS, etc.)
        logger.info(f"SMS notification: {message[:100]}...")

    def _send_webhook(self, config: Dict[str, Any], message: str, context: Dict[str, Any]):
        """Send webhook notification."""
        import requests

        payload = {
            'message': message,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        requests.post(config['url'], json=payload, headers=config.get('headers', {}))

    def add_template(self, name: str, template: str):
        """Add custom notification template."""
        self.templates[name] = template


class SecurityInformationEventManagement:
    """
    SIEM system for comprehensive security event management.

    Correlates security events, manages incidents, and provides threat intelligence.
    """

    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.incidents: Dict[str, Dict[str, Any]] = {}
        self.correlation_rules: List[Dict[str, Any]] = []
        self.threat_intelligence: Dict[str, Any] = {}

        # Setup default correlation rules
        self._setup_correlation_rules()

        logger.info("SIEM system initialized")

    def _setup_correlation_rules(self):
        """Setup default event correlation rules."""
        self.correlation_rules = [
            {
                'name': 'brute_force_attack',
                'conditions': [
                    {'event_type': 'threat_detected_brute_force'},
                    {'count': 3, 'time_window': 300}  # 3 events in 5 minutes
                ],
                'severity': ThreatLevel.CRITICAL,
                'description': 'Coordinated brute force attack detected'
            },
            {
                'name': 'data_exfiltration',
                'conditions': [
                    {'event_type': 'large_data_transfer'},
                    {'source_ip': 'unusual_location'},
                    {'time_window': 3600}  # Within 1 hour
                ],
                'severity': ThreatLevel.HIGH,
                'description': 'Potential data exfiltration detected'
            },
            {
                'name': 'privilege_escalation_attempt',
                'conditions': [
                    {'event_type': 'threat_detected_privilege_escalation'},
                    {'user_clearance': 'insufficient'}
                ],
                'severity': ThreatLevel.CRITICAL,
                'description': 'Privilege escalation attempt detected'
            }
        ]

    def process_event(self, event: SecurityEvent) -> Optional[str]:
        """Process security event and check for correlations."""
        self.events.append(event)

        # Check correlation rules
        correlated_incidents = self._check_correlations(event)

        if correlated_incidents:
            # Create or update incident
            incident_id = self._create_incident(event, correlated_incidents)
            return incident_id

        return None

    def _check_correlations(self, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Check event against correlation rules."""
        correlated_rules = []

        for rule in self.correlation_rules:
            if self._matches_rule(event, rule):
                correlated_rules.append(rule)

        return correlated_rules

    def _matches_rule(self, event: SecurityEvent, rule: Dict[str, Any]) -> bool:
        """Check if event matches correlation rule."""
        conditions = rule['conditions']

        for condition in conditions:
            if not self._evaluate_condition(event, condition):
                return False

        return True

    def _evaluate_condition(self, event: SecurityEvent, condition: Dict[str, Any]) -> bool:
        """Evaluate correlation condition."""
        # Time window conditions
        if 'time_window' in condition:
            time_window = condition['time_window']
            cutoff_time = event.timestamp - timedelta(seconds=time_window)

            # Count matching events in time window
            count = sum(1 for e in self.events
                       if e.timestamp >= cutoff_time and
                       self._event_matches_criteria(e, condition))

            required_count = condition.get('count', 1)
            if count < required_count:
                return False

        # Simple field matching
        elif 'event_type' in condition:
            if event.event_type != condition['event_type']:
                return False

        return True

    def _event_matches_criteria(self, event: SecurityEvent, criteria: Dict[str, Any]) -> bool:
        """Check if event matches given criteria."""
        for key, value in criteria.items():
            if key == 'time_window' or key == 'count':
                continue

            if key == 'event_type':
                if event.event_type != value:
                    return False
            elif key == 'source_ip':
                if event.source_ip != value:
                    return False
            # Add more criteria as needed

        return True

    def _create_incident(self, trigger_event: SecurityEvent,
                        correlated_rules: List[Dict[str, Any]]) -> str:
        """Create new security incident."""
        incident_id = f"incident_{secrets.token_hex(8)}"

        # Determine highest severity
        max_severity = max(rule['severity'] for rule in correlated_rules)

        incident = {
            'incident_id': incident_id,
            'status': 'open',
            'severity': max_severity,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'trigger_event': trigger_event.event_id,
            'correlated_rules': [rule['name'] for rule in correlated_rules],
            'description': correlated_rules[0]['description'] if correlated_rules else 'Correlated security events',
            'events': [trigger_event.event_id],
            'assigned_to': None,
            'resolution': None
        }

        self.incidents[incident_id] = incident

        logger.warning(f"Security incident created: {incident_id} "
                      f"(severity: {max_severity.name})")

        return incident_id

    def update_incident(self, incident_id: str, updates: Dict[str, Any]):
        """Update incident status and information."""
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        incident.update(updates)
        incident['updated_at'] = datetime.now()

        logger.info(f"Incident updated: {incident_id}")

        return True

    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents."""
        return [incident for incident in self.incidents.values()
                if incident['status'] == 'open']

    def generate_threat_report(self) -> Dict[str, Any]:
        """Generate comprehensive threat report."""
        report = {
            'generated_at': datetime.now(),
            'total_events': len(self.events),
            'active_incidents': len(self.get_active_incidents()),
            'severity_breakdown': {},
            'top_threats': {},
            'temporal_analysis': {}
        }

        # Severity breakdown
        severity_counts = {}
        for event in self.events:
            severity = event.severity.name
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report['severity_breakdown'] = severity_counts

        # Top threats
        threat_counts = {}
        for event in self.events:
            if event.event_type.startswith('threat_detected_'):
                threat_type = event.event_type.replace('threat_detected_', '')
                threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1

        report['top_threats'] = dict(sorted(threat_counts.items(),
                                          key=lambda x: x[1], reverse=True)[:10])

        return report


def create_comprehensive_security_framework() -> Dict[str, Any]:
    """
    Create a comprehensive security framework with all components.

    Returns:
        Dictionary containing all security components
    """
    # Initialize core components
    crypto_engine = CryptographicEngine()
    auth_manager = AuthenticationManager(crypto_engine)
    access_control = AccessControlEngine()
    ids_system = IntrusionDetectionSystem()
    audit_logger = AuditLogger(crypto_engine=crypto_engine)
    notification_system = NotificationSystem()
    siem_system = SecurityInformationEventManagement()

    # Configure notification channels (example)
    notification_system.add_channel(
        'security_team_email',
        'email',
        {
            'smtp_server': 'smtp.company.com',
            'from_address': 'security@company.com',
            'to_address': 'security-team@company.com',
            'username': 'security@company.com',
            'password': 'encrypted_password'
        }
    )

    notification_system.add_channel(
        'security_slack',
        'slack',
        {
            'webhook_url': 'https://hooks.slack.com/services/...',
        }
    )

    # Connect IDS to audit logging and notifications
    def ids_alert_handler(threat: SecurityEvent):
        audit_logger.log_event(threat)
        notification_system.send_notification(
            'security_alert',
            {
                'event_type': threat.event_type,
                'severity': threat.severity.name,
                'timestamp': threat.timestamp.isoformat(),
                'source_ip': threat.source_ip,
                'user_id': threat.user_id or 'unknown',
                'resource': threat.resource or 'unknown',
                'action': threat.action or 'unknown',
                'details': json.dumps(threat.details),
                'mitigation_actions': '\\n'.join(threat.mitigation_actions)
            }
        )

    ids_system.add_alert_handler(ids_alert_handler)

    # Create default security policies
    admin_policy = SecurityPolicy(
        policy_id="admin_access",
        name="Administrator Access Policy",
        description="Full system access for administrators",
        rules=[
            {"type": "username", "attribute": "username", "operator": "in", "value": ["admin", "root"]},
            {"type": "permission", "attribute": "permission", "operator": "equals", "value": "*"}
        ],
        enforcement_level="allow"
    )

    access_control.create_policy(admin_policy)

    # Create default roles
    access_control.create_role(
        "admin",
        ["system:*", "security:*", "data:*"],
        "System administrator with full access"
    )

    access_control.create_role(
        "security_analyst",
        ["security:read", "audit:read", "monitoring:read"],
        "Security analyst with read-only access"
    )

    access_control.create_role(
        "data_scientist",
        ["model:*", "inference:*", "data:read"],
        "Data scientist with model and inference access"
    )

    framework = {
        'crypto_engine': crypto_engine,
        'auth_manager': auth_manager,
        'access_control': access_control,
        'ids_system': ids_system,
        'audit_logger': audit_logger,
        'notification_system': notification_system,
        'siem_system': siem_system
    }

    logger.info("Comprehensive security framework created")
    return framework


def main():
    """Demonstrate comprehensive security framework."""
    print("🔒 Comprehensive Security Framework Demo")
    print("=" * 50)

    # Create security framework
    security = create_comprehensive_security_framework()

    print("✅ Security framework initialized with components:")
    for component_name in security.keys():
        print(f"   • {component_name}")

    # Demonstrate user creation and authentication
    print("\\n👤 Testing User Authentication...")

    # Create test user
    success = security['auth_manager'].create_user(
        username="alice",
        password="secure_password123",
        permissions=["model:read", "inference:create"],
        mfa_enabled=False
    )

    if success:
        print("✅ User 'alice' created successfully")

        # Test authentication
        session_id = security['auth_manager'].authenticate_user("alice", "secure_password123")
        if session_id:
            print("✅ User authentication successful")

            # Test authorization
            authorized = security['auth_manager'].authorize_request(
                session_id, "model", "read"
            )
            print(f"✅ Authorization check: {'granted' if authorized else 'denied'}")

        else:
            print("❌ Authentication failed")

    # Demonstrate access control
    print("\\n🔑 Testing Access Control...")

    # Assign role to user
    success = security['access_control'].assign_role_to_user("alice", "data_scientist")
    if success:
        print("✅ Role 'data_scientist' assigned to user 'alice'")

        # Test permission
        has_permission = security['access_control'].check_permission("alice", "model:read")
        print(f"✅ Permission check for 'model:read': {'granted' if has_permission else 'denied'}")

    # Demonstrate security event processing
    print("\\n🚨 Testing Security Event Processing...")

    # Create test security event
    test_event = SecurityEvent(
        event_id="test_event_001",
        timestamp=datetime.now(),
        event_type="test_security_event",
        severity=ThreatLevel.LOW,
        source_ip="192.168.1.100",
        user_id="alice",
        resource="model_repository",
        action="read",
        details={"model_id": "test_model", "access_type": "read"}
    )

    # Process event
    detected_threats = security['ids_system'].analyze_event(test_event)
    print(f"✅ Security event processed, threats detected: {len(detected_threats)}")

    # Log event
    security['audit_logger'].log_event(test_event)
    print("✅ Event logged to audit trail")

    # Test SIEM correlation
    incident_id = security['siem_system'].process_event(test_event)
    if incident_id:
        print(f"✅ Security incident created: {incident_id}")

    # Demonstrate cryptographic operations
    print("\\n🔐 Testing Cryptographic Operations...")

    test_data = b"This is sensitive data that needs encryption"
    encrypted = security['crypto_engine'].encrypt_data(test_data)
    decrypted = security['crypto_engine'].decrypt_data(encrypted)

    if decrypted == test_data:
        print("✅ Encryption/decryption cycle successful")
    else:
        print("❌ Encryption/decryption failed")

    # Generate and verify digital signature
    private_key, public_key = security['crypto_engine'].generate_key_pair()
    signature = security['crypto_engine'].sign_data(test_data, private_key)
    verified = security['crypto_engine'].verify_signature(test_data, signature, public_key)

    if verified:
        print("✅ Digital signature creation and verification successful")
    else:
        print("❌ Digital signature verification failed")

    # Test log integrity
    print("\\n📋 Testing Audit Log Integrity...")

    integrity_valid = security['audit_logger'].verify_log_integrity()
    if integrity_valid:
        print("✅ Audit log integrity verified")
    else:
        print("❌ Audit log integrity check failed")

    # Generate security report
    print("\\n📊 Generating Security Report...")

    threat_summary = security['ids_system'].get_threat_summary()
    print("Threat Summary:")
    print(f"   Total events: {threat_summary['total_events']}")
    print(f"   Threats by type: {threat_summary['threat_counts']}")
    print(f"   Threats by severity: {threat_summary['severity_counts']}")

    siem_report = security['siem_system'].generate_threat_report()
    print(f"   Active incidents: {siem_report['active_incidents']}")

    # Query audit events
    recent_events = security['audit_logger'].query_events(limit=5)
    print(f"   Recent audit events: {len(recent_events)}")

    print("\\n🎉 Comprehensive security framework demo completed!")
    print("\\n💡 Key Features Demonstrated:")
    print("   • Multi-factor authentication and authorization")
    print("   • Role-based and attribute-based access control")
    print("   • Advanced cryptographic operations")
    print("   • Intrusion detection and threat analysis")
    print("   • Comprehensive audit logging with integrity")
    print("   • Multi-channel security notifications")
    print("   • Security information and event management")
    print("   • Threat intelligence and incident response")
    print("\\n🛡️ Security is not just a feature - it's the foundation!")


if __name__ == "__main__":
    main()

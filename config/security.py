"""
Quantum Edge AI Platform - Security Configuration

Advanced security configurations including encryption, authentication,
access control, and compliance frameworks for quantum edge AI systems.
"""

import os
import json
import hashlib
import hmac
import base64
import secrets
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
import ssl

# Third-party imports (would be installed in production)
try:
    import cryptography
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.x509 import Certificate
    import jwt
except ImportError:
    # Fallback for development without dependencies
    cryptography = jwt = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES256 = "aes256"
    AES128 = "aes128"
    CHACHA20 = "chacha20"
    RSA2048 = "rsa2048"
    RSA4096 = "rsa4096"

class KeyType(Enum):
    """Key types for encryption"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    JWT = "jwt"
    OAUTH2 = "oauth2"
    SAML = "saml"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    MFA = "mfa"

class AuthorizationModel(Enum):
    """Authorization models"""
    RBAC = "rbac"  # Role-Based Access Control
    ABAC = "abac"  # Attribute-Based Access Control
    PBAC = "pbac"  # Policy-Based Access Control

@dataclass
class EncryptionKey:
    """Encryption key configuration"""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_type: KeyType
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotation_required: bool = False
    usage_count: int = 0
    max_usage: Optional[int] = None

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    name: str
    description: str
    rules: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UserRole:
    """User role definition"""
    name: str
    description: str
    permissions: List[str]
    inherits_from: List[str] = field(default_factory=list)
    restrictions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessControlRule:
    """Access control rule"""
    resource: str
    action: str
    conditions: Dict[str, Any]
    effect: str  # "allow" or "deny"
    priority: int = 0

@dataclass
class EncryptionConfig:
    """Encryption configuration"""

    # Key management
    master_key_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES256
    key_rotation_days: int = 90
    key_backup_enabled: bool = True
    key_encryption_key: Optional[str] = None

    # Data encryption
    data_encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES256
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True

    # Quantum-safe encryption
    quantum_safe_enabled: bool = False
    lattice_based_algorithm: str = "kyber"

    # Key storage
    key_store_type: str = "file"  # file, vault, kms, hsm
    key_store_path: str = "./keys"
    key_store_password: Optional[str] = None

    # Certificate management
    certificate_authority: Optional[str] = None
    certificate_validity_days: int = 365
    auto_renewal_enabled: bool = True

    def generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        if self.master_key_algorithm == EncryptionAlgorithm.AES256:
            return secrets.token_bytes(32)
        elif self.master_key_algorithm == EncryptionAlgorithm.AES128:
            return secrets.token_bytes(16)
        else:
            raise ValueError(f"Unsupported master key algorithm: {self.master_key_algorithm}")

    def generate_data_key(self) -> bytes:
        """Generate data encryption key"""
        if self.data_encryption_algorithm == EncryptionAlgorithm.AES256:
            return secrets.token_bytes(32)
        elif self.data_encryption_algorithm == EncryptionAlgorithm.AES128:
            return secrets.token_bytes(16)
        else:
            return secrets.token_bytes(32)  # Default to AES256

@dataclass
class AuthenticationConfig:
    """Authentication configuration"""

    # Primary method
    primary_method: AuthenticationMethod = AuthenticationMethod.JWT

    # JWT configuration
    jwt_secret: str = "change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    jwt_refresh_enabled: bool = True
    jwt_refresh_expiry_days: int = 30

    # OAuth2 configuration
    oauth2_enabled: bool = False
    oauth2_providers: List[str] = field(default_factory=list)
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None

    # SAML configuration
    saml_enabled: bool = False
    saml_idp_url: Optional[str] = None
    saml_sp_entity_id: Optional[str] = None

    # API Key configuration
    api_key_enabled: bool = True
    api_key_length: int = 32
    api_key_prefix: str = "qeak_"  # Quantum Edge API Key

    # Certificate configuration
    certificate_auth_enabled: bool = False
    certificate_store_path: str = "./certs"
    client_certificate_required: bool = False

    # MFA configuration
    mfa_enabled: bool = False
    mfa_methods: List[str] = field(default_factory=lambda: ["totp", "sms"])

    # Session management
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    session_store_type: str = "memory"  # memory, redis, database

    def generate_api_key(self) -> str:
        """Generate API key"""
        key = secrets.token_urlsafe(self.api_key_length)
        return f"{self.api_key_prefix}{key}"

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key.startswith(self.api_key_prefix):
            return False
        key_part = api_key[len(self.api_key_prefix):]
        return len(key_part) == self.api_key_length

@dataclass
class AuthorizationConfig:
    """Authorization configuration"""

    # Authorization model
    model: AuthorizationModel = AuthorizationModel.RBAC

    # RBAC configuration
    roles: Dict[str, UserRole] = field(default_factory=dict)
    role_hierarchy: Dict[str, List[str]] = field(default_factory=dict)

    # ABAC configuration
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    policies: List[SecurityPolicy] = field(default_factory=list)

    # Access control rules
    access_rules: List[AccessControlRule] = field(default_factory=list)

    # Default permissions
    default_role: str = "user"
    anonymous_access: bool = False

    def add_role(self, role: UserRole):
        """Add role to configuration"""
        self.roles[role.name] = role

        # Update role hierarchy
        for parent_role in role.inherits_from:
            if parent_role not in self.role_hierarchy:
                self.role_hierarchy[parent_role] = []
            if role.name not in self.role_hierarchy[parent_role]:
                self.role_hierarchy[parent_role].append(role.name)

    def get_role_permissions(self, role_name: str) -> List[str]:
        """Get all permissions for a role including inherited ones"""
        permissions = set()
        visited = set()

        def collect_permissions(role):
            if role in visited:
                return
            visited.add(role)

            if role in self.roles:
                permissions.update(self.roles[role].permissions)

                # Collect from parent roles
                for parent in self.roles[role].inherits_from:
                    collect_permissions(parent)

        collect_permissions(role_name)
        return list(permissions)

    def check_permission(self, user_roles: List[str], resource: str, action: str,
                        context: Dict[str, Any] = None) -> bool:
        """Check if user has permission for action on resource"""
        if self.model == AuthorizationModel.RBAC:
            return self._check_rbac_permission(user_roles, resource, action)
        elif self.model == AuthorizationModel.ABAC:
            return self._check_abac_permission(user_roles, resource, action, context)
        else:
            return False

    def _check_rbac_permission(self, user_roles: List[str], resource: str, action: str) -> bool:
        """Check RBAC permission"""
        for role_name in user_roles:
            permissions = self.get_role_permissions(role_name)
            required_permission = f"{resource}:{action}"

            if required_permission in permissions or f"*:*" in permissions or f"{resource}:*" in permissions:
                return True

        return False

    def _check_abac_permission(self, user_roles: List[str], resource: str, action: str,
                             context: Dict[str, Any]) -> bool:
        """Check ABAC permission"""
        # Simplified ABAC implementation
        for policy in self.policies:
            if not policy.enabled:
                continue

            if self._evaluate_policy(policy, user_roles, resource, action, context):
                return policy.rules[0].get('effect', 'deny') == 'allow'

        return False

    def _evaluate_policy(self, policy: SecurityPolicy, user_roles: List[str],
                        resource: str, action: str, context: Dict[str, Any]) -> bool:
        """Evaluate ABAC policy"""
        # Simplified policy evaluation
        for rule in policy.rules:
            if rule.get('resource') == resource and rule.get('action') == action:
                conditions = rule.get('conditions', {})

                # Check role condition
                if 'roles' in conditions:
                    if not any(role in user_roles for role in conditions['roles']):
                        return False

                # Check context conditions
                for key, expected_value in conditions.get('context', {}).items():
                    if context.get(key) != expected_value:
                        return False

                return True

        return False

@dataclass
class AuditConfig:
    """Audit logging configuration"""

    # Audit settings
    enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(levelname)s - %(user)s - %(action)s - %(resource)s - %(result)s"

    # Storage
    storage_type: str = "file"  # file, database, elasticsearch, cloudwatch
    storage_path: str = "./logs/audit"
    retention_days: int = 365

    # Events to audit
    audit_events: List[str] = field(default_factory=lambda: [
        "authentication",
        "authorization",
        "data_access",
        "configuration_change",
        "model_training",
        "inference_request"
    ])

    # Compliance
    compliance_frameworks: List[str] = field(default_factory=lambda: [
        "gdpr", "hipaa", "soc2"
    ])

    # Real-time monitoring
    real_time_alerts: bool = True
    alert_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "failed_auth_attempts": 5,
        "suspicious_access": 3,
        "data_exfiltration": 1
    })

@dataclass
class SecurityConfig:
    """Main security configuration"""

    # Component configurations
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    authorization: AuthorizationConfig = field(default_factory=AuthorizationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    # General security settings
    security_level: str = "standard"  # minimal, standard, high, maximum
    fips_compliance: bool = False
    quantum_safe: bool = False

    # Network security
    tls_version: str = "1.3"
    cipher_suites: List[str] = field(default_factory=lambda: [
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES128-GCM-SHA256"
    ])

    # Headers and security policies
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    })

    # Rate limiting
    rate_limiting_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Intrusion detection
    ids_enabled: bool = False
    ids_rules: List[str] = field(default_factory=list)

    def initialize_default_roles(self):
        """Initialize default roles and permissions"""
        # Admin role
        admin_role = UserRole(
            name="admin",
            description="System administrator with full access",
            permissions=[
                "*:*",  # All permissions
                "system:configure",
                "security:manage",
                "audit:view"
            ]
        )

        # Data scientist role
        data_scientist_role = UserRole(
            name="data_scientist",
            description="Data scientist with ML model access",
            permissions=[
                "model:*",
                "data:read",
                "training:*",
                "inference:*",
                "experiment:*"
            ]
        )

        # API user role
        api_user_role = UserRole(
            name="api_user",
            description="API user with inference access",
            permissions=[
                "inference:predict",
                "model:list",
                "data:read"
            ]
        )

        # User role
        user_role = UserRole(
            name="user",
            description="Basic user role",
            permissions=[
                "inference:predict",
                "model:list"
            ]
        )

        # Add roles
        self.authorization.add_role(admin_role)
        self.authorization.add_role(data_scientist_role)
        self.authorization.add_role(api_user_role)
        self.authorization.add_role(user_role)

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers based on security level"""
        headers = dict(self.security_headers)

        if self.security_level == "high":
            headers.update({
                "X-Permitted-Cross-Domain-Policies": "none",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
            })
        elif self.security_level == "maximum":
            headers.update({
                "X-Permitted-Cross-Domain-Policies": "none",
                "Referrer-Policy": "no-referrer",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Resource-Policy": "same-origin"
            })

        return headers

    def validate_security_config(self) -> List[str]:
        """Validate security configuration"""
        errors = []

        # Check JWT secret
        if self.authentication.jwt_secret == "change-in-production":
            errors.append("JWT secret is using default value")

        # Check key lengths
        if self.encryption.key_encryption_key and len(self.encryption.key_encryption_key) < 32:
            errors.append("Key encryption key is too short")

        # Check role configuration
        if not self.authorization.roles:
            errors.append("No roles configured")

        # Check audit configuration
        if self.audit.enabled and not self.audit.audit_events:
            errors.append("Audit enabled but no events configured")

        return errors

class SecurityManager:
    """Security manager for runtime security operations"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize encryption keys
        self._master_key = None
        self._data_keys = {}

        # Initialize audit log
        self._audit_log = []

    def initialize_security(self):
        """Initialize security components"""
        # Generate or load master key
        self._master_key = self.config.encryption.generate_master_key()

        # Initialize default roles
        self.config.initialize_default_roles()

        self.logger.info("Security components initialized")

    def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user"""
        # Simplified authentication
        if self.config.authentication.primary_method == AuthenticationMethod.JWT:
            # JWT authentication would be implemented here
            return {"user_id": "user123", "roles": ["user"]}
        elif self.config.authentication.primary_method == AuthenticationMethod.API_KEY:
            api_key = credentials.get('api_key')
            if api_key and self.config.authentication.validate_api_key(api_key):
                return {"user_id": "api_user", "roles": ["api_user"]}

        return None

    def authorize_request(self, user_info: Dict[str, Any], resource: str,
                         action: str, context: Dict[str, Any] = None) -> bool:
        """Authorize request"""
        user_roles = user_info.get('roles', [])

        return self.config.authorization.check_permission(
            user_roles, resource, action, context
        )

    def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Encrypt data"""
        if not self._master_key:
            raise ValueError("Security not initialized")

        # Use provided key or generate data key
        if key_id and key_id in self._data_keys:
            key = self._data_keys[key_id]
        else:
            key = self.config.encryption.generate_data_key()
            if key_id:
                self._data_keys[key_id] = key

        # Simple AES encryption (in production, use proper crypto library)
        if self.config.encryption.data_encryption_algorithm == EncryptionAlgorithm.AES256:
            # This is a simplified implementation
            # In production, use cryptography library
            return self._simple_aes_encrypt(data, key)

        return data

    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data"""
        if key_id not in self._data_keys:
            raise ValueError(f"Unknown key ID: {key_id}")

        key = self._data_keys[key_id]

        # Simple AES decryption
        if self.config.encryption.data_encryption_algorithm == EncryptionAlgorithm.AES256:
            return self._simple_aes_decrypt(encrypted_data, key)

        return encrypted_data

    def _simple_aes_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple AES encryption (for demonstration)"""
        # In production, use proper AES implementation
        # This is just a placeholder
        return base64.b64encode(data)

    def _simple_aes_decrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple AES decryption (for demonstration)"""
        # In production, use proper AES implementation
        return base64.b64decode(data)

    def audit_log(self, event: str, user: str, resource: str,
                  action: str, result: str, details: Dict[str, Any] = None):
        """Log audit event"""
        if not self.config.audit.enabled:
            return

        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': event,
            'user': user,
            'resource': resource,
            'action': action,
            'result': result,
            'details': details or {}
        }

        self._audit_log.append(audit_entry)

        # In production, write to configured storage
        self.logger.info(f"AUDIT: {event} by {user} on {resource}: {result}")

    def get_audit_logs(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get audit logs"""
        logs = self._audit_log

        if filters:
            filtered_logs = []
            for log in logs:
                match = True
                for key, value in filters.items():
                    if log.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_logs.append(log)
            logs = filtered_logs

        return logs

    def rotate_keys(self):
        """Rotate encryption keys"""
        old_master_key = self._master_key
        self._master_key = self.config.encryption.generate_master_key()

        # Re-encrypt data keys with new master key
        for key_id, data_key in self._data_keys.items():
            # In production, re-encrypt data keys
            pass

        self.audit_log(
            'key_rotation',
            'system',
            'encryption_keys',
            'rotate',
            'success',
            {'old_key_hash': hashlib.sha256(old_master_key).hexdigest()}
        )

        self.logger.info("Encryption keys rotated")

    def check_security_health(self) -> Dict[str, Any]:
        """Check security system health"""
        return {
            'encryption_keys_initialized': self._master_key is not None,
            'audit_enabled': self.config.audit.enabled,
            'authentication_method': self.config.authentication.primary_method.value,
            'authorization_model': self.config.authorization.model.value,
            'audit_events_logged': len(self._audit_log),
            'security_headers_configured': len(self.config.get_security_headers()) > 0
        }

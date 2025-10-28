"""
Quantum Edge AI Platform - Configuration Validators

Advanced validation system for configuration files, environment variables,
and runtime configuration with custom validation rules and error reporting.
"""

import re
import json
import ipaddress
import os
from typing import Dict, List, Optional, Any, Union, Callable, Pattern, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
import numbers

# Third-party imports (would be installed in production)
try:
    import jsonschema
    from jsonschema import validate, ValidationError as JSONValidationError
except ImportError:
    jsonschema = None
    JSONValidationError = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Configuration validation error"""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, rule: Optional[str] = None):
        self.message = message
        self.field = field
        self.value = value
        self.rule = rule
        super().__init__(self.message)

@dataclass
class ValidationRule:
    """Validation rule definition"""
    name: str
    validator: Callable[[Any], bool]
    message: str
    severity: str = "error"  # error, warning, info

@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)

    def add_error(self, error: ValidationError):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: ValidationError):
        """Add validation warning"""
        self.warnings.append(warning)

    def add_info(self, info: ValidationError):
        """Add validation info"""
        self.info.append(info)

    def get_all_issues(self) -> List[ValidationError]:
        """Get all validation issues"""
        return self.errors + self.warnings + self.info

class ConfigValidator:
    """Configuration validator"""

    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._register_default_rules()
        self._register_default_schemas()

    def _register_default_rules(self):
        """Register default validation rules"""

        # String rules
        self.register_rule(ValidationRule(
            name="string_not_empty",
            validator=lambda x: isinstance(x, str) and len(x.strip()) > 0,
            message="String must not be empty"
        ))

        self.register_rule(ValidationRule(
            name="string_max_length",
            validator=lambda x, max_len=255: isinstance(x, str) and len(x) <= max_len,
            message="String exceeds maximum length"
        ))

        # Number rules
        self.register_rule(ValidationRule(
            name="number_range",
            validator=lambda x, min_val=None, max_val=None: (
                isinstance(x, numbers.Number) and
                (min_val is None or x >= min_val) and
                (max_val is None or x <= max_val)
            ),
            message="Number is outside allowed range"
        ))

        self.register_rule(ValidationRule(
            name="positive_number",
            validator=lambda x: isinstance(x, numbers.Number) and x > 0,
            message="Number must be positive"
        ))

        # Network rules
        self.register_rule(ValidationRule(
            name="valid_ip_address",
            validator=lambda x: self._is_valid_ip(x),
            message="Invalid IP address format"
        ))

        self.register_rule(ValidationRule(
            name="valid_port",
            validator=lambda x: isinstance(x, int) and 1 <= x <= 65535,
            message="Port must be between 1 and 65535"
        ))

        # Path rules
        self.register_rule(ValidationRule(
            name="valid_path",
            validator=lambda x: self._is_valid_path(x),
            message="Invalid file path"
        ))

        self.register_rule(ValidationRule(
            name="path_exists",
            validator=lambda x: os.path.exists(x),
            message="Path does not exist",
            severity="warning"
        ))

        # URL rules
        self.register_rule(ValidationRule(
            name="valid_url",
            validator=lambda x: self._is_valid_url(x),
            message="Invalid URL format"
        ))

        # Email rules
        self.register_rule(ValidationRule(
            name="valid_email",
            validator=lambda x: self._is_valid_email(x),
            message="Invalid email format"
        ))

        # Security rules
        self.register_rule(ValidationRule(
            name="secure_password",
            validator=lambda x: self._is_secure_password(x),
            message="Password does not meet security requirements"
        ))

        # Quantum-specific rules
        self.register_rule(ValidationRule(
            name="valid_qubits",
            validator=lambda x: isinstance(x, int) and 1 <= x <= 1000,
            message="Number of qubits must be between 1 and 1000"
        ))

        self.register_rule(ValidationRule(
            name="valid_backend",
            validator=lambda x: x in ['simulator', 'ibmq', 'aws_braket', 'azure_quantum', 'ionq'],
            message="Invalid quantum backend"
        ))

    def _register_default_schemas(self):
        """Register default JSON schemas"""

        # Server configuration schema
        self.schemas['server'] = {
            'type': 'object',
            'properties': {
                'host': {'type': 'string', 'format': 'ipv4'},
                'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535},
                'workers': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                'timeout': {'type': 'integer', 'minimum': 1},
                'max_connections': {'type': 'integer', 'minimum': 1}
            },
            'required': ['host', 'port']
        }

        # Database configuration schema
        self.schemas['database'] = {
            'type': 'object',
            'properties': {
                'type': {'type': 'string', 'enum': ['postgresql', 'mysql', 'mongodb', 'redis']},
                'host': {'type': 'string'},
                'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535},
                'name': {'type': 'string'},
                'user': {'type': 'string'},
                'password': {'type': 'string'},
                'pool_size': {'type': 'integer', 'minimum': 1},
                'max_overflow': {'type': 'integer', 'minimum': 0}
            },
            'required': ['type', 'host', 'port', 'name']
        }

        # Security configuration schema
        self.schemas['security'] = {
            'type': 'object',
            'properties': {
                'jwt_secret': {'type': 'string', 'minLength': 32},
                'jwt_expiry_hours': {'type': 'integer', 'minimum': 1},
                'encryption_algorithm': {'type': 'string'},
                'key_rotation_days': {'type': 'integer', 'minimum': 1},
                'rate_limit_requests': {'type': 'integer', 'minimum': 1},
                'rate_limit_window': {'type': 'integer', 'minimum': 1}
            }
        }

        # Quantum configuration schema
        self.schemas['quantum'] = {
            'type': 'object',
            'properties': {
                'backend': {'type': 'string', 'enum': ['simulator', 'ibmq', 'aws_braket', 'azure_quantum', 'ionq']},
                'max_qubits': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                'optimization_level': {'type': 'integer', 'minimum': 0, 'maximum': 3},
                'shots': {'type': 'integer', 'minimum': 1},
                'timeout': {'type': 'integer', 'minimum': 1}
            },
            'required': ['backend']
        }

    def register_rule(self, rule: ValidationRule):
        """Register validation rule"""
        self.rules[rule.name] = rule

    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register JSON schema"""
        self.schemas[name] = schema

    def validate_config(self, config: Dict[str, Any],
                       schema_name: Optional[str] = None) -> ValidationResult:
        """Validate configuration against schema and rules"""
        result = ValidationResult(is_valid=True)

        # JSON Schema validation
        if schema_name and schema_name in self.schemas and jsonschema:
            try:
                validate(config, self.schemas[schema_name])
            except JSONValidationError as e:
                result.add_error(ValidationError(
                    message=f"Schema validation failed: {e.message}",
                    field=e.absolute_path[0] if e.absolute_path else None,
                    value=e.instance,
                    rule="json_schema"
                ))

        # Custom rule validation
        self._validate_with_rules(config, result, "")

        return result

    def _validate_with_rules(self, config: Dict[str, Any], result: ValidationResult, path: str):
        """Recursively validate configuration with custom rules"""
        if not isinstance(config, dict):
            return

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            # Check if there are validation rules for this field
            field_rules = self._get_field_rules(key, value, config)

            for rule in field_rules:
                try:
                    if not rule.validator(value):
                        error = ValidationError(
                            message=rule.message,
                            field=current_path,
                            value=value,
                            rule=rule.name
                        )

                        if rule.severity == "error":
                            result.add_error(error)
                        elif rule.severity == "warning":
                            result.add_warning(error)
                        else:
                            result.add_info(error)

                except Exception as e:
                    result.add_error(ValidationError(
                        message=f"Validation rule error: {str(e)}",
                        field=current_path,
                        rule=rule.name
                    ))

            # Recursively validate nested objects
            if isinstance(value, dict):
                self._validate_with_rules(value, result, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_with_rules(item, result, f"{current_path}[{i}]")

    def _get_field_rules(self, field: str, value: Any, config: Dict[str, Any]) -> List[ValidationRule]:
        """Get validation rules for a field"""
        rules = []

        # Field-specific rules
        if field == 'host':
            rules.append(self.rules['valid_ip_address'])
        elif field == 'port':
            rules.append(self.rules['valid_port'])
        elif field in ['max_qubits', 'workers', 'pool_size']:
            rules.append(self.rules['positive_number'])
        elif field == 'jwt_secret':
            rules.append(ValidationRule(
                name="jwt_secret_length",
                validator=lambda x: len(str(x)) >= 32,
                message="JWT secret must be at least 32 characters"
            ))
        elif field == 'backend':
            rules.append(self.rules['valid_backend'])
        elif field == 'log_level':
            rules.append(ValidationRule(
                name="valid_log_level",
                validator=lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                message="Invalid log level"
            ))
        elif field.endswith('_url'):
            rules.append(self.rules['valid_url'])
        elif field.endswith('_email'):
            rules.append(self.rules['valid_email'])
        elif 'password' in field.lower():
            rules.append(self.rules['secure_password'])
        elif field.endswith('_path') or field.endswith('_dir'):
            rules.append(self.rules['valid_path'])

        return rules

    def validate_environment_variables(self, env_vars: Dict[str, str]) -> ValidationResult:
        """Validate environment variables"""
        result = ValidationResult(is_valid=True)

        for key, value in env_vars.items():
            # Check for sensitive data in non-secret variables
            if 'secret' not in key.lower() and 'password' not in key.lower():
                if self._contains_sensitive_data(value):
                    result.add_warning(ValidationError(
                        message="Potential sensitive data in environment variable",
                        field=key,
                        rule="sensitive_data_check"
                    ))

            # Validate specific environment variables
            if key.endswith('_PORT'):
                try:
                    port = int(value)
                    if not (1 <= port <= 65535):
                        result.add_error(ValidationError(
                            message="Invalid port number",
                            field=key,
                            value=value,
                            rule="valid_port"
                        ))
                except ValueError:
                    result.add_error(ValidationError(
                        message="Port must be a number",
                        field=key,
                        value=value,
                        rule="number_format"
                    ))

            elif key.endswith('_URL'):
                if not self._is_valid_url(value):
                    result.add_error(ValidationError(
                        message="Invalid URL format",
                        field=key,
                        value=value,
                        rule="valid_url"
                    ))

        return result

    def validate_file_permissions(self, config_dir: Path) -> ValidationResult:
        """Validate configuration file permissions"""
        result = ValidationResult(is_valid=True)

        if not config_dir.exists():
            result.add_error(ValidationError(
                message="Configuration directory does not exist",
                field=str(config_dir),
                rule="directory_exists"
            ))
            return result

        # Check config directory permissions
        if oct(config_dir.stat().st_mode)[-3:] not in ['755', '775']:
            result.add_warning(ValidationError(
                message="Configuration directory has incorrect permissions",
                field=str(config_dir),
                rule="secure_permissions"
            ))

        # Check individual config files
        config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        for config_file in config_files:
            file_perms = oct(config_file.stat().st_mode)[-3:]

            # Config files should not be world-readable if they contain secrets
            if file_perms[2] in ['4', '5', '6', '7']:  # World readable
                # Check if file contains sensitive data
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        if any(keyword in content.lower() for keyword in ['password', 'secret', 'key', 'token']):
                            result.add_warning(ValidationError(
                                message="Config file with sensitive data is world-readable",
                                field=str(config_file),
                                rule="secure_file_permissions"
                            ))
                except Exception:
                    pass

        return result

    # Utility methods
    def _is_valid_ip(self, value: str) -> bool:
        """Check if string is valid IP address"""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    def _is_valid_path(self, value: str) -> bool:
        """Check if string is valid file path"""
        if not isinstance(value, str):
            return False

        # Check for dangerous path traversal
        if '..' in value or value.startswith('/'):
            # More thorough check
            normalized = os.path.normpath(value)
            if '..' in normalized or normalized.startswith('/'):
                return False

        return True

    def _is_valid_url(self, value: str) -> bool:
        """Check if string is valid URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return url_pattern.match(value) is not None

    def _is_valid_email(self, value: str) -> bool:
        """Check if string is valid email"""
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        return email_pattern.match(value) is not None

    def _is_secure_password(self, value: str) -> bool:
        """Check if password meets security requirements"""
        if not isinstance(value, str):
            return False

        # At least 8 characters, contains uppercase, lowercase, digit, and special char
        if len(value) < 8:
            return False

        has_upper = any(c.isupper() for c in value)
        has_lower = any(c.islower() for c in value)
        has_digit = any(c.isdigit() for c in value)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in value)

        return has_upper and has_lower and has_digit and has_special

    def _contains_sensitive_data(self, value: str) -> bool:
        """Check if string contains potentially sensitive data"""
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',  # Phone
            r'\b(?:sk|pk|token|key|secret|password|auth)[\-_]?\w*\s*[:=]\s*\S+',  # API keys/tokens
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

class SecurityValidator:
    """Security-focused configuration validator"""

    def __init__(self):
        self.risk_patterns = self._load_risk_patterns()

    def _load_risk_patterns(self) -> Dict[str, Pattern]:
        """Load security risk patterns"""
        return {
            'weak_password': re.compile(r'^.{0,7}$|^\d+$|^[a-zA-Z]+$|^[^a-zA-Z0-9]+$'),
            'exposed_secret': re.compile(r'(?i)(secret|password|key|token)\s*[:=]\s*\S+'),
            'insecure_protocol': re.compile(r'http://(?!localhost|127\.0\.0\.1)'),
            'weak_cipher': re.compile(r'(?i)(des|rc4|md5)'),
            'debug_enabled': re.compile(r'(?i)debug.*true|debug.*enabled'),
        }

    def validate_security_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate security-related configuration"""
        result = ValidationResult(is_valid=True)

        # Check for security misconfigurations
        self._check_security_misconfigurations(config, result, "")

        # Check for exposed secrets
        self._check_exposed_secrets(config, result)

        # Check for insecure defaults
        self._check_insecure_defaults(config, result)

        return result

    def _check_security_misconfigurations(self, config: Dict[str, Any],
                                        result: ValidationResult, path: str):
        """Check for security misconfigurations"""
        if not isinstance(config, dict):
            return

        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            # Check specific security fields
            if key.lower() in ['password', 'secret', 'key', 'token']:
                if isinstance(value, str) and len(value) < 16:
                    result.add_warning(ValidationError(
                        message="Potentially weak secret/key",
                        field=current_path,
                        rule="weak_secret"
                    ))

            elif key == 'ssl_enabled' and not value:
                result.add_warning(ValidationError(
                    message="SSL/TLS is disabled",
                    field=current_path,
                    rule="ssl_disabled"
                ))

            elif key == 'debug' and value:
                result.add_warning(ValidationError(
                    message="Debug mode is enabled in production",
                    field=current_path,
                    rule="debug_enabled"
                ))

            # Recursively check nested objects
            if isinstance(value, dict):
                self._check_security_misconfigurations(value, result, current_path)

    def _check_exposed_secrets(self, config: Dict[str, Any], result: ValidationResult):
        """Check for exposed secrets in configuration"""
        config_str = json.dumps(config, default=str)

        if self.risk_patterns['exposed_secret'].search(config_str):
            result.add_error(ValidationError(
                message="Configuration contains exposed secrets",
                rule="exposed_secrets"
            ))

    def _check_insecure_defaults(self, config: Dict[str, Any], result: ValidationResult):
        """Check for insecure default values"""
        # Check for default passwords
        if config.get('database', {}).get('password') == 'password':
            result.add_error(ValidationError(
                message="Database using default password",
                field="database.password",
                rule="default_password"
            ))

        # Check for default JWT secret
        if config.get('security', {}).get('jwt_secret') == 'change-in-production':
            result.add_error(ValidationError(
                message="JWT secret is default value",
                field="security.jwt_secret",
                rule="default_secret"
            ))

class PerformanceValidator:
    """Performance-focused configuration validator"""

    def validate_performance_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate performance-related configuration"""
        result = ValidationResult(is_valid=True)

        # Check memory limits
        max_memory = config.get('edge_ai', {}).get('memory_limit_mb', 512)
        if max_memory > 4096:  # 4GB
            result.add_warning(ValidationError(
                message="Very high memory limit may cause issues",
                field="edge_ai.memory_limit_mb",
                value=max_memory,
                rule="high_memory_limit"
            ))

        # Check thread pool sizes
        workers = config.get('server', {}).get('workers', 4)
        if workers > 50:
            result.add_warning(ValidationError(
                message="Very high worker count may cause contention",
                field="server.workers",
                value=workers,
                rule="high_worker_count"
            ))

        # Check cache sizes
        cache_size = config.get('edge_ai', {}).get('model_cache_size', 100)
        if cache_size > 1000:
            result.add_warning(ValidationError(
                message="Large cache size may consume excessive memory",
                field="edge_ai.model_cache_size",
                value=cache_size,
                rule="large_cache_size"
            ))

        return result

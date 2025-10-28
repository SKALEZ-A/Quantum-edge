"""
Quantum Edge AI Platform - Configuration Models

Data models and schemas for configuration management,
validation, and serialization.
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
import re

# Third-party imports (would be installed in production)
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic.dataclasses import dataclass as pydantic_dataclass
except ImportError:
    # Fallback for development without dependencies
    BaseModel = Field = validator = ValidationError = None

    def pydantic_dataclass(cls):
        return cls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigSection(Enum):
    """Configuration sections"""
    SERVER = "server"
    DATABASE = "database"
    QUANTUM = "quantum"
    EDGE_AI = "edge_ai"
    FEDERATED = "federated"
    SECURITY = "security"
    MONITORING = "monitoring"
    DEPLOYMENT = "deployment"
    LOGGING = "logging"

@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    section: ConfigSection
    version: str = "1.0.0"
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    field_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    field_descriptions: Dict[str, str] = field(default_factory=dict)

    def validate_section(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration section"""
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in self.field_types.items():
            if field in config_data:
                value = config_data[field]
                if not self._check_type(value, expected_type):
                    errors.append(f"Field {field} has incorrect type. Expected {expected_type}")

        # Check constraints
        for field, constraints in self.field_constraints.items():
            if field in config_data:
                value = config_data[field]
                field_errors = self._check_constraints(value, constraints, field)
                errors.extend(field_errors)

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check value type"""
        type_checks = {
            'string': lambda x: isinstance(x, str),
            'integer': lambda x: isinstance(x, int),
            'number': lambda x: isinstance(x, (int, float)),
            'boolean': lambda x: isinstance(x, bool),
            'array': lambda x: isinstance(x, list),
            'object': lambda x: isinstance(x, dict)
        }

        if expected_type in type_checks:
            return type_checks[expected_type](value)

        return True

    def _check_constraints(self, value: Any, constraints: Dict[str, Any], field: str) -> List[str]:
        """Check value constraints"""
        errors = []

        if 'min' in constraints and isinstance(value, (int, float)):
            if value < constraints['min']:
                errors.append(f"Field {field} must be >= {constraints['min']}")

        if 'max' in constraints and isinstance(value, (int, float)):
            if value > constraints['max']:
                errors.append(f"Field {field} must be <= {constraints['max']}")

        if 'pattern' in constraints and isinstance(value, str):
            if not re.match(constraints['pattern'], value):
                errors.append(f"Field {field} does not match required pattern")

        if 'enum' in constraints:
            if value not in constraints['enum']:
                errors.append(f"Field {field} must be one of: {constraints['enum']}")

        if 'min_length' in constraints and isinstance(value, (str, list)):
            if len(value) < constraints['min_length']:
                errors.append(f"Field {field} must have length >= {constraints['min_length']}")

        if 'max_length' in constraints and isinstance(value, (str, list)):
            if len(value) > constraints['max_length']:
                errors.append(f"Field {field} must have length <= {constraints['max_length']}")

        return errors

# Default configuration schemas
DEFAULT_SCHEMAS = {
    ConfigSection.SERVER: ConfigSchema(
        section=ConfigSection.SERVER,
        required_fields=['host', 'port'],
        optional_fields=['workers', 'timeout', 'max_connections'],
        field_types={
            'host': 'string',
            'port': 'integer',
            'workers': 'integer',
            'timeout': 'integer',
            'max_connections': 'integer'
        },
        field_constraints={
            'port': {'min': 1, 'max': 65535},
            'workers': {'min': 1, 'max': 100},
            'timeout': {'min': 1},
            'max_connections': {'min': 1}
        },
        field_descriptions={
            'host': 'Server bind address',
            'port': 'Server port number',
            'workers': 'Number of worker processes',
            'timeout': 'Request timeout in seconds',
            'max_connections': 'Maximum concurrent connections'
        }
    ),

    ConfigSection.DATABASE: ConfigSchema(
        section=ConfigSection.DATABASE,
        required_fields=['type', 'host', 'port', 'name'],
        optional_fields=['user', 'password', 'pool_size', 'max_overflow'],
        field_types={
            'type': 'string',
            'host': 'string',
            'port': 'integer',
            'name': 'string',
            'user': 'string',
            'password': 'string',
            'pool_size': 'integer',
            'max_overflow': 'integer'
        },
        field_constraints={
            'type': {'enum': ['postgresql', 'mysql', 'mongodb', 'redis']},
            'port': {'min': 1, 'max': 65535},
            'pool_size': {'min': 1},
            'max_overflow': {'min': 0}
        }
    ),

    ConfigSection.QUANTUM: ConfigSchema(
        section=ConfigSection.QUANTUM,
        required_fields=['backend'],
        optional_fields=['max_qubits', 'optimization_level', 'shots', 'timeout'],
        field_types={
            'backend': 'string',
            'max_qubits': 'integer',
            'optimization_level': 'integer',
            'shots': 'integer',
            'timeout': 'integer'
        },
        field_constraints={
            'backend': {'enum': ['simulator', 'ibmq', 'aws_braket', 'azure_quantum', 'ionq']},
            'max_qubits': {'min': 1, 'max': 1000},
            'optimization_level': {'min': 0, 'max': 3},
            'shots': {'min': 1},
            'timeout': {'min': 1}
        }
    ),

    ConfigSection.SECURITY: ConfigSchema(
        section=ConfigSection.SECURITY,
        required_fields=['jwt_secret'],
        optional_fields=['jwt_expiry_hours', 'encryption_algorithm', 'rate_limit_requests', 'rate_limit_window'],
        field_types={
            'jwt_secret': 'string',
            'jwt_expiry_hours': 'integer',
            'encryption_algorithm': 'string',
            'rate_limit_requests': 'integer',
            'rate_limit_window': 'integer'
        },
        field_constraints={
            'jwt_secret': {'min_length': 32},
            'jwt_expiry_hours': {'min': 1},
            'rate_limit_requests': {'min': 1},
            'rate_limit_window': {'min': 1}
        }
    ),

    ConfigSection.MONITORING: ConfigSchema(
        section=ConfigSection.MONITORING,
        optional_fields=['enabled', 'metrics_interval', 'log_level', 'alert_thresholds'],
        field_types={
            'enabled': 'boolean',
            'metrics_interval': 'integer',
            'log_level': 'string',
            'alert_thresholds': 'object'
        },
        field_constraints={
            'metrics_interval': {'min': 1},
            'log_level': {'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']}
        }
    )
}

@dataclass
class ConfigTemplate:
    """Configuration template"""
    name: str
    description: str
    environment: str
    config_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigTemplate':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

class ConfigValidator:
    """Enhanced configuration validator"""

    def __init__(self):
        self.schemas = DEFAULT_SCHEMAS.copy()
        self.templates: Dict[str, ConfigTemplate] = {}

    def add_schema(self, schema: ConfigSchema):
        """Add configuration schema"""
        self.schemas[schema.section] = schema

    def validate_full_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate full configuration"""
        errors = {}

        for section_name, section_data in config.items():
            try:
                section = ConfigSection(section_name.upper())
                if section in self.schemas:
                    schema = self.schemas[section]
                    section_errors = schema.validate_section(section_data)
                    if section_errors:
                        errors[section_name] = section_errors
            except ValueError:
                errors[section_name] = [f"Unknown configuration section: {section_name}"]

        return errors

    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get configuration template"""
        return self.templates.get(name)

    def add_template(self, template: ConfigTemplate):
        """Add configuration template"""
        self.templates[template.name] = template

    def create_from_template(self, template_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create configuration from template"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        config = template.config_data.copy()

        if overrides:
            # Deep merge overrides
            self._deep_merge(config, overrides)

        return config

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

class ConfigSerializer:
    """Configuration serialization utilities"""

    @staticmethod
    def to_json(config: Dict[str, Any], pretty: bool = True) -> str:
        """Serialize configuration to JSON"""
        return json.dumps(config, indent=2 if pretty else None, default=str)

    @staticmethod
    def from_json(json_str: str) -> Dict[str, Any]:
        """Deserialize configuration from JSON"""
        return json.loads(json_str)

    @staticmethod
    def to_env_file(config: Dict[str, Any], prefix: str = "QUANTUM_EDGE_") -> str:
        """Serialize configuration to environment file format"""
        lines = []

        def flatten_config(config_dict: Dict[str, Any], path: str = ""):
            for key, value in config_dict.items():
                current_path = f"{path}_{key}" if path else key
                env_key = f"{prefix}{current_path.upper()}"

                if isinstance(value, dict):
                    flatten_config(value, current_path)
                elif isinstance(value, list):
                    lines.append(f'{env_key}={json.dumps(value)}')
                elif isinstance(value, bool):
                    lines.append(f'{env_key}={str(value).lower()}')
                else:
                    lines.append(f'{env_key}={value}')

        flatten_config(config)
        return "\n".join(lines)

    @staticmethod
    def mask_sensitive_data(config: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in configuration"""
        masked = config.copy()
        sensitive_keys = ['password', 'secret', 'key', 'token']

        def mask_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    if isinstance(value, str) and len(value) > 4:
                        data[key] = f"{value[:2]}***{value[-2:]}"
                elif isinstance(value, dict):
                    data[key] = mask_dict(value)
            return data

        return mask_dict(masked)

# Pydantic models for validation (if available)
if BaseModel:
    class ServerConfigModel(BaseModel):
        """Pydantic model for server configuration"""
        host: str = Field(..., description="Server bind address")
        port: int = Field(..., ge=1, le=65535, description="Server port")
        workers: int = Field(default=4, ge=1, le=100, description="Number of workers")
        timeout: int = Field(default=30, ge=1, description="Request timeout")
        max_connections: int = Field(default=1000, ge=1, description="Max connections")

    class DatabaseConfigModel(BaseModel):
        """Pydantic model for database configuration"""
        type: str = Field(..., regex=r'^(postgresql|mysql|mongodb|redis)$')
        host: str = Field(..., description="Database host")
        port: int = Field(..., ge=1, le=65535, description="Database port")
        name: str = Field(..., description="Database name")
        user: Optional[str] = Field(default=None, description="Database user")
        password: Optional[str] = Field(default=None, description="Database password")
        pool_size: int = Field(default=10, ge=1, description="Connection pool size")

    class SecurityConfigModel(BaseModel):
        """Pydantic model for security configuration"""
        jwt_secret: str = Field(..., min_length=32, description="JWT secret key")
        jwt_expiry_hours: int = Field(default=24, ge=1, description="JWT expiry hours")
        encryption_algorithm: str = Field(default="AES256", description="Encryption algorithm")
        rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests")
        rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window")

    class QuantumConfigModel(BaseModel):
        """Pydantic model for quantum configuration"""
        backend: str = Field(..., regex=r'^(simulator|ibmq|aws_braket|azure_quantum|ionq)$')
        max_qubits: int = Field(default=32, ge=1, le=1000, description="Maximum qubits")
        optimization_level: int = Field(default=1, ge=0, le=3, description="Optimization level")
        shots: int = Field(default=1000, ge=1, description="Number of shots")
        timeout: int = Field(default=300, ge=1, description="Timeout in seconds")

# Export all classes
__all__ = [
    'ConfigSection', 'ConfigSchema', 'ConfigTemplate',
    'ConfigValidator', 'ConfigSerializer'
]

# Add Pydantic models if available
if BaseModel:
    __all__.extend([
        'ServerConfigModel', 'DatabaseConfigModel',
        'SecurityConfigModel', 'QuantumConfigModel'
    ])

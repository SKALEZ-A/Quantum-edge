"""
Quantum Edge AI Platform - Environment Configuration

Environment-specific configuration management with variable parsing,
secrets handling, and environment detection.
"""

import os
import json
import base64
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
import re
import platform
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    CI_CD = "ci_cd"

class SecretType(Enum):
    """Secret storage types"""
    PLAINTEXT = "plaintext"
    BASE64 = "base64"
    VAULT = "vault"
    AWS_SECRETS = "aws_secrets"
    GCP_SECRETS = "gcp_secrets"
    AZURE_KEYVAULT = "azure_keyvault"

@dataclass
class EnvironmentVariable:
    """Environment variable definition"""
    name: str
    value: Any
    type: type
    required: bool = False
    default: Any = None
    description: str = ""
    secret: bool = False
    secret_type: SecretType = SecretType.PLAINTEXT
    validation_pattern: Optional[str] = None
    validation_func: Optional[Callable[[Any], bool]] = None

@dataclass
class EnvironmentConfig:
    """Environment configuration container"""

    # Basic environment info
    type: EnvironmentType = EnvironmentType.DEVELOPMENT
    name: str = ""
    version: str = "1.0.0"
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)

    # Runtime environment
    python_version: str = field(default_factory=platform.python_version)
    working_directory: str = field(default_factory=lambda: os.getcwd())
    process_id: int = field(default_factory=os.getpid)
    user: str = field(default_factory=lambda: os.getenv('USER', 'unknown'))

    # Configuration sources
    config_files: List[str] = field(default_factory=list)
    env_files: List[str] = field(default_factory=list)

    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_cores: Optional[int] = None
    timeout_seconds: int = 300

    # Network configuration
    bind_address: str = "0.0.0.0"
    port: int = 8080
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # Database configuration
    database_url: Optional[str] = None
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # External services
    redis_url: Optional[str] = None
    rabbitmq_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None

    # Monitoring
    monitoring_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_check_endpoint: str = "/health"

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None

    # Security
    jwt_secret: Optional[str] = None
    encryption_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Quantum computing
    quantum_backend: str = "simulator"
    ibmq_token: Optional[str] = None
    aws_braket_region: Optional[str] = None

    # Edge AI
    edge_devices: List[str] = field(default_factory=list)
    model_cache_dir: str = "./models"
    inference_batch_size: int = 32

    # Federated learning
    federated_enabled: bool = True
    coordinator_address: Optional[str] = None

    # Cloud configuration
    cloud_provider: Optional[str] = None
    cloud_region: Optional[str] = None
    cloud_project: Optional[str] = None

    # Container configuration
    container_runtime: str = "docker"
    container_registry: Optional[str] = None
    container_image: str = "quantum-edge-ai:latest"

    # Deployment
    deployment_type: str = "standalone"
    replica_count: int = 1
    auto_scaling_enabled: bool = False

class EnvironmentVariables:
    """Environment variables manager"""

    def __init__(self, prefix: str = "QUANTUM_EDGE_"):
        self.prefix = prefix
        self.variables: Dict[str, EnvironmentVariable] = {}
        self._secrets_cache: Dict[str, Any] = {}
        self._cache_timeout = 300  # 5 minutes

        # Register default environment variables
        self._register_defaults()

    def _register_defaults(self):
        """Register default environment variables"""

        # Server configuration
        self.register_variable(EnvironmentVariable(
            name="HOST",
            value="0.0.0.0",
            type=str,
            description="Server bind address"
        ))

        self.register_variable(EnvironmentVariable(
            name="PORT",
            value=8080,
            type=int,
            description="Server port"
        ))

        self.register_variable(EnvironmentVariable(
            name="WORKERS",
            value=4,
            type=int,
            description="Number of worker processes"
        ))

        # Database configuration
        self.register_variable(EnvironmentVariable(
            name="DATABASE_URL",
            value=None,
            type=str,
            required=False,
            description="Database connection URL"
        ))

        self.register_variable(EnvironmentVariable(
            name="DATABASE_POOL_SIZE",
            value=10,
            type=int,
            description="Database connection pool size"
        ))

        # Security configuration
        self.register_variable(EnvironmentVariable(
            name="JWT_SECRET",
            value=None,
            type=str,
            required=False,
            secret=True,
            description="JWT signing secret"
        ))

        self.register_variable(EnvironmentVariable(
            name="ENCRYPTION_KEY",
            value=None,
            type=str,
            required=False,
            secret=True,
            secret_type=SecretType.BASE64,
            description="Data encryption key"
        ))

        # Quantum computing
        self.register_variable(EnvironmentVariable(
            name="QUANTUM_BACKEND",
            value="simulator",
            type=str,
            description="Quantum computing backend"
        ))

        self.register_variable(EnvironmentVariable(
            name="IBMQ_TOKEN",
            value=None,
            type=str,
            required=False,
            secret=True,
            description="IBM Quantum token"
        ))

        # Cloud configuration
        self.register_variable(EnvironmentVariable(
            name="CLOUD_PROVIDER",
            value=None,
            type=str,
            required=False,
            description="Cloud provider (aws, gcp, azure)"
        ))

        self.register_variable(EnvironmentVariable(
            name="CLOUD_REGION",
            value=None,
            type=str,
            required=False,
            description="Cloud region"
        ))

        # Logging
        self.register_variable(EnvironmentVariable(
            name="LOG_LEVEL",
            value="INFO",
            type=str,
            validation_pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$',
            description="Logging level"
        ))

        # Feature flags
        self.register_variable(EnvironmentVariable(
            name="FEDERATED_LEARNING_ENABLED",
            value=True,
            type=bool,
            description="Enable federated learning"
        ))

        self.register_variable(EnvironmentVariable(
            name="MONITORING_ENABLED",
            value=True,
            type=bool,
            description="Enable monitoring"
        ))

    def register_variable(self, variable: EnvironmentVariable):
        """Register environment variable"""
        key = f"{self.prefix}{variable.name}"
        self.variables[key] = variable

    def get(self, name: str, default: Any = None) -> Any:
        """Get environment variable value"""
        env_key = f"{self.prefix}{name}"
        env_value = os.getenv(env_key)

        if env_value is None:
            # Check if variable is registered
            if env_key in self.variables:
                var_def = self.variables[env_key]
                if var_def.default is not None:
                    return var_def.default
                elif var_def.required:
                    raise ValueError(f"Required environment variable {env_key} not set")
            return default

        # Parse and validate value
        return self._parse_and_validate(env_key, env_value)

    def get_all(self) -> Dict[str, Any]:
        """Get all environment variables"""
        result = {}
        for env_key, var_def in self.variables.items():
            try:
                value = self.get(var_def.name)
                if value is not None:
                    result[var_def.name] = value
            except ValueError:
                # Skip missing required variables
                continue
        return result

    def _parse_and_validate(self, env_key: str, env_value: str) -> Any:
        """Parse and validate environment variable value"""
        if env_key not in self.variables:
            # Unknown variable, return as string
            return env_value

        var_def = self.variables[env_key]

        # Handle secrets
        if var_def.secret:
            return self._handle_secret(var_def, env_value)

        # Parse value based on type
        try:
            if var_def.type == bool:
                parsed_value = env_value.lower() in ('true', 'yes', '1', 'on')
            elif var_def.type == int:
                parsed_value = int(env_value)
            elif var_def.type == float:
                parsed_value = float(env_value)
            elif var_def.type == list:
                parsed_value = json.loads(env_value)
            elif var_def.type == dict:
                parsed_value = json.loads(env_value)
            else:
                parsed_value = str(env_value)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid value for {env_key}: {env_value} ({str(e)})")

        # Custom validation
        if var_def.validation_pattern and not re.match(var_def.validation_pattern, str(parsed_value)):
            raise ValueError(f"Value for {env_key} does not match required pattern")

        if var_def.validation_func and not var_def.validation_func(parsed_value):
            raise ValueError(f"Value for {env_key} failed custom validation")

        return parsed_value

    def _handle_secret(self, var_def: EnvironmentVariable, env_value: str) -> Any:
        """Handle secret values"""
        cache_key = f"{var_def.name}:{hashlib.md5(env_value.encode()).hexdigest()}"

        # Check cache first
        if cache_key in self._secrets_cache:
            cached_time, cached_value = self._secrets_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return cached_value

        # Process secret based on type
        if var_def.secret_type == SecretType.BASE64:
            try:
                decoded = base64.b64decode(env_value)
                value = decoded.decode('utf-8')
            except Exception as e:
                raise ValueError(f"Failed to decode base64 secret {var_def.name}: {str(e)}")

        elif var_def.secret_type == SecretType.VAULT:
            # HashiCorp Vault integration would go here
            value = self._fetch_from_vault(var_def.name, env_value)

        elif var_def.secret_type == SecretType.AWS_SECRETS:
            # AWS Secrets Manager integration would go here
            value = self._fetch_from_aws_secrets(var_def.name, env_value)

        elif var_def.secret_type == SecretType.GCP_SECRETS:
            # Google Cloud Secret Manager integration would go here
            value = self._fetch_from_gcp_secrets(var_def.name, env_value)

        elif var_def.secret_type == SecretType.AZURE_KEYVAULT:
            # Azure Key Vault integration would go here
            value = self._fetch_from_azure_keyvault(var_def.name, env_value)

        else:  # PLAINTEXT
            value = env_value

        # Cache the result
        self._secrets_cache[cache_key] = (time.time(), value)

        return value

    def _fetch_from_vault(self, name: str, path: str) -> str:
        """Fetch secret from HashiCorp Vault"""
        # Implementation would use hvac library
        logger.warning(f"Vault integration not implemented, using path as-is: {path}")
        return path

    def _fetch_from_aws_secrets(self, name: str, secret_id: str) -> str:
        """Fetch secret from AWS Secrets Manager"""
        # Implementation would use boto3
        logger.warning(f"AWS Secrets Manager integration not implemented, using secret_id as-is: {secret_id}")
        return secret_id

    def _fetch_from_gcp_secrets(self, name: str, secret_id: str) -> str:
        """Fetch secret from Google Cloud Secret Manager"""
        # Implementation would use google-cloud-secret-manager
        logger.warning(f"GCP Secret Manager integration not implemented, using secret_id as-is: {secret_id}")
        return secret_id

    def _fetch_from_azure_keyvault(self, name: str, secret_id: str) -> str:
        """Fetch secret from Azure Key Vault"""
        # Implementation would use azure-identity and azure-keyvault-secrets
        logger.warning(f"Azure Key Vault integration not implemented, using secret_id as-is: {secret_id}")
        return secret_id

    def validate_all(self) -> List[str]:
        """Validate all registered environment variables"""
        errors = []

        for env_key, var_def in self.variables.items():
            try:
                value = self.get(var_def.name)
                if var_def.required and value is None:
                    errors.append(f"Missing required variable: {env_key}")
            except ValueError as e:
                errors.append(str(e))

        return errors

    def export_to_file(self, filepath: str, include_secrets: bool = False):
        """Export environment variables to file"""
        env_data = {}

        for env_key, var_def in self.variables.items():
            try:
                value = self.get(var_def.name)
                if value is not None and (not var_def.secret or include_secrets):
                    env_data[env_key] = value
            except ValueError:
                continue

        with open(filepath, 'w') as f:
            for key, value in env_data.items():
                if isinstance(value, (list, dict)):
                    f.write(f'{key}={json.dumps(value)}\n')
                else:
                    f.write(f'{key}={value}\n')

        logger.info(f"Environment variables exported to {filepath}")

class EnvironmentDetector:
    """Environment detection utilities"""

    @staticmethod
    def detect_environment() -> EnvironmentType:
        """Detect current environment type"""
        # Check environment variables
        env_var = os.getenv('QUANTUM_EDGE_ENV', os.getenv('ENV', '')).lower()

        if env_var in ['prod', 'production']:
            return EnvironmentType.PRODUCTION
        elif env_var in ['staging', 'stage']:
            return EnvironmentType.STAGING
        elif env_var in ['test', 'testing']:
            return EnvironmentType.TESTING
        elif env_var in ['ci', 'cicd', 'ci_cd']:
            return EnvironmentType.CI_CD
        else:
            return EnvironmentType.DEVELOPMENT

    @staticmethod
    def is_containerized() -> bool:
        """Check if running in a container"""
        return (
            os.path.exists('/.dockerenv') or
            os.getenv('KUBERNETES_SERVICE_HOST') is not None or
            'container' in os.getenv('HOSTNAME', '').lower()
        )

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information"""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_gb': None,  # Would need psutil
            'hostname': socket.gethostname(),
            'containerized': EnvironmentDetector.is_containerized(),
            'environment': EnvironmentDetector.detect_environment().value
        }

    @staticmethod
    def get_resource_limits() -> Dict[str, Any]:
        """Get system resource limits"""
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            memory_limit = soft if soft != -1 else None
        except ImportError:
            memory_limit = None

        return {
            'memory_limit_bytes': memory_limit,
            'cpu_limit': os.cpu_count(),
            'file_descriptors': None,  # Would need resource module
            'max_processes': None
        }

class EnvironmentConfigBuilder:
    """Builder for environment configuration"""

    def __init__(self):
        self.config = EnvironmentConfig()
        self.env_vars = EnvironmentVariables()

    def with_environment_type(self, env_type: EnvironmentType) -> 'EnvironmentConfigBuilder':
        """Set environment type"""
        self.config.type = env_type
        return self

    def with_server_config(self, host: str = "0.0.0.0", port: int = 8080,
                           ssl_enabled: bool = False) -> 'EnvironmentConfigBuilder':
        """Configure server settings"""
        self.config.bind_address = host
        self.config.port = port
        self.config.ssl_enabled = ssl_enabled
        return self

    def with_database_config(self, url: Optional[str] = None,
                           pool_size: int = 10) -> 'EnvironmentConfigBuilder':
        """Configure database settings"""
        self.config.database_url = url or self.env_vars.get('DATABASE_URL')
        self.config.database_pool_size = pool_size
        return self

    def with_security_config(self, jwt_secret: Optional[str] = None,
                           cors_origins: List[str] = None) -> 'EnvironmentConfigBuilder':
        """Configure security settings"""
        self.config.jwt_secret = jwt_secret or self.env_vars.get('JWT_SECRET')
        if cors_origins:
            self.config.cors_origins = cors_origins
        return self

    def with_quantum_config(self, backend: str = "simulator",
                           ibmq_token: Optional[str] = None) -> 'EnvironmentConfigBuilder':
        """Configure quantum computing settings"""
        self.config.quantum_backend = backend
        self.config.ibmq_token = ibmq_token or self.env_vars.get('IBMQ_TOKEN')
        return self

    def with_monitoring_config(self, enabled: bool = True,
                              log_level: str = "INFO") -> 'EnvironmentConfigBuilder':
        """Configure monitoring settings"""
        self.config.monitoring_enabled = enabled
        self.config.log_level = log_level
        return self

    def from_environment_variables(self) -> 'EnvironmentConfigBuilder':
        """Load configuration from environment variables"""
        env_data = self.env_vars.get_all()

        # Map environment variables to config
        self.config.bind_address = env_data.get('HOST', self.config.bind_address)
        self.config.port = env_data.get('PORT', self.config.port)
        self.config.database_url = env_data.get('DATABASE_URL', self.config.database_url)
        self.config.jwt_secret = env_data.get('JWT_SECRET', self.config.jwt_secret)
        self.config.quantum_backend = env_data.get('QUANTUM_BACKEND', self.config.quantum_backend)
        self.config.monitoring_enabled = env_data.get('MONITORING_ENABLED', self.config.monitoring_enabled)
        self.config.log_level = env_data.get('LOG_LEVEL', self.config.log_level)
        self.config.federated_enabled = env_data.get('FEDERATED_LEARNING_ENABLED', self.config.federated_enabled)

        return self

    def build(self) -> EnvironmentConfig:
        """Build the environment configuration"""
        # Auto-detect environment if not set
        if not self.config.name:
            self.config.name = self.config.type.value

        # Validate configuration
        errors = self.env_vars.validate_all()
        if errors:
            logger.warning(f"Environment validation errors: {errors}")

        return self.config

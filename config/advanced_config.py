#!/usr/bin/env python3
"""
Advanced Configuration Management System for Quantum Edge AI Platform

This module provides sophisticated configuration management capabilities including:
- Dynamic configuration loading and validation
- Environment-specific overrides
- Configuration encryption and security
- Hot-reloading capabilities
- Configuration migration and versioning
- Multi-source configuration merging
"""

import os
import json
import yaml
import hashlib
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import jsonschema
from jsonschema import validate, ValidationError
import watchfiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """Represents a configuration source."""
    name: str
    type: str  # 'file', 'env', 'database', 'remote'
    path: Optional[str] = None
    url: Optional[str] = None
    priority: int = 0
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str
    rule_type: str  # 'required', 'type', 'range', 'pattern', 'custom'
    value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    validator: Optional[Callable] = None
    error_message: Optional[str] = None


class ConfigurationEncryptor:
    """Handles encryption/decryption of sensitive configuration values."""

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            # Generate a key for demonstration (in production, use a secure key)
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(b"demo_key"))
        self.cipher = Fernet(key)

    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        encrypted = self.cipher.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        encrypted = base64.urlsafe_b64decode(encrypted_value)
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

    def rotate_key(self, new_key: bytes):
        """Rotate encryption key."""
        self.cipher = Fernet(new_key)
        logger.info("Configuration encryption key rotated")


class ConfigurationValidator:
    """Advanced configuration validation system."""

    def __init__(self):
        self.schema_cache: Dict[str, Dict] = {}
        self.validation_rules: List[ConfigValidationRule] = []
        self.custom_validators: Dict[str, Callable] = {}

    def load_schema(self, schema_path: str, schema_name: str):
        """Load a JSON schema for validation."""
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        self.schema_cache[schema_name] = schema
        logger.info(f"Loaded configuration schema: {schema_name}")

    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add a custom validation rule."""
        self.validation_rules.append(rule)

    def add_custom_validator(self, name: str, validator: Callable):
        """Add a custom validator function."""
        self.custom_validators[name] = validator

    def validate_config(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> List[str]:
        """Validate configuration against schema and rules."""
        errors = []

        # Schema validation
        if schema_name and schema_name in self.schema_cache:
            try:
                validate(config, self.schema_cache[schema_name])
            except ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")

        # Custom rule validation
        for rule in self.validation_rules:
            rule_errors = self._validate_rule(config, rule)
            errors.extend(rule_errors)

        return errors

    def _validate_rule(self, config: Dict[str, Any], rule: ConfigValidationRule) -> List[str]:
        """Validate a single rule."""
        errors = []
        value = self._get_nested_value(config, rule.field_path)

        if rule.rule_type == 'required' and value is None:
            errors.append(f"Required field '{rule.field_path}' is missing")

        elif value is not None:
            if rule.rule_type == 'type':
                if not isinstance(value, rule.value):
                    errors.append(f"Field '{rule.field_path}' must be of type {rule.value.__name__}")

            elif rule.rule_type == 'range':
                if not isinstance(value, (int, float)):
                    errors.append(f"Field '{rule.field_path}' must be numeric for range validation")
                elif rule.min_value is not None and value < rule.min_value:
                    errors.append(f"Field '{rule.field_path}' must be >= {rule.min_value}")
                elif rule.max_value is not None and value > rule.max_value:
                    errors.append(f"Field '{rule.field_path}' must be <= {rule.max_value}")

            elif rule.rule_type == 'pattern':
                import re
                if not isinstance(value, str) or not re.match(rule.pattern, value):
                    errors.append(f"Field '{rule.field_path}' does not match pattern {rule.pattern}")

            elif rule.rule_type == 'custom' and rule.validator:
                try:
                    if not rule.validator(value):
                        error_msg = rule.error_message or f"Custom validation failed for '{rule.field_path}'"
                        errors.append(error_msg)
                except Exception as e:
                    errors.append(f"Custom validation error for '{rule.field_path}': {e}")

        return errors

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from configuration using dot notation."""
        keys = path.split('.')
        current = config

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                else:
                    return None
            return current
        except (KeyError, TypeError):
            return None


class ConfigurationManager:
    """
    Advanced configuration management system.

    Features:
    - Multi-source configuration loading
    - Environment-specific overrides
    - Hot-reloading capabilities
    - Configuration encryption
    - Validation and migration
    - Change tracking and auditing
    """

    def __init__(self, config_dir: str = "config", environment: str = "development"):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.encryptor = ConfigurationEncryptor()
        self.validator = ConfigurationValidator()
        self.sources: List[ConfigSource] = []

        # Configuration state
        self.current_config: Dict[str, Any] = {}
        self.config_history: List[Dict[str, Any]] = []
        self.change_callbacks: List[Callable] = []

        # Hot-reloading
        self.hot_reload_enabled = False
        self.reload_thread: Optional[threading.Thread] = None
        self.last_reload_time = datetime.now()

        # Initialize
        self._setup_default_sources()
        self._load_validation_rules()

        logger.info(f"Initialized ConfigurationManager for environment: {environment}")

    def _setup_default_sources(self):
        """Set up default configuration sources."""
        # Base configuration
        self.sources.append(ConfigSource(
            name="base_config",
            type="file",
            path=str(self.config_dir / "config.yaml"),
            priority=0
        ))

        # Environment-specific configuration
        self.sources.append(ConfigSource(
            name="env_config",
            type="file",
            path=str(self.config_dir / f"config.{self.environment}.yaml"),
            priority=1
        ))

        # Local overrides
        self.sources.append(ConfigSource(
            name="local_config",
            type="file",
            path=str(self.config_dir / "config.local.yaml"),
            priority=2
        ))

        # Environment variables
        self.sources.append(ConfigSource(
            name="environment",
            type="env",
            priority=3
        ))

    def _load_validation_rules(self):
        """Load built-in validation rules."""
        # System configuration rules
        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="system.max_workers",
            rule_type="range",
            min_value=1,
            max_value=64
        ))

        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="system.log_level",
            rule_type="pattern",
            pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
        ))

        # Edge runtime rules
        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="edge_runtime.max_memory_mb",
            rule_type="range",
            min_value=64,
            max_value=8192
        ))

        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="edge_runtime.default_precision",
            rule_type="pattern",
            pattern="^(FP32|FP16|INT8|INT4|BINARY)$"
        ))

        # Quantum engine rules
        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="quantum_engine.n_qubits",
            rule_type="range",
            min_value=1,
            max_value=100
        ))

        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="quantum_engine.optimization_level",
            rule_type="range",
            min_value=0,
            max_value=3
        ))

        # Privacy rules
        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="privacy.default_epsilon",
            rule_type="range",
            min_value=0.01,
            max_value=10.0
        ))

        # API rules
        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="api.port",
            rule_type="range",
            min_value=1024,
            max_value=65535
        ))

        # Custom validators
        def validate_database_url(url: str) -> bool:
            """Validate database URL format."""
            import re
            # Simple validation for common database URL patterns
            patterns = [
                r'^sqlite:///.*$',
                r'^postgresql://.*$',
                r'^mysql://.*$',
                r'^mongodb://.*$'
            ]
            return any(re.match(pattern, url) for pattern in patterns)

        self.validator.add_custom_validator("database_url", validate_database_url)
        self.validator.add_validation_rule(ConfigValidationRule(
            field_path="database.url",
            rule_type="custom",
            validator=self.validator.custom_validators["database_url"],
            error_message="Database URL must be a valid database connection string"
        ))

    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load configuration from all sources."""
        if not force_reload and self.current_config:
            return self.current_config

        logger.info("Loading configuration from all sources")

        # Load configuration from each source
        configs = []
        for source in sorted(self.sources, key=lambda s: s.priority):
            try:
                config = self._load_source_config(source)
                if config:
                    configs.append((source, config))
                    logger.debug(f"Loaded config from source: {source.name}")
            except Exception as e:
                logger.error(f"Failed to load config from {source.name}: {e}")

        # Merge configurations
        merged_config = self._merge_configs(configs)

        # Validate configuration
        validation_errors = self.validator.validate_config(merged_config)
        if validation_errors:
            logger.error("Configuration validation errors:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ValueError("Configuration validation failed")

        # Decrypt sensitive values
        merged_config = self._decrypt_sensitive_values(merged_config)

        # Store configuration
        self.current_config = merged_config
        self.config_history.append({
            'timestamp': datetime.now(),
            'config': merged_config.copy(),
            'source': 'load_config'
        })

        logger.info("Configuration loaded and validated successfully")
        return merged_config

    def _load_source_config(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from a specific source."""
        if source.type == "file":
            return self._load_file_config(source)
        elif source.type == "env":
            return self._load_env_config(source)
        elif source.type == "database":
            return self._load_database_config(source)
        elif source.type == "remote":
            return self._load_remote_config(source)
        else:
            logger.warning(f"Unknown config source type: {source.type}")
            return None

    def _load_file_config(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        if not source.path or not Path(source.path).exists():
            return None

        file_path = Path(source.path)
        file_ext = file_path.suffix.lower()

        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_ext == '.json':
                config = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {file_ext}")
                return None

        # Update source metadata
        source.last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        source.checksum = self._calculate_checksum(str(config))

        return config

    def _load_env_config(self, source: ConfigSource) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        prefix = "QUANTUM_EDGE_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix):].lower()
                config_key_parts = config_key.split('_')

                # Convert snake_case to nested dict
                current = config
                for part in config_key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[config_key_parts[-1]] = self._parse_env_value(value)

        return config

    def _load_database_config(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from database."""
        # Implementation would depend on specific database setup
        logger.info("Database config loading not implemented")
        return None

    def _load_remote_config(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from remote source."""
        # Implementation would depend on remote service
        logger.info("Remote config loading not implemented")
        return None

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, list]:
        """Parse environment variable value."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _merge_configs(self, configs: List[Tuple[ConfigSource, Dict[str, Any]]]) -> Dict[str, Any]:
        """Merge configurations from multiple sources."""
        merged = {}

        for source, config in configs:
            self._deep_merge(merged, config)

        return merged

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _decrypt_sensitive_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration values."""
        sensitive_fields = [
            'database.password',
            'api.jwt_secret_key',
            'security.encryption_key',
            'privacy.encryption_key'
        ]

        decrypted_config = config.copy()

        for field_path in sensitive_fields:
            value = self.validator._get_nested_value(config, field_path)
            if value and isinstance(value, str) and value.startswith('encrypted:'):
                try:
                    encrypted_value = value[10:]  # Remove 'encrypted:' prefix
                    decrypted_value = self.encryptor.decrypt_value(encrypted_value)
                    self._set_nested_value(decrypted_config, field_path, decrypted_value)
                except Exception as e:
                    logger.error(f"Failed to decrypt {field_path}: {e}")

        return decrypted_config

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested value in configuration using dot notation."""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum of configuration content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def save_config(self, config: Dict[str, Any], source_name: str = "manual"):
        """Save configuration to file."""
        config_path = self.config_dir / f"config.{source_name}.yaml"

        # Encrypt sensitive values before saving
        config_to_save = self._encrypt_sensitive_values(config)

        with open(config_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {config_path}")

        # Track change
        self.config_history.append({
            'timestamp': datetime.now(),
            'config': config.copy(),
            'source': source_name
        })

    def _encrypt_sensitive_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive values before saving."""
        sensitive_fields = [
            'database.password',
            'api.jwt_secret_key',
            'security.encryption_key',
            'privacy.encryption_key'
        ]

        encrypted_config = config.copy()

        for field_path in sensitive_fields:
            value = self.validator._get_nested_value(config, field_path)
            if value and isinstance(value, str) and not value.startswith('encrypted:'):
                try:
                    encrypted_value = self.encryptor.encrypt_value(value)
                    self._set_nested_value(encrypted_config, field_path, f"encrypted:{encrypted_value}")
                except Exception as e:
                    logger.error(f"Failed to encrypt {field_path}: {e}")

        return encrypted_config

    def update_config(self, updates: Dict[str, Any], source: str = "runtime"):
        """Update configuration at runtime."""
        # Merge updates
        self._deep_merge(self.current_config, updates)

        # Validate updated config
        validation_errors = self.validator.validate_config(self.current_config)
        if validation_errors:
            logger.error("Configuration update validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration update")

        # Notify change callbacks
        self._notify_change_callbacks(updates)

        # Track change
        self.config_history.append({
            'timestamp': datetime.now(),
            'config': self.current_config.copy(),
            'updates': updates,
            'source': source
        })

        logger.info(f"Configuration updated from source: {source}")

    def _notify_change_callbacks(self, changes: Dict[str, Any]):
        """Notify registered change callbacks."""
        for callback in self.change_callbacks:
            try:
                callback(changes)
            except Exception as e:
                logger.error(f"Configuration change callback failed: {e}")

    def add_change_callback(self, callback: Callable):
        """Add a callback to be notified of configuration changes."""
        self.change_callbacks.append(callback)

    def enable_hot_reload(self, check_interval: float = 5.0):
        """Enable hot reloading of configuration files."""
        if self.hot_reload_enabled:
            return

        self.hot_reload_enabled = True
        self.reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            args=(check_interval,),
            daemon=True
        )
        self.reload_thread.start()
        logger.info("Hot reload enabled for configuration files")

    def disable_hot_reload(self):
        """Disable hot reloading."""
        self.hot_reload_enabled = False
        if self.reload_thread:
            self.reload_thread.join(timeout=5.0)
        logger.info("Hot reload disabled")

    def _hot_reload_loop(self, check_interval: float):
        """Hot reload monitoring loop."""
        file_paths = [source.path for source in self.sources
                     if source.type == "file" and source.path]

        while self.hot_reload_enabled:
            try:
                # Check for file changes
                for source in self.sources:
                    if source.type == "file" and source.path:
                        file_path = Path(source.path)
                        if file_path.exists():
                            current_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if source.last_modified and current_mtime > source.last_modified:
                                logger.info(f"Configuration file changed: {source.path}")
                                self.load_config(force_reload=True)
                                break
            except Exception as e:
                logger.error(f"Hot reload error: {e}")

            time.sleep(check_interval)

    def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self.config_history[-limit:] if limit > 0 else self.config_history

    def export_config(self, format: str = "yaml", include_sensitive: bool = False) -> str:
        """Export current configuration."""
        config_to_export = self.current_config.copy()

        if not include_sensitive:
            # Remove or mask sensitive fields
            sensitive_fields = [
                'database.password',
                'api.jwt_secret_key',
                'security.encryption_key',
                'privacy.encryption_key'
            ]
            for field in sensitive_fields:
                value = self.validator._get_nested_value(config_to_export, field)
                if value:
                    self._set_nested_value(config_to_export, field, "***masked***")

        if format == "yaml":
            return yaml.dump(config_to_export, default_flow_style=False, sort_keys=False)
        elif format == "json":
            return json.dumps(config_to_export, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def validate_config_file(self, file_path: str) -> List[str]:
        """Validate a configuration file without loading it."""
        try:
            temp_source = ConfigSource(name="validation", type="file", path=file_path)
            config = self._load_source_config(temp_source)
            if config:
                return self.validator.validate_config(config)
            else:
                return ["Failed to load configuration file"]
        except Exception as e:
            return [f"Error validating config file: {e}"]


def create_advanced_config_manager(environment: str = "development") -> ConfigurationManager:
    """Create and configure an advanced configuration manager."""
    config_manager = ConfigurationManager(environment=environment)

    # Load configuration schemas
    schema_dir = Path(__file__).parent
    schema_files = {
        "main": schema_dir / "schemas" / "config.schema.json",
        "quantum": schema_dir / "schemas" / "quantum.schema.json",
        "privacy": schema_dir / "schemas" / "privacy.schema.json"
    }

    for schema_name, schema_path in schema_files.items():
        if schema_path.exists():
            config_manager.validator.load_schema(str(schema_path), schema_name)

    # Add additional validation rules
    config_manager.validator.add_validation_rule(ConfigValidationRule(
        field_path="monitoring.metrics_retention_days",
        rule_type="range",
        min_value=1,
        max_value=365
    ))

    config_manager.validator.add_validation_rule(ConfigValidationRule(
        field_path="federated_learning.max_training_rounds",
        rule_type="range",
        min_value=1,
        max_value=1000
    ))

    # Enable hot reload for development
    if environment == "development":
        config_manager.enable_hot_reload()

    return config_manager


def main():
    """Demonstrate advanced configuration management."""
    print("⚙️  Advanced Configuration Management Demo")
    print("=" * 50)

    # Create configuration manager
    config_manager = create_advanced_config_manager("development")

    # Load configuration
    print("\\n📥 Loading configuration...")
    try:
        config = config_manager.load_config()
        print("✅ Configuration loaded successfully")
        print(f"   Environment: {config_manager.environment}")
        print(f"   Quantum qubits: {config.get('quantum_engine', {}).get('n_qubits', 'N/A')}")
        print(f"   Privacy epsilon: {config.get('privacy', {}).get('default_epsilon', 'N/A')}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return 1

    # Demonstrate runtime configuration updates
    print("\\n🔄 Testing runtime configuration updates...")
    try:
        # Update quantum engine settings
        updates = {
            'quantum_engine': {
                'optimization_level': 2,
                'enable_error_mitigation': True
            },
            'edge_runtime': {
                'max_memory_mb': 1024
            }
        }

        config_manager.update_config(updates, source="demo")
        print("✅ Configuration updated successfully")
        print(f"   New optimization level: {config_manager.current_config['quantum_engine']['optimization_level']}")
    except Exception as e:
        print(f"❌ Configuration update failed: {e}")

    # Demonstrate configuration export
    print("\\n💾 Exporting configuration...")
    try:
        exported_yaml = config_manager.export_config("yaml", include_sensitive=False)
        exported_json = config_manager.export_config("json", include_sensitive=False)

        with open("config_export.yaml", "w") as f:
            f.write(exported_yaml)

        with open("config_export.json", "w") as f:
            f.write(exported_json)

        print("✅ Configuration exported to files:")
        print("   - config_export.yaml")
        print("   - config_export.json")
    except Exception as e:
        print(f"❌ Configuration export failed: {e}")

    # Demonstrate configuration validation
    print("\\n🔍 Testing configuration validation...")
    try:
        # Test invalid configuration
        invalid_config = {
            'system': {
                'max_workers': 1000  # Invalid: too high
            },
            'quantum_engine': {
                'n_qubits': 1000  # Invalid: too high
            }
        }

        validation_errors = config_manager.validator.validate_config(invalid_config)
        if validation_errors:
            print("✅ Validation correctly caught errors:")
            for error in validation_errors:
                print(f"   - {error}")
        else:
            print("❌ Validation should have caught errors")
    except Exception as e:
        print(f"❌ Validation test failed: {e}")

    # Show configuration history
    print("\\n📜 Configuration change history:")
    history = config_manager.get_config_history(limit=5)
    for entry in history[-3:]:  # Show last 3 entries
        timestamp = entry['timestamp'].strftime('%H:%M:%S')
        source = entry.get('source', 'unknown')
        print(f"   {timestamp} - {source}")

    print("\\n✅ Advanced configuration management demo completed!")

    # Cleanup
    config_manager.disable_hot_reload()

    return 0


if __name__ == "__main__":
    exit(main())

"""
Quantum Edge AI Platform - Configuration Manager

Centralized configuration management with dynamic loading, validation,
environment handling, and hot-reloading capabilities.
"""

import os
import json
import yaml
import hashlib
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
import copy
import re

# Third-party imports (would be installed in production)
try:
    import cerberus
    from dotenv import load_dotenv
except ImportError:
    # Fallback for development without dependencies
    cerberus = None
    load_dotenv = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Configuration-related error"""
    pass

class ConfigFormat(Enum):
    """Supported configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ConfigSource:
    """Configuration source information"""
    path: Optional[str] = None
    format: ConfigFormat = ConfigFormat.JSON
    priority: int = 0
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    environment: Optional[Environment] = None

@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    timestamp: datetime
    source: ConfigSource
    changes: Dict[str, Any]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    change_type: str  # 'add', 'update', 'delete'

class ConfigManager:
    """Main configuration manager"""

    def __init__(self, config_dir: str = "config",
                 environment: Environment = Environment.DEVELOPMENT,
                 auto_reload: bool = True,
                 validation_enabled: bool = True):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.auto_reload = auto_reload
        self.validation_enabled = validation_enabled

        # Configuration storage
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._sources: List[ConfigSource] = []
        self._change_listeners: List[Callable[[ConfigChangeEvent], None]] = []

        # File monitoring
        self._file_hashes: Dict[str, str] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._reload_interval = 5.0  # seconds

        # Validation
        self._validator = None
        if cerberus and validation_enabled:
            self._validator = cerberus.Validator()

        # Environment variables
        self._env_prefix = "QUANTUM_EDGE_"
        self._load_dotenv()

        # Initialize
        self._load_defaults()
        self._discover_config_files()
        self._load_configuration()

        if auto_reload:
            self._start_file_monitoring()

    def _load_dotenv(self):
        """Load environment variables from .env files"""
        if load_dotenv:
            # Try multiple .env files
            env_files = [
                self.config_dir / ".env",
                self.config_dir / f".env.{self.environment.value}",
                Path(".env"),
                Path(f".env.{self.environment.value}")
            ]

            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file)
                    logger.info(f"Loaded environment file: {env_file}")
                    break

    def _load_defaults(self):
        """Load default configuration values"""
        self._defaults = {
            # Server configuration
            'server': {
                'host': '0.0.0.0',
                'port': 8080,
                'workers': 4,
                'timeout': 30,
                'max_connections': 1000
            },

            # Database configuration
            'database': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'name': 'quantum_edge',
                'user': 'quantum_user',
                'password': '',
                'pool_size': 10,
                'max_overflow': 20
            },

            # Quantum computing
            'quantum': {
                'backend': 'simulator',
                'max_qubits': 32,
                'optimization_level': 1,
                'shots': 1000,
                'timeout': 300
            },

            # Edge AI
            'edge_ai': {
                'inference_engine': 'tensorflow_lite',
                'model_cache_size': 100,
                'max_batch_size': 32,
                'precision': 'fp16',
                'memory_limit_mb': 512
            },

            # Federated learning
            'federated': {
                'enabled': True,
                'min_clients': 3,
                'max_clients': 100,
                'aggregation_rounds': 10,
                'privacy_budget': 1.0
            },

            # Security
            'security': {
                'jwt_secret': 'change-in-production',
                'jwt_expiry_hours': 24,
                'encryption_algorithm': 'AES256',
                'key_rotation_days': 30,
                'rate_limit_requests': 100,
                'rate_limit_window': 60
            },

            # Monitoring
            'monitoring': {
                'enabled': True,
                'metrics_interval': 60,
                'log_level': 'INFO',
                'alert_thresholds': {
                    'cpu_usage': 80.0,
                    'memory_usage': 85.0,
                    'error_rate': 5.0
                }
            },

            # Deployment
            'deployment': {
                'docker_enabled': True,
                'kubernetes_enabled': False,
                'auto_scaling': True,
                'min_replicas': 1,
                'max_replicas': 10
            }
        }

    def _discover_config_files(self):
        """Discover configuration files in config directory"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return

        # Priority order for loading
        priority_files = [
            f"config.{self.environment.value}.json",
            f"config.{self.environment.value}.yaml",
            f"config.{self.environment.value}.yml",
            "config.json",
            "config.yaml",
            "config.yml"
        ]

        for filename in priority_files:
            filepath = self.config_dir / filename
            if filepath.exists():
                file_format = self._detect_format(filepath)
                if file_format:
                    source = ConfigSource(
                        path=str(filepath),
                        format=file_format,
                        priority=len(self._sources),
                        environment=self.environment
                    )
                    self._sources.append(source)
                    logger.info(f"Discovered config file: {filepath}")

    def _detect_format(self, filepath: Path) -> Optional[ConfigFormat]:
        """Detect configuration file format"""
        suffix = filepath.suffix.lower()
        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.toml':
            return ConfigFormat.TOML
        elif suffix == '.env':
            return ConfigFormat.ENV
        return None

    def _load_configuration(self):
        """Load configuration from all sources"""
        # Start with defaults
        self._config = copy.deepcopy(self._defaults)

        # Load from environment variables
        self._load_from_environment()

        # Load from files (in priority order)
        for source in sorted(self._sources, key=lambda s: s.priority):
            if source.path:
                self._load_from_file(source)

        # Validate configuration
        if self.validation_enabled:
            self._validate_configuration()

        logger.info(f"Configuration loaded for environment: {self.environment.value}")

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_config = {}

        # Get all environment variables with our prefix
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(self._env_prefix):].lower()
                env_config = self._set_nested_value(env_config, config_key.split('_'), self._parse_env_value(value))

        # Merge into main config
        self._deep_merge(self._config, env_config)

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to parse as boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False

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

    def _load_from_file(self, source: ConfigSource):
        """Load configuration from file"""
        try:
            filepath = Path(source.path)
            if not filepath.exists():
                return

            # Check if file has changed
            current_hash = self._calculate_file_hash(filepath)
            if source.checksum == current_hash:
                return  # No changes

            # Load file content
            with open(filepath, 'r', encoding='utf-8') as f:
                if source.format == ConfigFormat.JSON:
                    file_config = json.load(f)
                elif source.format in [ConfigFormat.YAML]:
                    if yaml:
                        file_config = yaml.safe_load(f)
                    else:
                        logger.warning(f"YAML support not available, skipping {filepath}")
                        return
                else:
                    logger.warning(f"Unsupported format {source.format.value}, skipping {filepath}")
                    return

            # Merge into main config
            self._deep_merge(self._config, file_config)

            # Update source info
            source.checksum = current_hash
            source.last_modified = datetime.fromtimestamp(filepath.stat().st_mtime)

            logger.info(f"Loaded configuration from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load config from {source.path}: {str(e)}")

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate file hash for change detection"""
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _set_nested_value(self, config: Dict[str, Any], keys: List[str], value: Any) -> Dict[str, Any]:
        """Set nested value in configuration dictionary"""
        if len(keys) == 1:
            config[keys[0]] = value
            return config

        if keys[0] not in config:
            config[keys[0]] = {}

        config[keys[0]] = self._set_nested_value(config[keys[0]], keys[1:], value)
        return config

    def _validate_configuration(self):
        """Validate configuration against schema"""
        if not self._validator:
            return

        # Define validation schema
        schema = {
            'server': {
                'type': 'dict',
                'schema': {
                    'host': {'type': 'string'},
                    'port': {'type': 'integer', 'min': 1, 'max': 65535},
                    'workers': {'type': 'integer', 'min': 1, 'max': 100},
                    'timeout': {'type': 'integer', 'min': 1},
                    'max_connections': {'type': 'integer', 'min': 1}
                }
            },
            'quantum': {
                'type': 'dict',
                'schema': {
                    'max_qubits': {'type': 'integer', 'min': 1, 'max': 1000},
                    'optimization_level': {'type': 'integer', 'min': 0, 'max': 3},
                    'shots': {'type': 'integer', 'min': 1},
                    'timeout': {'type': 'integer', 'min': 1}
                }
            },
            'security': {
                'type': 'dict',
                'schema': {
                    'jwt_expiry_hours': {'type': 'integer', 'min': 1},
                    'rate_limit_requests': {'type': 'integer', 'min': 1},
                    'rate_limit_window': {'type': 'integer', 'min': 1}
                }
            }
        }

        if not self._validator.validate(self._config, schema):
            errors = self._validator.errors
            raise ConfigurationError(f"Configuration validation failed: {errors}")

    def _start_file_monitoring(self):
        """Start file monitoring for hot reloading"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self._monitor_thread.start()
        logger.info("Configuration file monitoring started")

    def _monitor_files(self):
        """Monitor configuration files for changes"""
        while self._monitoring:
            try:
                for source in self._sources:
                    if source.path:
                        filepath = Path(source.path)
                        if filepath.exists():
                            current_hash = self._calculate_file_hash(filepath)
                            if source.checksum != current_hash:
                                logger.info(f"Configuration file changed: {source.path}")
                                self._reload_configuration(source)

                time.sleep(self._reload_interval)

            except Exception as e:
                logger.error(f"File monitoring error: {str(e)}")
                time.sleep(self._reload_interval)

    def _reload_configuration(self, changed_source: ConfigSource):
        """Reload configuration when file changes"""
        old_config = copy.deepcopy(self._config)

        try:
            # Reload all configuration
            self._load_configuration()

            # Detect changes
            changes = self._detect_changes(old_config, self._config)
            if changes:
                # Notify listeners
                event = ConfigChangeEvent(
                    timestamp=datetime.utcnow(),
                    source=changed_source,
                    changes=changes,
                    old_values=old_config,
                    new_values=self._config,
                    change_type='update'
                )

                self._notify_change_listeners(event)

                logger.info(f"Configuration reloaded from {changed_source.path}")

        except Exception as e:
            logger.error(f"Configuration reload failed: {str(e)}")

    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes between old and new configuration"""
        changes = {}

        def detect_recursive(old_dict, new_dict, path=""):
            for key in set(old_dict.keys()) | set(new_dict.keys()):
                current_path = f"{path}.{key}" if path else key

                if key not in old_dict:
                    changes[current_path] = {'type': 'added', 'value': new_dict[key]}
                elif key not in new_dict:
                    changes[current_path] = {'type': 'removed', 'value': old_dict[key]}
                elif isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                    detect_recursive(old_dict[key], new_dict[key], current_path)
                elif old_dict[key] != new_dict[key]:
                    changes[current_path] = {
                        'type': 'changed',
                        'old_value': old_dict[key],
                        'new_value': new_dict[key]
                    }

        detect_recursive(old_config, new_config)
        return changes

    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """Add configuration change listener"""
        self._change_listeners.append(listener)

    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """Remove configuration change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    def _notify_change_listeners(self, event: ConfigChangeEvent):
        """Notify all change listeners"""
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Change listener error: {str(e)}")

    # Configuration access methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any, persist: bool = False):
        """Set configuration value"""
        keys = key.split('.')
        config = self._config

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        # Set the value
        old_value = config.get(keys[-1])
        config[keys[-1]] = value

        # Persist to file if requested
        if persist:
            self._persist_to_file(key, value)

        # Notify listeners
        if old_value != value:
            event = ConfigChangeEvent(
                timestamp=datetime.utcnow(),
                source=ConfigSource(),
                changes={key: {'type': 'changed', 'old_value': old_value, 'new_value': value}},
                old_values={key: old_value},
                new_values={key: value},
                change_type='update'
            )
            self._notify_change_listeners(event)

    def _persist_to_file(self, key: str, value: Any):
        """Persist configuration change to file"""
        # Find the highest priority config file
        if self._sources:
            source = max(self._sources, key=lambda s: s.priority)
            if source.path:
                try:
                    filepath = Path(source.path)
                    # Load current file content
                    if filepath.exists():
                        with open(filepath, 'r', encoding='utf-8') as f:
                            if source.format == ConfigFormat.JSON:
                                file_config = json.load(f)
                            elif source.format in [ConfigFormat.YAML] and yaml:
                                file_config = yaml.safe_load(f)
                            else:
                                return

                        # Update the value
                        self._set_nested_value(file_config, key.split('.'), value)

                        # Write back to file
                        with open(filepath, 'w', encoding='utf-8') as f:
                            if source.format == ConfigFormat.JSON:
                                json.dump(file_config, f, indent=2)
                            elif source.format in [ConfigFormat.YAML] and yaml:
                                yaml.dump(file_config, f, default_flow_style=False)

                        logger.info(f"Persisted configuration change to {filepath}")

                except Exception as e:
                    logger.error(f"Failed to persist config change: {str(e)}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return copy.deepcopy(self._config)

    def reload(self):
        """Manually reload configuration"""
        logger.info("Manual configuration reload triggered")
        self._load_configuration()

    def validate(self) -> bool:
        """Validate current configuration"""
        try:
            self._validate_configuration()
            return True
        except ConfigurationError:
            return False

    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'environment': self.environment.value,
            'config_dir': str(self.config_dir),
            'sources': [
                {
                    'path': source.path,
                    'format': source.format.value,
                    'priority': source.priority,
                    'last_modified': source.last_modified.isoformat() if source.last_modified else None
                }
                for source in self._sources
            ],
            'auto_reload': self.auto_reload,
            'validation_enabled': self.validation_enabled
        }

    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """Create configuration backup"""
        if not backup_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = f"config_backup_{timestamp}.json"

        backup_file = Path(backup_path)
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, default=str)

        logger.info(f"Configuration backup created: {backup_file}")
        return str(backup_file)

    def restore_from_backup(self, backup_path: str):
        """Restore configuration from backup"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise ConfigurationError(f"Backup file not found: {backup_path}")

        with open(backup_file, 'r', encoding='utf-8') as f:
            backup_config = json.load(f)

        # Replace current config
        self._config = backup_config

        # Re-validate
        if self.validation_enabled:
            self._validate_configuration()

        logger.info(f"Configuration restored from backup: {backup_path}")

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict-like access"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dict-like access"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists"""
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return True
        except (KeyError, TypeError):
            return False

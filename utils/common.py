"""
Quantum Edge AI Platform - Common Utilities

General-purpose utilities for configuration management, async operations,
performance profiling, logging enhancements, and system interactions.
"""

import asyncio
import concurrent.futures
import functools
import inspect
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, TypeVar, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import uuid
from pathlib import Path
import configparser
import yaml
import toml
import hashlib
import base64
import secrets
import string

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"

class AsyncExecutor:
    """Asynchronous task executor with thread pool"""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, os.cpu_count() * 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.loop = None

    async def run_in_executor(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run function in thread pool executor"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()

        return await self.loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def gather(self, *coroutines: Awaitable[Any]) -> List[Any]:
        """Run multiple coroutines concurrently"""
        return await asyncio.gather(*coroutines)

    async def map(self, func: Callable[[T], R], iterable: List[T]) -> List[R]:
        """Apply function to each item concurrently"""
        tasks = [self.run_in_executor(func, item) for item in iterable]
        return await asyncio.gather(*tasks)

    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)

class ConfigManager:
    """Configuration management utility"""

    def __init__(self):
        self.configs = {}
        self.config_files = {}
        self.environment_overrides = {}

    def load_config(self, file_path: Union[str, Path], format: ConfigFormat = None) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        if format is None:
            format = self._detect_format(path)

        with open(path, 'r', encoding='utf-8') as f:
            if format == ConfigFormat.JSON:
                config = json.load(f)
            elif format == ConfigFormat.YAML:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML support")
            elif format == ConfigFormat.TOML:
                try:
                    import toml
                    config = toml.load(f)
                except ImportError:
                    raise ImportError("toml required for TOML support")
            elif format == ConfigFormat.INI:
                config = self._load_ini(f)
            elif format == ConfigFormat.ENV:
                config = self._load_env(f)
            else:
                raise ValueError(f"Unsupported config format: {format}")

        # Store config
        self.configs[str(path)] = config
        self.config_files[str(path)] = {'format': format, 'path': path, 'mtime': path.stat().st_mtime}

        # Apply environment overrides
        config = self._apply_env_overrides(config)

        return config

    def save_config(self, config: Dict[str, Any], file_path: Union[str, Path],
                   format: ConfigFormat = ConfigFormat.JSON):
        """Save configuration to file"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            if format == ConfigFormat.JSON:
                json.dump(config, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                try:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML required for YAML support")
            elif format == ConfigFormat.TOML:
                try:
                    import toml
                    toml.dump(config, f)
                except ImportError:
                    raise ImportError("toml required for TOML support")
            else:
                raise ValueError(f"Unsupported save format: {format}")

    def get_config(self, key: str, default: Any = None, config_file: str = None) -> Any:
        """Get configuration value"""
        if config_file and config_file in self.configs:
            config = self.configs[config_file]
        else:
            # Get from first loaded config
            config = next(iter(self.configs.values())) if self.configs else {}

        return self._get_nested_value(config, key.split('.'), default)

    def set_config(self, key: str, value: Any, config_file: str = None):
        """Set configuration value"""
        if config_file and config_file in self.configs:
            config = self.configs[config_file]
        else:
            # Set in first loaded config
            config_file = next(iter(self.configs.keys())) if self.configs else 'default'
            config = self.configs.setdefault(config_file, {})

        self._set_nested_value(config, key.split('.'), value)

    def watch_config(self, file_path: Union[str, Path], callback: Callable[[Dict[str, Any]], None]):
        """Watch configuration file for changes"""
        path = Path(file_path)

        def watch_loop():
            last_mtime = path.stat().st_mtime

            while True:
                try:
                    current_mtime = path.stat().st_mtime
                    if current_mtime > last_mtime:
                        # File changed, reload
                        new_config = self.load_config(path)
                        callback(new_config)
                        last_mtime = current_mtime

                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error watching config file: {e}")
                    time.sleep(5)

        thread = threading.Thread(target=watch_loop, daemon=True)
        thread.start()

    def _detect_format(self, path: Path) -> ConfigFormat:
        """Detect configuration file format from extension"""
        suffix = path.suffix.lower()

        format_map = {
            '.json': ConfigFormat.JSON,
            '.yaml': ConfigFormat.YAML,
            '.yml': ConfigFormat.YAML,
            '.toml': ConfigFormat.TOML,
            '.ini': ConfigFormat.INI,
            '.cfg': ConfigFormat.INI,
            '.env': ConfigFormat.ENV
        }

        return format_map.get(suffix, ConfigFormat.JSON)

    def _load_ini(self, file) -> Dict[str, Any]:
        """Load INI configuration"""
        config = configparser.ConfigParser()
        config.read_file(file)

        result = {}
        for section in config.sections():
            result[section] = dict(config[section])

        return result

    def _load_env(self, file) -> Dict[str, Any]:
        """Load environment variables from file"""
        result = {}
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    result[key.strip()] = value.strip()
        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        result = config.copy()

        for key, env_var in self.environment_overrides.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                self._set_nested_value(result, key.split('.'), env_value)

        return result

    def _get_nested_value(self, data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """Get nested dictionary value"""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _set_nested_value(self, data: Dict[str, Any], keys: List[str], value: Any):
        """Set nested dictionary value"""
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def add_env_override(self, config_key: str, env_var: str):
        """Add environment variable override"""
        self.environment_overrides[config_key] = env_var

class PerformanceProfiler:
    """Performance profiling utility"""

    def __init__(self):
        self.profiles = {}
        self.current_profile = None

    def start_profile(self, name: str):
        """Start profiling a code block"""
        if name in self.profiles:
            profile = self.profiles[name]
        else:
            profile = {
                'name': name,
                'calls': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'avg_time': 0.0,
                'start_times': []
            }
            self.profiles[name] = profile

        profile['calls'] += 1
        profile['start_times'].append(time.perf_counter())

    def end_profile(self, name: str):
        """End profiling a code block"""
        if name not in self.profiles:
            return

        profile = self.profiles[name]
        if not profile['start_times']:
            return

        start_time = profile['start_times'].pop()
        elapsed = time.perf_counter() - start_time

        profile['total_time'] += elapsed
        profile['min_time'] = min(profile['min_time'], elapsed)
        profile['max_time'] = max(profile['max_time'], elapsed)
        profile['avg_time'] = profile['total_time'] / profile['calls']

    def profile(self, name: str = None):
        """Decorator for profiling functions"""
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_profile(profile_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_profile(profile_name)

            return wrapper
        return decorator

    def get_profile_report(self, name: str = None) -> Dict[str, Any]:
        """Get profiling report"""
        if name:
            return self.profiles.get(name, {})

        # Return all profiles
        return dict(self.profiles)

    def reset_profiles(self):
        """Reset all profiling data"""
        self.profiles.clear()

    def export_profiles(self, output_path: str):
        """Export profiling data"""
        report = {
            'exported_at': datetime.utcnow().isoformat(),
            'profiles': self.get_profile_report()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

class EnhancedLogger:
    """Enhanced logging utility with structured logging and filtering"""

    def __init__(self, name: str = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler with custom formatter
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.context = {}

    def set_context(self, **kwargs):
        """Set logging context"""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logging context"""
        self.context.clear()

    def log_structured(self, level: int, message: str, **kwargs):
        """Log structured message"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, f"{message} | {json.dumps(extra)}")

    def debug_structured(self, message: str, **kwargs):
        """Debug level structured logging"""
        self.log_structured(logging.DEBUG, message, **kwargs)

    def info_structured(self, message: str, **kwargs):
        """Info level structured logging"""
        self.log_structured(logging.INFO, message, **kwargs)

    def warning_structured(self, message: str, **kwargs):
        """Warning level structured logging"""
        self.log_structured(logging.WARNING, message, **kwargs)

    def error_structured(self, message: str, **kwargs):
        """Error level structured logging"""
        self.log_structured(logging.ERROR, message, **kwargs)

    def critical_structured(self, message: str, **kwargs):
        """Critical level structured logging"""
        self.log_structured(logging.CRITICAL, message, **kwargs)

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        self.info_structured(
            f"Performance: {operation}",
            operation=operation,
            duration=duration,
            duration_unit='seconds',
            **metrics
        )

    def log_error_with_traceback(self, error: Exception, message: str = None, **kwargs):
        """Log error with full traceback"""
        error_msg = message or str(error)
        traceback_str = traceback.format_exc()

        self.error_structured(
            f"Error: {error_msg}",
            error_type=type(error).__name__,
            traceback=traceback_str,
            **kwargs
        )

class SystemUtils:
    """System utility functions"""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            import psutil
            import platform

            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'cpu_count': os.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_total': psutil.disk_usage('/').total,
                'disk_free': psutil.disk_usage('/').free,
                'disk_percent': psutil.disk_usage('/').percent,
                'boot_time': psutil.boot_time()
            }
        except ImportError:
            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'cpu_count': os.cpu_count(),
                'error': 'psutil not available'
            }

    @staticmethod
    def generate_id(prefix: str = "", length: int = 8) -> str:
        """Generate unique ID"""
        unique_id = str(uuid.uuid4()).replace('-', '')[:length]
        return f"{prefix}{unique_id}" if prefix else unique_id

    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """Hash string using specified algorithm"""
        if algorithm == 'sha256':
            return hashlib.sha256(text.encode()).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(text.encode()).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(text.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @staticmethod
    def generate_password(length: int = 12, include_special: bool = True) -> str:
        """Generate secure random password"""
        chars = string.ascii_letters + string.digits
        if include_special:
            chars += "!@#$%^&*"

        return ''.join(secrets.choice(chars) for _ in range(length))

    @staticmethod
    def encode_base64(data: Union[str, bytes]) -> str:
        """Encode data to base64"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64encode(data).decode('utf-8')

    @staticmethod
    def decode_base64(data: str) -> bytes:
        """Decode base64 data"""
        return base64.b64decode(data)

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """Get file size in bytes"""
        return Path(path).stat().st_size

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return ".1f"
            size_bytes /= 1024.0
        return ".1f"

    @staticmethod
    def cleanup_temp_files(pattern: str = "*", age_hours: int = 24):
        """Clean up temporary files older than specified age"""
        import tempfile
        import glob

        temp_dir = Path(tempfile.gettempdir())
        cutoff_time = time.time() - (age_hours * 3600)

        cleaned = 0
        for file_path in temp_dir.glob(pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Error cleaning temp file {file_path}: {e}")

        logger.info(f"Cleaned up {cleaned} temporary files")
        return cleaned

class RetryMechanism:
    """Retry mechanism with exponential backoff"""

    def __init__(self, max_attempts: int = 3, initial_delay: float = 1.0,
                 backoff_factor: float = 2.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Retry function with exponential backoff"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_attempts - 1:
                    delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f} seconds: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_attempts} attempts failed")

        raise last_exception

    def async_retry(self, coro: Callable[..., Awaitable[T]], *args, **kwargs) -> Awaitable[T]:
        """Async version of retry mechanism"""
        async def retry_coro():
            last_exception = None

            for attempt in range(self.max_attempts):
                try:
                    return await coro(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < self.max_attempts - 1:
                        delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f} seconds: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {self.max_attempts} attempts failed")

            raise last_exception

        return retry_coro()

# Global instances
async_executor = AsyncExecutor()
config_manager = ConfigManager()
performance_profiler = PerformanceProfiler()
enhanced_logger = EnhancedLogger()
retry_mechanism = RetryMechanism()

def get_async_executor() -> AsyncExecutor:
    """Get global async executor"""
    return async_executor

def get_config_manager() -> ConfigManager:
    """Get global config manager"""
    return config_manager

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler"""
    return performance_profiler

def get_enhanced_logger() -> EnhancedLogger:
    """Get global enhanced logger"""
    return enhanced_logger

def get_retry_mechanism() -> RetryMechanism:
    """Get global retry mechanism"""
    return retry_mechanism

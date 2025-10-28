"""
Quantum Edge AI Platform - Caching Utilities

Advanced caching system with multiple backends, TTL support,
intelligent eviction policies, and distributed caching capabilities.
"""

import time
import threading
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import pickle
import sqlite3
import redis
import os
from pathlib import Path
import heapq
from collections import OrderedDict, defaultdict
import functools

logger = logging.getLogger(__name__)

class CacheBackend(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    REDIS = "redis"
    DISTRIBUTED = "distributed"

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    SIZE = "size"        # Size-based eviction

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # seconds
    size: int = 0  # bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl

    def touch(self):
        """Update access time and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size: int = 0
    uptime: float = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total_requests = self.hits + self.misses
        return self.hits / total_requests if total_requests > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate miss rate"""
        total_requests = self.hits + self.misses
        return self.misses / total_requests if total_requests > 0 else 0.0

class BaseCacheBackend:
    """Base cache backend interface"""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        raise NotImplementedError

    def clear(self) -> bool:
        """Clear all cache entries"""
        raise NotImplementedError

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        raise NotImplementedError

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        raise NotImplementedError

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        raise NotImplementedError

class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend"""

    def __init__(self, max_size: int = 1000, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.cache = OrderedDict()  # For LRU
        self.access_freq = defaultdict(int)  # For LFU
        self.access_order = []  # For FIFO
        self.stats = CacheStats()
        self.start_time = time.time()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            if key in self.cache:
                entry = self.cache[key]

                if entry.is_expired():
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return None

                entry.touch()
                if self.eviction_policy == EvictionPolicy.LRU:
                    self.cache.move_to_end(key)
                elif self.eviction_policy == EvictionPolicy.LFU:
                    self.access_freq[key] += 1

                self.stats.hits += 1
                return entry.value
            else:
                self.stats.misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            # Estimate size
            size = self._estimate_size(value)

            # Check if we need to evict
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_entries()

            # Create or update entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl=ttl,
                size=size
            )

            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size -= old_entry.size
            else:
                self.stats.total_entries += 1

            self.cache[key] = entry
            self.stats.total_size += size
            self.stats.sets += 1

            if self.eviction_policy == EvictionPolicy.FIFO:
                if key not in self.access_order:
                    self.access_order.append(key)

            return True

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_size -= entry.size
                self.stats.total_entries -= 1
                self.stats.deletes += 1

                del self.cache[key]
                if self.eviction_policy == EvictionPolicy.LFU:
                    del self.access_freq[key]
                if self.eviction_policy == EvictionPolicy.FIFO:
                    self.access_order.remove(key)

                return True

            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            self.cache.clear()
            self.access_freq.clear()
            self.access_order.clear()
            self.stats = CacheStats()
            self.start_time = time.time()

            return True

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        with self.lock:
            return key in self.cache and not self.cache[key].is_expired()

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time
            return self.stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        with self.lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._evict_entry(key)

            return len(expired_keys)

    def _evict_entries(self):
        """Evict entries based on policy"""
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            key, _ = self.cache.popitem(last=False)
            self._evict_entry(key)

        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            if self.access_freq:
                key = min(self.access_freq, key=self.access_freq.get)
                self._evict_entry(key)

        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove first in
            if self.access_order:
                key = self.access_order.pop(0)
                self._evict_entry(key)

    def _evict_entry(self, key: str):
        """Evict a specific entry"""
        if key in self.cache:
            entry = self.cache[key]
            self.stats.total_size -= entry.size
            self.stats.total_entries -= 1
            self.stats.evictions += 1

            del self.cache[key]
            if self.eviction_policy == EvictionPolicy.LFU:
                self.access_freq.pop(key, None)
            if self.eviction_policy == EvictionPolicy.FIFO:
                self.access_order.remove(key)

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, (int, float, bool)):
                return 28  # Approximate size of Python numeric objects
            elif isinstance(value, str):
                return len(value.encode('utf-8')) + 49  # String overhead
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value) + 64  # Container overhead
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items()) + 240  # Dict overhead
            elif hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                return 1000  # Default estimate
        except:
            return 1000

class FileCacheBackend(BaseCacheBackend):
    """File-based cache backend"""

    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()
        self.stats = CacheStats()
        self.start_time = time.time()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            if key in self.cache_index:
                entry = self.cache_index[key]
                cache_file = self.cache_dir / f"{key}.pkl"

                if cache_file.exists():
                    try:
                        # Check TTL
                        if entry.get('ttl') and \
                           (datetime.utcnow() - datetime.fromisoformat(entry['created_at'])).total_seconds() > entry['ttl']:
                            self.delete(key)
                            self.stats.misses += 1
                            return None

                        with open(cache_file, 'rb') as f:
                            value = pickle.load(f)

                        # Update access time
                        entry['accessed_at'] = datetime.utcnow().isoformat()
                        entry['access_count'] = entry.get('access_count', 0) + 1
                        self._save_index()

                        self.stats.hits += 1
                        return value

                    except Exception as e:
                        logger.error(f"Error reading cache file for key {key}: {e}")
                        self.delete(key)

            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            try:
                # Check size limit
                current_size = sum(entry.get('size', 0) for entry in self.cache_index.values())
                value_size = self._estimate_size(value)

                if current_size + value_size > self.max_size_bytes:
                    self._evict_to_make_space(value_size)

                # Save to file
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)

                # Update index
                self.cache_index[key] = {
                    'created_at': datetime.utcnow().isoformat(),
                    'accessed_at': datetime.utcnow().isoformat(),
                    'access_count': 1,
                    'ttl': ttl,
                    'size': value_size
                }

                self._save_index()
                self.stats.sets += 1
                self.stats.total_entries = len(self.cache_index)
                self.stats.total_size = sum(entry.get('size', 0) for entry in self.cache_index.values())

                return True

            except Exception as e:
                logger.error(f"Error setting cache key {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            if key in self.cache_index:
                cache_file = self.cache_dir / f"{key}.pkl"
                try:
                    if cache_file.exists():
                        cache_file.unlink()
                    del self.cache_index[key]
                    self._save_index()
                    self.stats.deletes += 1
                    self.stats.total_entries = len(self.cache_index)
                    return True
                except Exception as e:
                    logger.error(f"Error deleting cache key {key}: {e}")

            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time

            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

                self.cache_index.clear()
                self._save_index()
                self.stats = CacheStats()
                self.start_time = time.time()

                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return False

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        with self.lock:
            if key in self.cache_index:
                entry = self.cache_index[key]
                # Check TTL
                if entry.get('ttl') and \
                   (datetime.utcnow() - datetime.fromisoformat(entry['created_at'])).total_seconds() > entry['ttl']:
                    self.delete(key)
                    return False
                return True
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            self.stats.uptime = time.time() - self.start_time
            return self.stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        with self.lock:
            expired_keys = []
            current_time = datetime.utcnow()

            for key, entry in self.cache_index.items():
                if entry.get('ttl'):
                    created_at = datetime.fromisoformat(entry['created_at'])
                    if (current_time - created_at).total_seconds() > entry['ttl']:
                        expired_keys.append(key)

            for key in expired_keys:
                self.delete(key)

            return len(expired_keys)

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from file"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache index: {e}")

        return {}

    def _save_index(self):
        """Save cache index to file"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")

    def _evict_to_make_space(self, required_space: int):
        """Evict entries to make space"""
        # Simple LRU eviction
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get('accessed_at', '1970-01-01')
        )

        freed_space = 0
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break
            freed_space += entry.get('size', 0)
            self.delete(key)

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value"""
        return len(pickle.dumps(value))

class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend"""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 db: int = 0, password: Optional[str] = None, **kwargs):
        try:
            self.redis = redis.Redis(
                host=host, port=port, db=db, password=password,
                decode_responses=False, **kwargs
            )
            self.redis.ping()  # Test connection
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        self.stats = CacheStats()
        self.start_time = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats.uptime = time.time() - self.start_time

        try:
            value = self.redis.get(key)
            if value is None:
                self.stats.misses += 1
                return None

            # Deserialize
            value = pickle.loads(value)
            self.stats.hits += 1
            return value

        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        self.stats.uptime = time.time() - self.start_time

        try:
            # Serialize
            serialized_value = pickle.dumps(value)

            # Set with optional TTL
            if ttl:
                result = self.redis.setex(key, ttl, serialized_value)
            else:
                result = self.redis.set(key, serialized_value)

            if result:
                self.stats.sets += 1
                return True
            return False

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        self.stats.uptime = time.time() - self.start_time

        try:
            result = self.redis.delete(key)
            if result > 0:
                self.stats.deletes += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries"""
        self.stats.uptime = time.time() - self.start_time

        try:
            self.redis.flushdb()
            self.stats = CacheStats()
            self.start_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        self.stats.uptime = time.time() - self.start_time

        try:
            info = self.redis.info()
            self.stats.total_entries = info.get('db0', {}).get('keys', 0)
            self.stats.total_size = info.get('used_memory', 0)
        except:
            pass

        return self.stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries (Redis handles this automatically)"""
        return 0

class DistributedCacheBackend(BaseCacheBackend):
    """Distributed cache backend using multiple backends"""

    def __init__(self, backends: List[BaseCacheBackend], replication_factor: int = 2):
        self.backends = backends
        self.replication_factor = min(replication_factor, len(backends))
        self.stats = CacheStats()
        self.start_time = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Get value from first available backend"""
        self.stats.uptime = time.time() - self.start_time

        for backend in self.backends:
            try:
                value = backend.get(key)
                if value is not None:
                    self.stats.hits += 1
                    return value
            except Exception as e:
                logger.error(f"Error getting from backend: {e}")
                continue

        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in multiple backends"""
        self.stats.uptime = time.time() - self.start_time

        success_count = 0
        for i, backend in enumerate(self.backends[:self.replication_factor]):
            try:
                if backend.set(key, value, ttl):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error setting in backend {i}: {e}")

        success = success_count > 0
        if success:
            self.stats.sets += 1

        return success

    def delete(self, key: str) -> bool:
        """Delete from all backends"""
        self.stats.uptime = time.time() - self.start_time

        success_count = 0
        for backend in self.backends:
            try:
                if backend.delete(key):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error deleting from backend: {e}")

        success = success_count > 0
        if success:
            self.stats.deletes += 1

        return success

    def clear(self) -> bool:
        """Clear all backends"""
        self.stats.uptime = time.time() - self.start_time

        success_count = 0
        for backend in self.backends:
            try:
                if backend.clear():
                    success_count += 1
            except Exception as e:
                logger.error(f"Error clearing backend: {e}")

        return success_count > 0

    def has_key(self, key: str) -> bool:
        """Check if key exists in any backend"""
        for backend in self.backends:
            try:
                if backend.has_key(key):
                    return True
            except Exception as e:
                logger.error(f"Error checking backend: {e}")

        return False

    def get_stats(self) -> CacheStats:
        """Get combined statistics"""
        self.stats.uptime = time.time() - self.start_time
        return self.stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries in all backends"""
        total_cleaned = 0
        for backend in self.backends:
            try:
                total_cleaned += backend.cleanup_expired()
            except Exception as e:
                logger.error(f"Error cleaning backend: {e}")

        return total_cleaned

class CacheManager:
    """High-level cache manager with multiple backends and strategies"""

    def __init__(self, backend: BaseCacheBackend = None):
        self.backend = backend or MemoryCacheBackend()
        self.cache_strategies = {}
        self.cleanup_thread = None
        self.is_running = False

    def start(self, cleanup_interval: int = 300):
        """Start cache manager with background cleanup"""
        if self.is_running:
            return

        self.is_running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(cleanup_interval,),
            daemon=True
        )
        self.cleanup_thread.start()
        logger.info("Cache manager started")

    def stop(self):
        """Stop cache manager"""
        self.is_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
        logger.info("Cache manager stopped")

    def _cleanup_loop(self, interval: int):
        """Background cleanup loop"""
        while self.is_running:
            try:
                cleaned = self.backend.cleanup_expired()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired cache entries")
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

            time.sleep(interval)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        value = self.backend.get(key)
        return value if value is not None else default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        return self.backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        return self.backend.delete(key)

    def clear(self) -> bool:
        """Clear all cache entries"""
        return self.backend.clear()

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        return self.backend.has_key(key)

    def get_or_set(self, key: str, func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value from cache or compute and set it"""
        value = self.get(key)
        if value is None:
            value = func()
            self.set(key, value, ttl)
        return value

    def memoize(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Decorator for function memoization"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_data = f"{func.__name__}:{args}:{kwargs}"
                    key = hashlib.md5(key_data.encode()).hexdigest()

                return self.get_or_set(key, lambda: func(*args, **kwargs), ttl)

            return wrapper
        return decorator

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def batch_set(self, key_value_pairs: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        success = True
        for key, value in key_value_pairs.items():
            if not self.set(key, value, ttl):
                success = False
        return success

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.backend.get_stats()

    def export_cache(self, output_path: str):
        """Export cache contents (for debugging)"""
        # This is a simplified export - in practice you'd need to implement
        # proper serialization for each backend type
        stats = self.get_stats()
        export_data = {
            'stats': {
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': stats.hit_rate,
                'total_entries': stats.total_entries,
                'total_size': stats.total_size,
                'uptime': stats.uptime
            },
            'exported_at': datetime.utcnow().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Cache exported to {output_path}")

# Convenience functions
def create_memory_cache(max_size: int = 1000, policy: EvictionPolicy = EvictionPolicy.LRU) -> CacheManager:
    """Create memory-based cache"""
    backend = MemoryCacheBackend(max_size=max_size, eviction_policy=policy)
    return CacheManager(backend)

def create_file_cache(cache_dir: str = "./cache", max_size_mb: int = 100) -> CacheManager:
    """Create file-based cache"""
    backend = FileCacheBackend(cache_dir=cache_dir, max_size_mb=max_size_mb)
    return CacheManager(backend)

def create_redis_cache(host: str = 'localhost', port: int = 6379, **kwargs) -> CacheManager:
    """Create Redis-based cache"""
    backend = RedisCacheBackend(host=host, port=port, **kwargs)
    return CacheManager(backend)

def create_distributed_cache(backends: List[BaseCacheBackend], replication: int = 2) -> CacheManager:
    """Create distributed cache"""
    backend = DistributedCacheBackend(backends, replication_factor=replication)
    return CacheManager(backend)

# Global cache instance
default_cache = create_memory_cache()

def get_cache() -> CacheManager:
    """Get default cache instance"""
    return default_cache

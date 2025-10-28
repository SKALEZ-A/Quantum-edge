"""
Quantum Edge AI Platform - API Middleware

Comprehensive middleware components for authentication, rate limiting,
logging, caching, and security features across all API services.
"""

import time
import hashlib
import hmac
import base64
import json
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from functools import wraps
import threading
import re
from urllib.parse import urlparse

# Third-party imports (would be installed in production)
try:
    import jwt
    from flask import request, g, Response
    import redis
except ImportError:
    # Fallback for development without dependencies
    jwt = request = g = Response = redis = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MiddlewareConfig:
    """Middleware configuration"""
    enabled: bool = True
    priority: int = 0  # Lower priority runs first
    name: str = ""

class APIMiddleware:
    """Base API middleware class"""

    def __init__(self, config: MiddlewareConfig):
        self.config = config
        self.name = config.name or self.__class__.__name__

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request before handling"""
        return request_data

    def after_request(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process response after handling"""
        return response_data

    def on_error(self, error: Exception, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors"""
        return {
            'error': str(error),
            'status_code': 500,
            'timestamp': datetime.utcnow().isoformat()
        }

class AuthenticationMiddleware(APIMiddleware):
    """JWT-based authentication middleware"""

    def __init__(self, secret_key: str, algorithm: str = 'HS256',
                 exclude_paths: List[str] = None):
        super().__init__(MiddlewareConfig(name="AuthenticationMiddleware"))
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.exclude_paths = exclude_paths or ['/health', '/auth/login']

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate request"""
        path = request_data.get('path', '')

        # Skip authentication for excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            request_data['authenticated'] = False
            return request_data

        # Check for authorization header
        headers = request_data.get('headers', {})
        auth_header = headers.get('Authorization') or headers.get('authorization')

        if not auth_header or not auth_header.startswith('Bearer '):
            raise ValueError("Missing or invalid authorization header")

        token = auth_header.split(' ')[1]
        payload = self.verify_token(token)

        if not payload:
            raise ValueError("Invalid or expired token")

        # Add user information to request
        request_data['authenticated'] = True
        request_data['user_id'] = payload['user_id']
        request_data['roles'] = payload.get('roles', [])
        request_data['permissions'] = payload.get('permissions', [])

        return request_data

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        if not jwt:
            # Simple token verification for development
            return {'user_id': 'dev_user', 'roles': ['user'], 'permissions': ['read']}

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            if datetime.utcfromtimestamp(payload['exp']) < datetime.utcnow():
                return None

            return payload
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def create_token(self, user_id: str, roles: List[str] = None,
                    permissions: List[str] = None, expires_in: int = 3600) -> str:
        """Create JWT token"""
        if not jwt:
            return "dev_token"

        if roles is None:
            roles = []
        if permissions is None:
            permissions = []

        payload = {
            'user_id': user_id,
            'roles': roles,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

class RateLimitMiddleware(APIMiddleware):
    """Rate limiting middleware with sliding window"""

    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        super().__init__(MiddlewareConfig(name="RateLimitMiddleware"))
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
        self.lock = threading.Lock()

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit"""
        client_ip = request_data.get('client_ip', 'unknown')
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        with self.lock:
            if client_ip not in self.requests:
                self.requests[client_ip] = []

            # Remove old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > cutoff
            ]

            if len(self.requests[client_ip]) >= self.requests_per_minute:
                raise ValueError("Rate limit exceeded")

            self.requests[client_ip].append(now)

        return request_data

class LoggingMiddleware(APIMiddleware):
    """Request/response logging middleware"""

    def __init__(self, log_level: str = 'INFO', include_body: bool = False):
        super().__init__(MiddlewareConfig(name="LoggingMiddleware"))
        self.log_level = log_level
        self.include_body = include_body
        self.request_count = 0
        self.error_count = 0

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log incoming request"""
        self.request_count += 1
        request_id = request_data.get('request_id', 'unknown')

        if self.log_level == 'DEBUG' or logger.isEnabledFor(logging.DEBUG):
            body_info = ""
            if self.include_body and request_data.get('body'):
                body_preview = str(request_data['body'])[:200]
                body_info = f" body_preview='{body_preview}...'"

            logger.info(
                f"REQUEST [{request_id}] {request_data.get('method')} "
                f"{request_data.get('path')} from {request_data.get('client_ip')}{body_info}"
            )

        # Add request start time
        request_data['_start_time'] = time.time()
        return request_data

    def after_request(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log outgoing response"""
        request_data = response_data.get('_request_data', {})
        request_id = request_data.get('request_id', 'unknown')
        start_time = request_data.get('_start_time', time.time())
        processing_time = time.time() - start_time

        status_code = response_data.get('status_code', 200)

        if status_code >= 400:
            self.error_count += 1
            logger.warning(
                f"RESPONSE [{request_id}] {status_code} "
                f"{processing_time:.3f}s"
            )
        elif self.log_level == 'DEBUG' or logger.isEnabledFor(logging.DEBUG):
            logger.info(
                f"RESPONSE [{request_id}] {status_code} "
                f"{processing_time:.3f}s"
            )

        return response_data

    def on_error(self, error: Exception, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log error"""
        self.error_count += 1
        request_id = request_data.get('request_id', 'unknown')

        logger.error(
            f"ERROR [{request_id}] {type(error).__name__}: {str(error)} "
            f"for {request_data.get('method')} {request_data.get('path')}"
        )

        return super().on_error(error, request_data)

class CachingMiddleware(APIMiddleware):
    """Response caching middleware"""

    def __init__(self, redis_url: str = None, default_ttl: int = 300):
        super().__init__(MiddlewareConfig(name="CachingMiddleware"))
        self.default_ttl = default_ttl
        self.redis_client = None

        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {str(e)}")

        # In-memory cache as fallback
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache for GET requests"""
        if request_data.get('method') != 'GET':
            return request_data

        cache_key = self._generate_cache_key(request_data)

        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cached_response = json.loads(cached_data)
                    # Check if cache is still valid
                    if datetime.fromisoformat(cached_response['cached_at']) > datetime.utcnow() - timedelta(seconds=self.default_ttl):
                        return {
                            'cached': True,
                            'cache_key': cache_key,
                            **cached_response
                        }
            except Exception as e:
                logger.warning(f"Redis cache error: {str(e)}")

        # Try memory cache
        with self.cache_lock:
            if cache_key in self.memory_cache:
                cached_response = self.memory_cache[cache_key]
                if datetime.fromisoformat(cached_response['cached_at']) > datetime.utcnow() - timedelta(seconds=self.default_ttl):
                    return {
                        'cached': True,
                        'cache_key': cache_key,
                        **cached_response
                    }

        request_data['cache_key'] = cache_key
        return request_data

    def after_request(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cache successful GET responses"""
        if response_data.get('method') != 'GET' or response_data.get('status_code', 200) != 200:
            return response_data

        cache_key = response_data.get('cache_key')
        if not cache_key:
            return response_data

        # Prepare cache data
        cache_data = {
            'status_code': response_data.get('status_code'),
            'headers': response_data.get('headers', {}),
            'body': response_data.get('body'),
            'cached_at': datetime.utcnow().isoformat(),
            'ttl': self.default_ttl
        }

        # Cache in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.default_ttl,
                    json.dumps(cache_data)
                )
            except Exception as e:
                logger.warning(f"Redis cache write error: {str(e)}")

        # Cache in memory
        with self.cache_lock:
            self.memory_cache[cache_key] = cache_data

            # Clean up old entries (simple LRU)
            if len(self.memory_cache) > 1000:
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: datetime.fromisoformat(self.memory_cache[k]['cached_at'])
                )
                del self.memory_cache[oldest_key]

        return response_data

    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key from request"""
        key_parts = [
            request_data.get('method', 'GET'),
            request_data.get('path', '/'),
            str(sorted(request_data.get('query_params', {}).items())),
            request_data.get('user_id', 'anonymous')
        ]
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()

class SecurityMiddleware(APIMiddleware):
    """Security middleware for input validation and sanitization"""

    def __init__(self, max_body_size: int = 1048576,  # 1MB
                 allowed_content_types: List[str] = None):
        super().__init__(MiddlewareConfig(name="SecurityMiddleware"))
        self.max_body_size = max_body_size
        self.allowed_content_types = allowed_content_types or [
            'application/json',
            'application/x-www-form-urlencoded',
            'text/plain'
        ]

        # SQL injection patterns
        self.sql_patterns = [
            r'\bUNION\b.*\bSELECT\b',
            r'\bDROP\b.*\bTABLE\b',
            r'\bDELETE\b.*\bFROM\b',
            r';\s*--',
            r';\s*/\*'
        ]

        # XSS patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>'
        ]

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize request"""
        # Check body size
        body = request_data.get('body')
        if body and len(body) > self.max_body_size:
            raise ValueError(f"Request body too large: {len(body)} bytes")

        # Check content type
        headers = request_data.get('headers', {})
        content_type = headers.get('Content-Type', '').split(';')[0].lower()
        if content_type and content_type not in self.allowed_content_types:
            raise ValueError(f"Unsupported content type: {content_type}")

        # Validate path
        path = request_data.get('path', '')
        if not self._is_valid_path(path):
            raise ValueError(f"Invalid path: {path}")

        # Sanitize input
        request_data = self._sanitize_request(request_data)

        # Check for malicious patterns
        self._check_security_patterns(request_data)

        return request_data

    def _is_valid_path(self, path: str) -> bool:
        """Validate request path"""
        if not path or not path.startswith('/'):
            return False

        # Check for path traversal attempts
        if '..' in path or '%' in path:
            # More thorough check for encoded traversal
            decoded_path = path
            while '%' in decoded_path:
                try:
                    decoded_path = urllib.parse.unquote(decoded_path)
                except:
                    break

            if '..' in decoded_path:
                return False

        # Check for extremely long paths
        if len(path) > 2048:
            return False

        return True

    def _sanitize_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data"""
        # Sanitize headers
        headers = request_data.get('headers', {})
        sanitized_headers = {}
        for key, value in headers.items():
            if isinstance(value, str):
                sanitized_headers[key] = self._sanitize_string(value)
            else:
                sanitized_headers[key] = value

        request_data['headers'] = sanitized_headers

        # Sanitize query parameters
        query_params = request_data.get('query_params', {})
        sanitized_query = {}
        for key, values in query_params.items():
            if isinstance(values, list):
                sanitized_query[key] = [self._sanitize_string(v) if isinstance(v, str) else v for v in values]
            elif isinstance(values, str):
                sanitized_query[key] = self._sanitize_string(values)
            else:
                sanitized_query[key] = values

        request_data['query_params'] = sanitized_query

        return request_data

    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input"""
        if not isinstance(text, str):
            return str(text)

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>]', '', text)
        return sanitized.strip()

    def _check_security_patterns(self, request_data: Dict[str, Any]):
        """Check for malicious patterns"""
        # Check query parameters and body for SQL injection
        text_to_check = []

        # Query parameters
        for key, values in request_data.get('query_params', {}).items():
            text_to_check.extend([key] if isinstance(key, str) else [])
            if isinstance(values, list):
                text_to_check.extend([v for v in values if isinstance(v, str)])
            elif isinstance(values, str):
                text_to_check.append(values)

        # Request body (if JSON)
        body = request_data.get('body')
        if body:
            try:
                body_text = body.decode('utf-8')
                text_to_check.append(body_text)
            except:
                pass

        # Check all text for patterns
        for text in text_to_check:
            # SQL injection check
            for pattern in self.sql_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    raise ValueError("Potential SQL injection detected")

            # XSS check
            for pattern in self.xss_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    raise ValueError("Potential XSS attack detected")

class CompressionMiddleware(APIMiddleware):
    """Response compression middleware"""

    def __init__(self, compression_level: int = 6, min_size: int = 1024):
        super().__init__(MiddlewareConfig(name="CompressionMiddleware"))
        self.compression_level = compression_level
        self.min_size = min_size

    def after_request(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress response if beneficial"""
        body = response_data.get('body')
        if not body or len(body) < self.min_size:
            return response_data

        # Check if client accepts compression
        headers = response_data.get('headers', {})
        accept_encoding = headers.get('Accept-Encoding', '')

        if 'gzip' in accept_encoding:
            try:
                import gzip
                compressed_body = gzip.compress(body, compresslevel=self.compression_level)
                if len(compressed_body) < len(body):  # Only compress if smaller
                    response_data['body'] = compressed_body
                    response_data['headers'] = response_data.get('headers', {})
                    response_data['headers']['Content-Encoding'] = 'gzip'
                    response_data['headers']['Content-Length'] = str(len(compressed_body))
            except ImportError:
                logger.warning("gzip not available for compression")
        elif 'deflate' in accept_encoding:
            try:
                import zlib
                compressed_body = zlib.compress(body, level=self.compression_level)
                if len(compressed_body) < len(body):
                    response_data['body'] = compressed_body
                    response_data['headers'] = response_data.get('headers', {})
                    response_data['headers']['Content-Encoding'] = 'deflate'
                    response_data['headers']['Content-Length'] = str(len(compressed_body))
            except ImportError:
                logger.warning("zlib not available for compression")

        return response_data

class MetricsMiddleware(APIMiddleware):
    """Metrics collection middleware"""

    def __init__(self, collect_detailed_metrics: bool = True):
        super().__init__(MiddlewareConfig(name="MetricsMiddleware"))
        self.collect_detailed_metrics = collect_detailed_metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'response_times': [],
            'status_codes': {},
            'endpoint_metrics': {},
            'client_metrics': {}
        }
        self.lock = threading.Lock()

    def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track request start"""
        request_data['_metrics_start'] = time.time()
        return request_data

    def after_request(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect response metrics"""
        with self.lock:
            self.metrics['total_requests'] += 1

            status_code = response_data.get('status_code', 200)
            if status_code < 400:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1

            # Status code distribution
            self.metrics['status_codes'][status_code] = \
                self.metrics['status_codes'].get(status_code, 0) + 1

            # Response time
            start_time = response_data.get('_request_data', {}).get('_metrics_start')
            if start_time:
                response_time = time.time() - start_time
                self.metrics['total_response_time'] += response_time
                self.metrics['response_times'].append(response_time)

                # Keep only last 1000 response times
                if len(self.metrics['response_times']) > 1000:
                    self.metrics['response_times'] = self.metrics['response_times'][-1000:]

            # Endpoint metrics
            if self.collect_detailed_metrics:
                path = response_data.get('_request_data', {}).get('path', 'unknown')
                if path not in self.metrics['endpoint_metrics']:
                    self.metrics['endpoint_metrics'][path] = {
                        'requests': 0,
                        'errors': 0,
                        'avg_response_time': 0.0
                    }

                endpoint_stats = self.metrics['endpoint_metrics'][path]
                endpoint_stats['requests'] += 1
                if status_code >= 400:
                    endpoint_stats['errors'] += 1

                if start_time:
                    current_avg = endpoint_stats['avg_response_time']
                    endpoint_stats['avg_response_time'] = \
                        (current_avg * (endpoint_stats['requests'] - 1) + response_time) / endpoint_stats['requests']

        return response_data

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            metrics_copy = dict(self.metrics)

            # Calculate averages
            if metrics_copy['total_requests'] > 0:
                metrics_copy['avg_response_time'] = \
                    metrics_copy['total_response_time'] / metrics_copy['total_requests']

            # Calculate percentiles
            if metrics_copy['response_times']:
                sorted_times = sorted(metrics_copy['response_times'])
                metrics_copy['p50_response_time'] = sorted_times[len(sorted_times) // 2]
                metrics_copy['p95_response_time'] = sorted_times[int(len(sorted_times) * 0.95)]
                metrics_copy['p99_response_time'] = sorted_times[int(len(sorted_times) * 0.99)]

            return metrics_copy

class MiddlewarePipeline:
    """Pipeline for executing middleware in order"""

    def __init__(self):
        self.middlewares: List[APIMiddleware] = []
        self.lock = threading.Lock()

    def add_middleware(self, middleware: APIMiddleware):
        """Add middleware to pipeline"""
        with self.lock:
            self.middlewares.append(middleware)
            # Sort by priority
            self.middlewares.sort(key=lambda m: m.config.priority)

    def remove_middleware(self, middleware_name: str):
        """Remove middleware from pipeline"""
        with self.lock:
            self.middlewares = [
                m for m in self.middlewares
                if m.name != middleware_name
            ]

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through middleware pipeline"""
        current_data = dict(request_data)

        for middleware in self.middlewares:
            if not middleware.config.enabled:
                continue

            try:
                current_data = middleware.before_request(current_data)
            except Exception as e:
                error_response = middleware.on_error(e, current_data)
                return error_response

        return current_data

    async def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process response through middleware pipeline"""
        current_data = dict(response_data)

        # Process in reverse order for response middleware
        for middleware in reversed(self.middlewares):
            if not middleware.config.enabled:
                continue

            try:
                current_data = middleware.after_request(current_data)
            except Exception as e:
                logger.error(f"Response middleware error in {middleware.name}: {str(e)}")
                # Continue with other middleware

        return current_data

# Flask-specific middleware decorators
def require_auth(f):
    """Flask decorator for authentication requirement"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id'):
            return Response('Authentication required', 401, {'WWW-Authenticate': 'Bearer'})
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(requests_per_minute: int = 60):
    """Flask decorator for rate limiting"""
    def decorator(f):
        limiter = RateLimitMiddleware(requests_per_minute)
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Create mock request data
                request_data = {
                    'client_ip': request.remote_addr if request else 'unknown'
                }
                limiter.before_request(request_data)
                return f(*args, **kwargs)
            except ValueError as e:
                return Response(str(e), 429)
        return decorated_function
    return decorator

# Export key classes
__all__ = [
    'APIMiddleware',
    'AuthenticationMiddleware',
    'RateLimitMiddleware',
    'LoggingMiddleware',
    'CachingMiddleware',
    'SecurityMiddleware',
    'CompressionMiddleware',
    'MetricsMiddleware',
    'MiddlewarePipeline',
    'require_auth',
    'rate_limit'
]

"""
Quantum Edge AI Platform - API Gateway

Intelligent API gateway for quantum edge AI services with load balancing,
circuit breaking, request routing, and advanced security features.
"""

import asyncio
import json
import time
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlparse, parse_qs, urljoin
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import random

# Third-party imports (would be installed in production)
try:
    import aiohttp
    from aiohttp import web, ClientSession, ClientTimeout
    import jwt
    from multidict import MultiDict
except ImportError:
    # Fallback for development without dependencies
    aiohttp = web = ClientSession = ClientTimeout = None
    jwt = MultiDict = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouteType(Enum):
    """API route types"""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    FEDERATED = "federated"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    IP_HASH = "ip_hash"

@dataclass
class BackendService:
    """Backend service configuration"""
    name: str
    url: str
    route_type: RouteType
    weight: int = 1
    max_connections: int = 100
    timeout_seconds: int = 30
    health_check_path: str = "/health"
    health_check_interval: int = 30
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    # Runtime state
    active_connections: int = 0
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    circuit_breaker_until: Optional[datetime] = None

@dataclass
class GatewayRoute:
    """API gateway route configuration"""
    path_pattern: str
    backend_service: str
    methods: List[str]
    authentication_required: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    cors_enabled: bool = True
    request_transform: Optional[Callable] = None
    response_transform: Optional[Callable] = None

@dataclass
class GatewayRequest:
    """Gateway request information"""
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, List[str]]
    body: Optional[bytes]
    client_ip: str
    user_agent: str
    request_id: str
    timestamp: datetime
    authenticated_user: Optional[str] = None
    route: Optional[GatewayRoute] = None

@dataclass
class GatewayResponse:
    """Gateway response information"""
    status_code: int
    headers: Dict[str, str]
    body: Optional[bytes]
    processing_time: float
    backend_service: Optional[str] = None
    backend_url: Optional[str] = None

class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def can_attempt_request(self) -> bool:
        """Check if request can be attempted"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False

class LoadBalancer:
    """Load balancer for backend services"""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_index = 0
        self.ip_hash_cache: Dict[str, int] = {}

    def select_backend(self, backends: List[BackendService], client_ip: str = None) -> Optional[BackendService]:
        """Select backend service based on strategy"""
        healthy_backends = [b for b in backends if b.is_healthy and self._can_attempt_request(b)]

        if not healthy_backends:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            backend = healthy_backends[self.round_robin_index % len(healthy_backends)]
            self.round_robin_index += 1
            return backend

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda b: b.active_connections)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            total_weight = sum(b.weight for b in healthy_backends)
            rand = random.uniform(0, total_weight)
            current_weight = 0
            for backend in healthy_backends:
                current_weight += backend.weight
                if rand <= current_weight:
                    return backend

        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_backends)

        elif self.strategy == LoadBalancingStrategy.IP_HASH and client_ip:
            hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
            index = hash_value % len(healthy_backends)
            return healthy_backends[index]

        # Default to round-robin
        return healthy_backends[self.round_robin_index % len(healthy_backends)]

    def _can_attempt_request(self, backend: BackendService) -> bool:
        """Check if request can be attempted on backend"""
        if backend.circuit_breaker_until and datetime.utcnow() < backend.circuit_breaker_until:
            return False
        return backend.active_connections < backend.max_connections

class RateLimiter:
    """Advanced rate limiter with sliding window"""

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []

            # Remove old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff
            ]

            if len(self.requests[client_id]) < self.requests_per_window:
                self.requests[client_id].append(now)
                return True

            return False

class QuantumAPIGateway:
    """Main API Gateway for Quantum Edge AI Platform"""

    def __init__(self, host: str = '0.0.0.0', port: int = 8080,
                 enable_auth: bool = True, enable_rate_limiting: bool = True):
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.enable_rate_limiting = enable_rate_limiting

        # Initialize components
        self.auth_manager = None  # Will be set during initialization
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)

        # Backend services and routes
        self.backend_services: Dict[str, BackendService] = {}
        self.routes: List[GatewayRoute] = []
        self.route_cache: Dict[str, GatewayRoute] = {}

        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Rate limiters per route
        self.rate_limiters: Dict[str, RateLimiter] = {}

        # HTTP client session
        self.session: Optional[ClientSession] = None

        # Monitoring
        self.request_count = 0
        self.error_count = 0
        self.avg_response_time = 0.0

        # Background tasks
        self.health_check_thread: Optional[threading.Thread] = None
        self.running = False

    def add_backend_service(self, service: BackendService):
        """Add backend service"""
        self.backend_services[service.name] = service
        self.circuit_breakers[service.name] = CircuitBreaker(
            failure_threshold=service.circuit_breaker_threshold,
            recovery_timeout=service.circuit_breaker_timeout
        )
        logger.info(f"Added backend service: {service.name} -> {service.url}")

    def add_route(self, route: GatewayRoute):
        """Add gateway route"""
        self.routes.append(route)
        self.rate_limiters[route.path_pattern] = RateLimiter(
            route.rate_limit_requests,
            route.rate_limit_window
        )
        # Clear route cache
        self.route_cache.clear()
        logger.info(f"Added route: {route.methods} {route.path_pattern} -> {route.backend_service}")

    def initialize_auth(self, secret_key: str):
        """Initialize authentication manager"""
        from .rest_api import AuthenticationManager
        self.auth_manager = AuthenticationManager(secret_key)

    async def start_gateway(self):
        """Start the API gateway"""
        if not aiohttp:
            logger.error("aiohttp not available. Install aiohttp to run the gateway.")
            return

        self.running = True
        self.session = ClientSession(timeout=ClientTimeout(total=30))

        # Start health checks
        self._start_health_checks()

        # Create aiohttp app
        app = web.Application(middlewares=[
            self._cors_middleware,
            self._logging_middleware,
            self._auth_middleware,
            self._rate_limit_middleware
        ])

        # Main request handler
        app.router.add_route('*', '/{path:.*}', self._handle_request)

        logger.info(f"Starting Quantum API Gateway on {self.host}:{self.port}")

        try:
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            logger.info("API Gateway started successfully")

            # Keep running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start API Gateway: {str(e)}")
        finally:
            await self.session.close()

    def stop_gateway(self):
        """Stop the API gateway"""
        self.running = False
        logger.info("API Gateway stopped")

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Main request handler"""
        start_time = time.time()

        try:
            # Create gateway request
            gateway_request = await self._create_gateway_request(request)

            # Find matching route
            route = self._find_route(gateway_request.method, gateway_request.path)
            if not route:
                return web.Response(status=404, text="Route not found")

            gateway_request.route = route

            # Get backend service
            backend = self.backend_services.get(route.backend_service)
            if not backend:
                return web.Response(status=503, text="Backend service not available")

            # Select backend instance (for load balancing)
            selected_backend = self.load_balancer.select_backend([backend], gateway_request.client_ip)
            if not selected_backend:
                return web.Response(status=503, text="No healthy backend available")

            # Transform request if needed
            transformed_request = gateway_request
            if route.request_transform:
                transformed_request = route.request_transform(gateway_request)

            # Forward request to backend
            response = await self._forward_request(transformed_request, selected_backend)

            # Transform response if needed
            if route.response_transform:
                response = route.response_transform(response)

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, response.status_code == 200)

            # Add gateway headers
            response.headers['X-Gateway-Request-ID'] = gateway_request.request_id
            response.headers['X-Gateway-Processing-Time'] = f"{processing_time:.3f}s"
            response.headers['X-Gateway-Backend'] = selected_backend.name

            return response

        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)
            return web.Response(status=500, text="Internal gateway error")

    async def _create_gateway_request(self, request: web.Request) -> GatewayRequest:
        """Create gateway request from aiohttp request"""
        body = await request.read()

        # Extract client IP
        client_ip = request.headers.get('X-Forwarded-For', request.remote).split(',')[0].strip()

        return GatewayRequest(
            method=request.method,
            path=request.path,
            headers=dict(request.headers),
            query_params=dict(request.query),
            body=body if body else None,
            client_ip=client_ip,
            user_agent=request.headers.get('User-Agent', ''),
            request_id=request.headers.get('X-Request-ID', hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:16]),
            timestamp=datetime.utcnow()
        )

    def _find_route(self, method: str, path: str) -> Optional[GatewayRoute]:
        """Find matching route for request"""
        # Check cache first
        cache_key = f"{method}:{path}"
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]

        # Find matching route
        for route in self.routes:
            if method not in route.methods:
                continue

            # Simple pattern matching (in production, use regex or path templates)
            if self._matches_pattern(path, route.path_pattern):
                self.route_cache[cache_key] = route
                return route

        return None

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern"""
        # Simple wildcard matching
        if '*' in pattern:
            regex_pattern = pattern.replace('*', '.*')
            return bool(re.match(f"^{regex_pattern}$", path))

        return path == pattern

    async def _forward_request(self, gateway_request: GatewayRequest,
                             backend: BackendService) -> web.Response:
        """Forward request to backend service"""
        try:
            # Build backend URL
            backend_url = urljoin(backend.url, gateway_request.path)

            # Prepare headers
            headers = dict(gateway_request.headers)
            headers['X-Forwarded-By'] = 'Quantum-API-Gateway'
            headers['X-Client-IP'] = gateway_request.client_ip

            # Remove hop-by-hop headers
            hop_by_hop_headers = [
                'connection', 'keep-alive', 'proxy-authenticate',
                'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
            ]
            for header in hop_by_hop_headers:
                headers.pop(header, None)

            # Prepare request
            timeout = ClientTimeout(total=backend.timeout_seconds)

            # Increment active connections
            backend.active_connections += 1

            try:
                async with self.session.request(
                    method=gateway_request.method,
                    url=backend_url,
                    headers=headers,
                    params=gateway_request.query_params,
                    data=gateway_request.body,
                    timeout=timeout
                ) as backend_response:

                    # Read response
                    response_body = await backend_response.read()
                    response_headers = dict(backend_response.headers)

                    # Create gateway response
                    response = web.Response(
                        status=backend_response.status,
                        headers=response_headers,
                        body=response_body
                    )

                    # Record success
                    circuit_breaker = self.circuit_breakers[backend.name]
                    circuit_breaker.record_success()

                    return response

            finally:
                # Decrement active connections
                backend.active_connections -= 1

        except Exception as e:
            # Record failure
            circuit_breaker = self.circuit_breakers[backend.name]
            circuit_breaker.record_failure()

            logger.error(f"Backend request failed for {backend.name}: {str(e)}")
            raise

    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler):
        """CORS middleware"""
        # Add CORS headers
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Request-ID'
        return response

    @web.middleware
    async def _logging_middleware(self, request: web.Request, handler):
        """Request logging middleware"""
        start_time = time.time()
        response = await handler(request)
        processing_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.path} {response.status} "
            f"{processing_time:.3f}s {request.remote}"
        )

        return response

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        """Authentication middleware"""
        if not self.enable_auth:
            return await handler(request)

        # Skip auth for health checks
        if request.path.startswith('/health'):
            return await handler(request)

        # Check for authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return web.Response(status=401, text="Missing or invalid authorization header")

        token = auth_header.split(' ')[1]

        if not self.auth_manager:
            return web.Response(status=500, text="Authentication manager not initialized")

        payload = self.auth_manager.verify_token(token)
        if not payload:
            return web.Response(status=401, text="Invalid or expired token")

        # Add user info to request
        request._user_id = payload['user_id']
        request._roles = payload.get('roles', [])
        request._permissions = payload.get('permissions', [])

        return await handler(request)

    @web.middleware
    async def _rate_limit_middleware(self, request: web.Request, handler):
        """Rate limiting middleware"""
        if not self.enable_rate_limiting:
            return await handler(request)

        # Find route for rate limiting
        route = self._find_route(request.method, request.path)
        if route and route.path_pattern in self.rate_limiters:
            limiter = self.rate_limiters[route.path_pattern]
            client_id = request.headers.get('X-Forwarded-For', request.remote).split(',')[0].strip()

            if not limiter.is_allowed(client_id):
                return web.Response(status=429, text="Rate limit exceeded")

        return await handler(request)

    def _start_health_checks(self):
        """Start background health checks"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return

        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()

    def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                for service_name, service in self.backend_services.items():
                    asyncio.run(self._check_service_health(service))

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
                time.sleep(5)

    async def _check_service_health(self, service: BackendService):
        """Check health of a backend service"""
        try:
            health_url = urljoin(service.url, service.health_check_path)
            timeout = ClientTimeout(total=5)  # Short timeout for health checks

            async with self.session.get(health_url, timeout=timeout) as response:
                is_healthy = response.status == 200
                service.is_healthy = is_healthy
                service.last_health_check = datetime.utcnow()

                if is_healthy:
                    service.consecutive_failures = 0
                    logger.debug(f"Service {service.name} is healthy")
                else:
                    service.consecutive_failures += 1
                    logger.warning(f"Service {service.name} health check failed: {response.status}")

                # Update circuit breaker
                if service.consecutive_failures >= service.circuit_breaker_threshold:
                    service.circuit_breaker_until = datetime.utcnow() + timedelta(seconds=service.circuit_breaker_timeout)
                    logger.warning(f"Circuit breaker opened for service {service.name}")

        except Exception as e:
            service.consecutive_failures += 1
            service.is_healthy = False
            service.last_health_check = datetime.utcnow()
            logger.warning(f"Health check failed for service {service.name}: {str(e)}")

    def _update_metrics(self, processing_time: float, success: bool):
        """Update gateway metrics"""
        self.request_count += 1
        if not success:
            self.error_count += 1

        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.request_count - 1)) + processing_time
        ) / self.request_count

    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'avg_response_time': self.avg_response_time,
            'backend_services': {
                name: {
                    'url': service.url,
                    'healthy': service.is_healthy,
                    'active_connections': service.active_connections,
                    'consecutive_failures': service.consecutive_failures
                }
                for name, service in self.backend_services.items()
            }
        }

if __name__ == '__main__':
    # Example usage
    async def main():
        gateway = QuantumAPIGateway()

        # Initialize authentication
        gateway.initialize_auth('quantum-gateway-secret-key-change-in-production')

        # Add backend services
        rest_service = BackendService(
            name="rest-api",
            url="http://localhost:8081",
            route_type=RouteType.REST
        )

        graphql_service = BackendService(
            name="graphql-api",
            url="http://localhost:8082",
            route_type=RouteType.GRAPHQL
        )

        websocket_service = BackendService(
            name="websocket-service",
            url="ws://localhost:8083",
            route_type=RouteType.WEBSOCKET
        )

        gateway.add_backend_service(rest_service)
        gateway.add_backend_service(graphql_service)
        gateway.add_backend_service(websocket_service)

        # Add routes
        gateway.add_route(GatewayRoute(
            path_pattern="/api/*",
            backend_service="rest-api",
            methods=["GET", "POST", "PUT", "DELETE"],
            authentication_required=True
        ))

        gateway.add_route(GatewayRoute(
            path_pattern="/graphql",
            backend_service="graphql-api",
            methods=["POST"],
            authentication_required=True
        ))

        gateway.add_route(GatewayRoute(
            path_pattern="/ws",
            backend_service="websocket-service",
            methods=["GET"],
            authentication_required=True
        ))

        await gateway.start_gateway()

    asyncio.run(main())

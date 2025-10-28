"""
Quantum Edge AI Platform - API Services Module

This module provides comprehensive API services for the quantum edge AI platform,
including REST APIs, GraphQL endpoints, WebSocket services, and API gateway functionality.
"""

__version__ = "1.0.0"
__author__ = "Quantum Edge AI Platform Team"

from .rest_api import QuantumEdgeAPI
from .graphql_api import QuantumGraphQLAPI
from .websocket_service import QuantumWebSocketService
from .api_gateway import QuantumAPIGateway
from .middleware import APIMiddleware, AuthenticationMiddleware, RateLimitMiddleware
from .models import APIRequest, APIResponse, UserSession, APIMetrics

__all__ = [
    'QuantumEdgeAPI',
    'QuantumGraphQLAPI',
    'QuantumWebSocketService',
    'QuantumAPIGateway',
    'APIMiddleware',
    'AuthenticationMiddleware',
    'RateLimitMiddleware',
    'APIRequest',
    'APIResponse',
    'UserSession',
    'APIMetrics'
]

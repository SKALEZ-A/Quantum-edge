"""
Quantum Edge AI Platform - Web Interface

Modern web dashboard for monitoring, managing, and interacting with
the Quantum Edge AI Platform through intuitive visualizations and controls.
"""

import logging
from typing import Dict, List, Any, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "Quantum Edge AI Platform Team"

# Import main components
try:
    from .app import create_app, QuantumEdgeDashboard
    from .routes import register_routes
    from .websocket_handler import WebSocketHandler
    from .auth import AuthManager
    from .components import DashboardComponents

    __all__ = [
        'create_app',
        'QuantumEdgeDashboard',
        'register_routes',
        'WebSocketHandler',
        'AuthManager',
        'DashboardComponents'
    ]

except ImportError as e:
    logger.warning(f"Some web interface components not available: {e}")
    __all__ = []

# Global instances
dashboard = None

def get_dashboard() -> Optional['QuantumEdgeDashboard']:
    """Get global dashboard instance"""
    return dashboard

def initialize_dashboard(config: Dict[str, Any] = None) -> 'QuantumEdgeDashboard':
    """Initialize the web dashboard"""
    global dashboard

    if dashboard is None:
        from .app import QuantumEdgeDashboard
        dashboard = QuantumEdgeDashboard(config or {})

    return dashboard

def run_dashboard(host: str = '0.0.0.0', port: int = 5000,
                 debug: bool = False, config: Dict[str, Any] = None):
    """Run the dashboard server"""
    dashboard = initialize_dashboard(config)
    app = dashboard.app

    logger.info(f"Starting Quantum Edge AI Dashboard on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

#!/usr/bin/env python3
"""
Comprehensive Web Dashboard for Quantum Edge AI Platform

This module provides a full-featured web dashboard with:
- Real-time monitoring and metrics visualization
- Model management and deployment interface
- Federated learning coordination
- Privacy budget monitoring
- System administration tools
- Interactive quantum circuit visualization
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import aiohttp
import aiofiles
from aiohttp import web, WSMsgType
import jinja2
import aiohttp_jinja2
from aiohttp_security import SessionIdentityPolicy, authorized_userid
from aiohttp_security import setup as setup_security
from cryptography.fernet import Fernet
import secrets
import hashlib
import base64

# Import platform components
from quantum_edge_ai.monitoring.metrics_collector import MetricsStorage, MetricsExporter
from quantum_edge_ai.edge_runtime.inference_engine import EdgeInferenceEngine
from quantum_edge_ai.quantum_algorithms.quantum_ml import QuantumMachineLearning
from quantum_edge_ai.privacy_security.privacy import PrivacyEngine
from quantum_edge_ai.federated_learning.federated_server import FederatedLearningServer


class DashboardWebSocketManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.connections: Dict[str, web.WebSocketResponse] = {}
        self.subscriptions: Dict[str, set] = {}
        self.lock = asyncio.Lock()

    async def register_connection(self, ws: web.WebSocketResponse, client_id: str):
        """Register a new WebSocket connection."""
        async with self.lock:
            self.connections[client_id] = ws
            self.subscriptions[client_id] = set()

    async def unregister_connection(self, client_id: str):
        """Unregister a WebSocket connection."""
        async with self.lock:
            if client_id in self.connections:
                del self.connections[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]

    async def subscribe(self, client_id: str, topic: str):
        """Subscribe client to a topic."""
        async with self.lock:
            if client_id in self.subscriptions:
                self.subscriptions[client_id].add(topic)

    async def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe client from a topic."""
        async with self.lock:
            if client_id in self.subscriptions:
                self.subscriptions[client_id].discard(topic)

    async def broadcast(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all subscribers of a topic."""
        async with self.lock:
            message_data = {
                'topic': topic,
                'timestamp': datetime.now().isoformat(),
                'data': message
            }

            for client_id, ws in self.connections.items():
                if topic in self.subscriptions.get(client_id, set()):
                    try:
                        await ws.send_json(message_data)
                    except Exception as e:
                        print(f"Failed to send message to {client_id}: {e}")


class QuantumEdgeDashboard:
    """
    Main dashboard application class.

    Provides web interface for monitoring and managing the Quantum Edge AI Platform.
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port

        # Core components
        self.app = web.Application()
        self.ws_manager = DashboardWebSocketManager()

        # Platform components
        self.metrics_storage: Optional[MetricsStorage] = None
        self.inference_engine: Optional[EdgeInferenceEngine] = None
        self.quantum_engine: Optional[QuantumMachineLearning] = None
        self.privacy_engine: Optional[PrivacyEngine] = None
        self.federated_server: Optional[FederatedLearningServer] = None

        # Dashboard state
        self.dashboard_data = {}
        self.update_interval = 5.0  # seconds
        self.update_task: Optional[asyncio.Task] = None

        # Setup application
        self._setup_routes()
        self._setup_templates()
        self._setup_static_files()
        self._setup_websockets()

    def _setup_routes(self):
        """Setup web application routes."""
        # Main dashboard
        self.app.router.add_get('/', self.dashboard_handler)
        self.app.router.add_get('/dashboard', self.dashboard_handler)

        # API endpoints
        self.app.router.add_get('/api/metrics', self.metrics_api_handler)
        self.app.router.add_get('/api/models', self.models_api_handler)
        self.app.router.add_post('/api/models/deploy', self.deploy_model_handler)
        self.app.router.add_get('/api/federated/status', self.federated_status_handler)
        self.app.router.add_post('/api/federated/start', self.start_federated_handler)
        self.app.router.add_get('/api/privacy/budget', self.privacy_budget_handler)
        self.app.router.add_get('/api/system/status', self.system_status_handler)

        # Inference endpoints
        self.app.router.add_post('/api/inference', self.inference_handler)
        self.app.router.add_get('/api/inference/history', self.inference_history_handler)

        # Quantum endpoints
        self.app.router.add_post('/api/quantum/circuit', self.quantum_circuit_handler)
        self.app.router.add_get('/api/quantum/jobs', self.quantum_jobs_handler)

        # WebSocket endpoint
        self.app.router.add_get('/ws', self.websocket_handler)

    def _setup_templates(self):
        """Setup Jinja2 templates."""
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)

        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(template_dir))
        )

    def _setup_static_files(self):
        """Setup static file serving."""
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)

        self.app.router.add_static('/static', static_dir)

    def _setup_websockets(self):
        """Setup WebSocket support."""
        pass  # WebSocket handler is already added to routes

    def initialize_platform_components(self,
                                     metrics_storage: MetricsStorage = None,
                                     inference_engine: EdgeInferenceEngine = None,
                                     quantum_engine: QuantumMachineLearning = None,
                                     privacy_engine: PrivacyEngine = None,
                                     federated_server: FederatedLearningServer = None):
        """Initialize platform components."""
        self.metrics_storage = metrics_storage or MetricsStorage()
        self.inference_engine = inference_engine or EdgeInferenceEngine()
        self.quantum_engine = quantum_engine or QuantumMachineLearning(n_qubits=4)
        self.privacy_engine = privacy_engine or PrivacyEngine()
        self.federated_server = federated_server

    async def start_dashboard(self):
        """Start the dashboard server."""
        print("🚀 Starting Quantum Edge AI Dashboard")
        print(f"📍 Server will be available at http://{self.host}:{self.port}")

        # Start background update task
        self.update_task = asyncio.create_task(self._background_update_loop())

        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            print("✅ Dashboard started successfully")

            # Keep running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\\n⏹️  Shutting down dashboard...")
        finally:
            if self.update_task:
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass

    async def _background_update_loop(self):
        """Background loop for updating dashboard data."""
        while True:
            try:
                await self._update_dashboard_data()
                await self._broadcast_updates()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _update_dashboard_data(self):
        """Update dashboard data from platform components."""
        dashboard_data = {}

        # System metrics
        if self.metrics_storage:
            dashboard_data['system'] = {
                'cpu_percent': self.metrics_storage.get_aggregated_metric('system.cpu.percent'),
                'memory_percent': self.metrics_storage.get_aggregated_metric('system.memory.percent'),
                'disk_percent': self.metrics_storage.get_aggregated_metric('system.disk.percent'),
                'network_bytes_sent': self.metrics_storage.get_aggregated_metric('system.network.bytes_sent', aggregator='sum'),
                'network_bytes_recv': self.metrics_storage.get_aggregated_metric('system.network.bytes_recv', aggregator='sum'),
            }

        # Application metrics
        if self.metrics_storage:
            dashboard_data['application'] = {
                'requests_total': self.metrics_storage.get_aggregated_metric('app.requests.total', aggregator='sum'),
                'requests_active': self.metrics_storage.get_aggregated_metric('app.requests.active'),
                'error_rate': self.metrics_storage.get_aggregated_metric('app.error_rate'),
                'avg_response_time': self.metrics_storage.get_aggregated_metric('app.response_time.avg'),
                'throughput_rps': self.metrics_storage.get_aggregated_metric('app.throughput_rps'),
            }

        # Quantum metrics
        if self.metrics_storage:
            dashboard_data['quantum'] = {
                'circuits_executed': self.metrics_storage.get_aggregated_metric('quantum.circuits.executed', aggregator='sum'),
                'error_rate': self.metrics_storage.get_aggregated_metric('quantum.error_rate'),
                'avg_fidelity': self.metrics_storage.get_aggregated_metric('quantum.fidelity.avg'),
                'avg_execution_time': self.metrics_storage.get_aggregated_metric('quantum.execution_time.avg'),
                'circuit_depth_avg': self.metrics_storage.get_aggregated_metric('quantum.circuit_depth.avg'),
                'gate_count_avg': self.metrics_storage.get_aggregated_metric('quantum.gate_count.avg'),
            }

        # Privacy metrics
        if self.metrics_storage:
            dashboard_data['privacy'] = {
                'queries_total': self.metrics_storage.get_aggregated_metric('privacy.queries.total', aggregator='sum'),
                'violations_total': self.metrics_storage.get_aggregated_metric('privacy.violations.total', aggregator='sum'),
                'violation_rate': self.metrics_storage.get_aggregated_metric('privacy.violation_rate'),
                'epsilon_usage_avg': self.metrics_storage.get_aggregated_metric('privacy.epsilon.avg'),
                'epsilon_total': self.metrics_storage.get_aggregated_metric('privacy.epsilon.total', aggregator='sum'),
                'audit_events': self.metrics_storage.get_aggregated_metric('privacy.audit_events', aggregator='sum'),
            }

        # Federated learning metrics
        if self.metrics_storage:
            dashboard_data['federated'] = {
                'rounds_completed': self.metrics_storage.get_aggregated_metric('federated.rounds_completed', aggregator='sum'),
                'clients_participated': self.metrics_storage.get_aggregated_metric('federated.clients_participated', aggregator='sum'),
                'participation_rate': self.metrics_storage.get_aggregated_metric('federated.participation_rate'),
                'communication_cost': self.metrics_storage.get_aggregated_metric('federated.communication_cost', aggregator='sum'),
                'model_convergence': self.metrics_storage.get_aggregated_metric('federated.model_convergence'),
            }

        # Model information
        dashboard_data['models'] = await self._get_model_info()

        # Update timestamp
        dashboard_data['timestamp'] = datetime.now().isoformat()
        dashboard_data['update_interval'] = self.update_interval

        self.dashboard_data = dashboard_data

    async def _get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about deployed models."""
        models = []

        # Mock model information - in practice this would come from model registry
        mock_models = [
            {
                'id': 'efficient_net_edge_v1',
                'name': 'EfficientNet Edge v1',
                'type': 'classification',
                'framework': 'tflite',
                'precision': 'INT8',
                'size_mb': 8.4,
                'accuracy': 0.789,
                'latency_ms': 45.2,
                'deployed_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'status': 'active',
                'usage_count': 1250,
            },
            {
                'id': 'quantum_svm_v2',
                'name': 'Quantum SVM v2',
                'type': 'quantum_classification',
                'framework': 'qiskit',
                'precision': 'quantum',
                'size_mb': 0.1,
                'accuracy': 0.823,
                'latency_ms': 1250.0,
                'deployed_at': (datetime.now() - timedelta(hours=1)).isoformat(),
                'status': 'active',
                'usage_count': 89,
            },
            {
                'id': 'privacy_preserved_model',
                'name': 'Privacy-Preserved Model',
                'type': 'federated_classification',
                'framework': 'pytorch',
                'precision': 'FP16',
                'size_mb': 45.2,
                'accuracy': 0.756,
                'latency_ms': 78.3,
                'deployed_at': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'status': 'training',
                'usage_count': 0,
            }
        ]

        return mock_models

    async def _broadcast_updates(self):
        """Broadcast dashboard updates to WebSocket clients."""
        if self.dashboard_data:
            await self.ws_manager.broadcast('dashboard_update', self.dashboard_data)

    # Web handlers
    @aiohttp_jinja2.template('dashboard.html')
    async def dashboard_handler(self, request):
        """Main dashboard page handler."""
        return {
            'title': 'Quantum Edge AI Platform Dashboard',
            'dashboard_data': self.dashboard_data,
            'timestamp': datetime.now().isoformat()
        }

    async def metrics_api_handler(self, request):
        """API endpoint for metrics data."""
        format = request.query.get('format', 'json')
        duration_hours = int(request.query.get('hours', 24))

        if self.metrics_storage:
            # Get metrics for specified duration
            duration = timedelta(hours=duration_hours)
            cutoff = datetime.now() - duration

            metrics_data = {}
            for series_name, series in self.metrics_storage.series.items():
                points = [p for p in series.points if p.timestamp >= cutoff]
                metrics_data[series_name] = [
                    {
                        'timestamp': p.timestamp.isoformat(),
                        'value': p.value,
                        'tags': p.tags
                    } for p in points
                ]

            if format == 'json':
                return web.json_response(metrics_data)
            else:
                # Export as requested format
                exporter = MetricsExporter(self.metrics_storage)
                if format == 'csv':
                    exporter.export_to_csv('metrics_export.csv')
                    return web.FileResponse('metrics_export.csv')
                elif format == 'prometheus':
                    exporter.export_to_prometheus('metrics.prom')
                    return web.FileResponse('metrics.prom')
        else:
            return web.json_response({'error': 'Metrics storage not available'}, status=503)

    async def models_api_handler(self, request):
        """API endpoint for model information."""
        models = await self._get_model_info()
        return web.json_response({'models': models})

    async def deploy_model_handler(self, request):
        """Handle model deployment requests."""
        try:
            data = await request.json()

            # Validate deployment request
            required_fields = ['model_id', 'target_device', 'config']
            for field in required_fields:
                if field not in data:
                    return web.json_response(
                        {'error': f'Missing required field: {field}'},
                        status=400
                    )

            # Mock deployment process
            deployment_id = f"deploy_{int(time.time())}"

            # In practice, this would initiate actual deployment
            deployment_result = {
                'deployment_id': deployment_id,
                'model_id': data['model_id'],
                'target_device': data['target_device'],
                'status': 'deploying',
                'started_at': datetime.now().isoformat(),
                'config': data['config']
            }

            # Broadcast deployment update
            await self.ws_manager.broadcast('deployment_update', deployment_result)

            return web.json_response(deployment_result, status=202)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def federated_status_handler(self, request):
        """Get federated learning status."""
        if self.federated_server:
            # Mock federated server status
            status = {
                'active_round': 5,
                'total_clients': 12,
                'active_clients': 8,
                'round_progress': 0.75,
                'global_accuracy': 0.842,
                'communication_cost': 245.6,
                'status': 'running'
            }
        else:
            status = {'status': 'not_initialized'}

        return web.json_response(status)

    async def start_federated_handler(self, request):
        """Start federated learning round."""
        try:
            data = await request.json()
            n_clients = data.get('n_clients', 5)
            n_rounds = data.get('n_rounds', 10)

            # Mock federated learning start
            result = {
                'federated_job_id': f"fed_{int(time.time())}",
                'n_clients': n_clients,
                'n_rounds': n_rounds,
                'status': 'starting',
                'started_at': datetime.now().isoformat()
            }

            # Broadcast federated learning start
            await self.ws_manager.broadcast('federated_start', result)

            return web.json_response(result, status=202)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def privacy_budget_handler(self, request):
        """Get privacy budget information."""
        if self.privacy_engine:
            # Mock privacy budget data
            budget_info = {
                'total_budget': 10.0,
                'used_budget': 3.2,
                'remaining_budget': 6.8,
                'budget_utilization': 0.32,
                'queries_today': 145,
                'violations_today': 2,
                'epsilon_history': [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1],
                'last_reset': (datetime.now() - timedelta(days=1)).isoformat()
            }
        else:
            budget_info = {'status': 'not_initialized'}

        return web.json_response(budget_info)

    async def system_status_handler(self, request):
        """Get system status information."""
        status = {
            'uptime': time.time() - getattr(self, '_start_time', time.time()),
            'version': '1.0.0',
            'components': {
                'inference_engine': 'healthy' if self.inference_engine else 'not_initialized',
                'quantum_engine': 'healthy' if self.quantum_engine else 'not_initialized',
                'privacy_engine': 'healthy' if self.privacy_engine else 'not_initialized',
                'federated_server': 'healthy' if self.federated_server else 'not_initialized',
                'metrics_storage': 'healthy' if self.metrics_storage else 'not_initialized'
            },
            'active_connections': len(self.ws_manager.connections),
            'last_update': datetime.now().isoformat()
        }

        return web.json_response(status)

    async def inference_handler(self, request):
        """Handle inference requests."""
        try:
            data = await request.json()

            # Mock inference result
            result = {
                'inference_id': f"inf_{int(time.time())}",
                'model_id': data.get('model_id', 'default'),
                'prediction': [0.1, 0.8, 0.1],  # Mock classification result
                'confidence': 0.8,
                'latency_ms': 45.2,
                'timestamp': datetime.now().isoformat()
            }

            # Broadcast inference result
            await self.ws_manager.broadcast('inference_result', result)

            return web.json_response(result)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def inference_history_handler(self, request):
        """Get inference history."""
        # Mock inference history
        history = [
            {
                'inference_id': f'inf_{i}',
                'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                'model_id': 'efficient_net_edge_v1',
                'latency_ms': 40 + i * 2,
                'success': i % 10 != 0  # 90% success rate
            } for i in range(20)
        ]

        return web.json_response({'history': history})

    async def quantum_circuit_handler(self, request):
        """Handle quantum circuit requests."""
        try:
            data = await request.json()

            # Mock quantum circuit execution
            result = {
                'circuit_id': f"qc_{int(time.time())}",
                'n_qubits': data.get('n_qubits', 4),
                'circuit_depth': data.get('depth', 3),
                'execution_time_ms': 1250.0,
                'fidelity': 0.89,
                'measurements': {'00': 0.52, '01': 0.18, '10': 0.15, '11': 0.15},
                'timestamp': datetime.now().isoformat()
            }

            # Broadcast quantum result
            await self.ws_manager.broadcast('quantum_result', result)

            return web.json_response(result)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def quantum_jobs_handler(self, request):
        """Get quantum job history."""
        # Mock quantum job history
        jobs = [
            {
                'job_id': f'qjob_{i}',
                'timestamp': (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                'circuit_type': 'vqe' if i % 3 == 0 else 'qsvm' if i % 3 == 1 else 'grover',
                'n_qubits': 4 + (i % 3),
                'status': 'completed' if i > 1 else 'running',
                'execution_time_ms': 800 + i * 100,
                'fidelity': 0.85 + (i % 5) * 0.01
            } for i in range(15)
        ]

        return web.json_response({'jobs': jobs})

    async def websocket_handler(self, request):
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Generate client ID
        client_id = f"client_{secrets.token_hex(8)}"

        # Register connection
        await self.ws_manager.register_connection(ws, client_id)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        if data.get('type') == 'subscribe':
                            # Subscribe to topics
                            topics = data.get('topics', [])
                            for topic in topics:
                                await self.ws_manager.subscribe(client_id, topic)

                        elif data.get('type') == 'unsubscribe':
                            # Unsubscribe from topics
                            topics = data.get('topics', [])
                            for topic in topics:
                                await self.ws_manager.unsubscribe(client_id, topic)

                        elif data.get('type') == 'ping':
                            # Respond to ping
                            await ws.send_json({'type': 'pong', 'timestamp': datetime.now().isoformat()})

                    except json.JSONDecodeError:
                        await ws.send_json({'error': 'Invalid JSON'})

                elif msg.type == WSMsgType.ERROR:
                    print(f'WebSocket error for {client_id}: {ws.exception()}')

        finally:
            # Unregister connection
            await self.ws_manager.unregister_connection(client_id)

        return ws


async def create_dashboard_app(host: str = "localhost", port: int = 8080) -> QuantumEdgeDashboard:
    """Create and configure the dashboard application."""
    dashboard = QuantumEdgeDashboard(host, port)

    # Initialize platform components (mock for demonstration)
    dashboard.initialize_platform_components()

    return dashboard


async def main():
    """Main function to run the dashboard."""
    print("🌐 Quantum Edge AI Dashboard")
    print("=" * 35)

    dashboard = await create_dashboard_app()

    try:
        await dashboard.start_dashboard()
    except KeyboardInterrupt:
        print("\\n👋 Dashboard shutdown requested")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Create templates directory and basic template
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)

    dashboard_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% if title %}{{ title }}{% else %}Quantum Edge AI Dashboard{% endif %}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .metric-change {
            font-size: 12px;
            color: #666;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-healthy { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-error { background-color: #F44336; }
        .real-time-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 Quantum Edge AI Platform Dashboard</h1>
            <p>Real-time monitoring and management interface</p>
        </div>

        <div class="real-time-indicator">
            <span class="status-indicator status-healthy"></span>
            Live Data
        </div>

        <!-- System Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">CPU Usage</div>
                <div class="metric-value" id="cpu-percent">--</div>
                <div class="metric-change" id="cpu-trend">Loading...</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value" id="memory-percent">--</div>
                <div class="metric-change" id="memory-trend">Loading...</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Active Requests</div>
                <div class="metric-value" id="active-requests">--</div>
                <div class="metric-change" id="requests-trend">Loading...</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Error Rate</div>
                <div class="metric-value" id="error-rate">--</div>
                <div class="metric-change" id="error-trend">Loading...</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="metrics-grid">
            <div class="chart-container">
                <h3>System Resources Over Time</h3>
                <canvas id="systemChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <h3>Application Performance</h3>
                <canvas id="appChart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Quantum Metrics -->
        <div class="metrics-grid">
            <div class="chart-container">
                <h3>Quantum Circuit Executions</h3>
                <canvas id="quantumChart" width="400" height="200"></canvas>
            </div>
            <div class="chart-container">
                <h3>Privacy Budget Usage</h3>
                <canvas id="privacyChart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Models Table -->
        <div class="chart-container">
            <h3>Deployed Models</h3>
            <table id="models-table" style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Model Name</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Type</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Status</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Accuracy</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Actions</th>
                    </tr>
                </thead>
                <tbody id="models-tbody">
                    <tr>
                        <td colspan="5" style="padding: 20px; text-align: center; color: #666;">Loading models...</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Global state
        let systemChart, appChart, quantumChart, privacyChart;
        let websocket;
        let dashboardData = {};

        // Initialize WebSocket connection
        function initWebSocket() {
            websocket = new WebSocket('ws://' + window.location.host + '/ws');

            websocket.onopen = function(event) {
                console.log('WebSocket connected');
                // Subscribe to dashboard updates
                websocket.send(JSON.stringify({
                    type: 'subscribe',
                    topics: ['dashboard_update', 'inference_result', 'quantum_result']
                }));
            };

            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };

            websocket.onclose = function(event) {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(initWebSocket, 5000);
            };

            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        // Handle WebSocket messages
        function handleWebSocketMessage(data) {
            if (data.topic === 'dashboard_update') {
                dashboardData = data.data;
                updateDashboard();
            } else if (data.topic === 'inference_result') {
                updateInferenceResult(data.data);
            } else if (data.topic === 'quantum_result') {
                updateQuantumResult(data.data);
            }
        }

        // Update dashboard with new data
        function updateDashboard() {
            // Update metrics
            updateMetric('cpu-percent', dashboardData.system?.cpu_percent, '%');
            updateMetric('memory-percent', dashboardData.system?.memory_percent, '%');
            updateMetric('active-requests', dashboardData.application?.requests_active, '');
            updateMetric('error-rate', dashboardData.application?.error_rate, '%');

            // Update charts
            updateCharts();
            updateModelsTable();
        }

        // Update individual metric display
        function updateMetric(elementId, value, unit) {
            const element = document.getElementById(elementId);
            if (element && value !== undefined && value !== null) {
                element.textContent = typeof value === 'number' ?
                    (unit === '%' ? value.toFixed(1) + unit : value.toFixed(0) + unit) :
                    value + unit;
            }
        }

        // Initialize charts
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            };

            // System chart
            const systemCtx = document.getElementById('systemChart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: chartOptions
            });

            // Application chart
            const appCtx = document.getElementById('appChart').getContext('2d');
            appChart = new Chart(appCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Response Time (ms)',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }, {
                        label: 'Throughput (req/s)',
                        data: [],
                        borderColor: 'rgb(255, 205, 86)',
                        tension: 0.1
                    }]
                },
                options: chartOptions
            });

            // Quantum chart
            const quantumCtx = document.getElementById('quantumChart').getContext('2d');
            quantumChart = new Chart(quantumCtx, {
                type: 'bar',
                data: {
                    labels: ['Circuits', 'Errors', 'Fidelity'],
                    datasets: [{
                        label: 'Quantum Metrics',
                        data: [],
                        backgroundColor: [
                            'rgba(153, 102, 255, 0.6)',
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(75, 192, 192, 0.6)'
                        ]
                    }]
                },
                options: chartOptions
            });

            // Privacy chart
            const privacyCtx = document.getElementById('privacyChart').getContext('2d');
            privacyChart = new Chart(privacyCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Used Budget', 'Remaining Budget'],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(75, 192, 192, 0.6)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }

        // Update charts with new data
        function updateCharts() {
            if (!dashboardData) return;

            const timestamp = new Date().toLocaleTimeString();

            // Update system chart
            if (systemChart) {
                systemChart.data.labels.push(timestamp);
                systemChart.data.datasets[0].data.push(dashboardData.system?.cpu_percent || 0);
                systemChart.data.datasets[1].data.push(dashboardData.system?.memory_percent || 0);

                if (systemChart.data.labels.length > 20) {
                    systemChart.data.labels.shift();
                    systemChart.data.datasets[0].data.shift();
                    systemChart.data.datasets[1].data.shift();
                }
                systemChart.update();
            }

            // Update application chart
            if (appChart) {
                appChart.data.labels.push(timestamp);
                appChart.data.datasets[0].data.push(dashboardData.application?.avg_response_time || 0);
                appChart.data.datasets[1].data.push(dashboardData.application?.throughput_rps || 0);

                if (appChart.data.labels.length > 20) {
                    appChart.data.labels.shift();
                    appChart.data.datasets[0].data.shift();
                    appChart.data.datasets[1].data.shift();
                }
                appChart.update();
            }

            // Update quantum chart
            if (quantumChart && dashboardData.quantum) {
                quantumChart.data.datasets[0].data = [
                    dashboardData.quantum.circuits_executed || 0,
                    (dashboardData.quantum.error_rate || 0) * 100,
                    (dashboardData.quantum.avg_fidelity || 0) * 100
                ];
                quantumChart.update();
            }

            // Update privacy chart
            if (privacyChart && dashboardData.privacy) {
                const used = dashboardData.privacy.epsilon_total || 0;
                const remaining = Math.max(0, 10 - used); // Assume total budget of 10
                privacyChart.data.datasets[0].data = [used, remaining];
                privacyChart.update();
            }
        }

        // Update models table
        function updateModelsTable() {
            const tbody = document.getElementById('models-tbody');
            if (!dashboardData.models) return;

            tbody.innerHTML = '';

            dashboardData.models.forEach(model => {
                const row = document.createElement('tr');

                const statusColor = model.status === 'active' ? '#4CAF50' :
                                  model.status === 'training' ? '#FF9800' : '#F44336';

                row.innerHTML = `
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">${model.name}</td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">${model.type}</td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        <span style="color: ${statusColor};">●</span> ${model.status}
                    </td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">${(model.accuracy * 100).toFixed(1)}%</td>
                    <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                        <button onclick="viewModel('${model.id}')" style="margin-right: 5px;">View</button>
                        <button onclick="deployModel('${model.id}')" ${model.status === 'active' ? 'disabled' : ''}>Deploy</button>
                    </td>
                `;

                tbody.appendChild(row);
            });
        }

        // Model action handlers
        function viewModel(modelId) {
            console.log('Viewing model:', modelId);
            // Implement model details view
        }

        function deployModel(modelId) {
            console.log('Deploying model:', modelId);
            // Implement model deployment
        }

        // Inference result handler
        function updateInferenceResult(result) {
            console.log('New inference result:', result);
            // Could show toast notification or update recent inferences list
        }

        // Quantum result handler
        function updateQuantumResult(result) {
            console.log('New quantum result:', result);
            // Could show quantum circuit visualization or update metrics
        }

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            initWebSocket();

            // Initial data fetch
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    console.log('Initial metrics loaded:', data);
                })
                .catch(error => console.error('Error loading initial metrics:', error));
        });

        // Periodic health check
        setInterval(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({type: 'ping'}));
            }
        }, 30000);
    </script>
</body>
</html>'''

    with open(template_dir / "dashboard.html", "w") as f:
        f.write(dashboard_template)

    # Run the dashboard
    asyncio.run(main())

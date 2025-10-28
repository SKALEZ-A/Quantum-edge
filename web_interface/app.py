"""
Quantum Edge AI Platform - Flask Application

Main Flask application for the web dashboard with real-time monitoring,
model management, and system controls.
"""

from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantumEdgeDashboard:
    """Main dashboard application class"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = self._create_app()
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Initialize components
        self.auth_manager = None
        self.websocket_handler = None
        self.monitoring_data = {}
        self.active_clients = set()

        # Setup routes and handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        self._setup_monitoring()

        logger.info("Quantum Edge AI Dashboard initialized")

    def _create_app(self) -> Flask:
        """Create Flask application"""
        app = Flask(__name__,
                   template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                   static_folder=os.path.join(os.path.dirname(__file__), 'static'))

        # Configure app
        app.config.update({
            'SECRET_KEY': self.config.get('secret_key', 'quantum-edge-ai-secret'),
            'SESSION_TYPE': 'filesystem',
            'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,  # 100MB
            'UPLOAD_FOLDER': self.config.get('upload_folder', './uploads'),
            'DEBUG': self.config.get('debug', False)
        })

        # Enable CORS
        CORS(app, resources={
            r"/api/*": {"origins": "*"},
            r"/ws/*": {"origins": "*"}
        })

        # Ensure upload folder exists
        Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

        return app

    def _setup_routes(self):
        """Setup Flask routes"""
        app = self.app

        @app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html',
                                 title="Quantum Edge AI Platform",
                                 version="1.0.0")

        @app.route('/models')
        def models():
            """Model management page"""
            return render_template('models.html',
                                 title="Model Management")

        @app.route('/monitoring')
        def monitoring():
            """System monitoring page"""
            return render_template('monitoring.html',
                                 title="System Monitoring")

        @app.route('/analytics')
        def analytics():
            """Analytics and insights page"""
            return render_template('analytics.html',
                                 title="Analytics & Insights")

        @app.route('/api/health')
        def health_check():
            """API health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'active_clients': len(self.active_clients)
            })

        @app.route('/api/system-info')
        def system_info():
            """Get system information"""
            try:
                import psutil
                import platform

                info = {
                    'platform': platform.platform(),
                    'cpu_count': os.cpu_count(),
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory': {
                        'total': psutil.virtual_memory().total,
                        'available': psutil.virtual_memory().available,
                        'percent': psutil.virtual_memory().percent
                    },
                    'disk': {
                        'total': psutil.disk_usage('/').total,
                        'free': psutil.disk_usage('/').free,
                        'percent': psutil.disk_usage('/').percent
                    },
                    'boot_time': psutil.boot_time(),
                    'python_version': platform.python_version()
                }
                return jsonify(info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/api/models', methods=['GET'])
        def get_models():
            """Get available models"""
            # Mock model data - in real implementation, get from model registry
            models = [
                {
                    'id': 'resnet50_edge',
                    'name': 'ResNet-50 Edge Optimized',
                    'type': 'classification',
                    'status': 'loaded',
                    'accuracy': 0.85,
                    'latency': 45.2,
                    'memory_usage': 256
                },
                {
                    'id': 'quantum_classifier',
                    'name': 'Quantum Support Vector Machine',
                    'type': 'quantum_classification',
                    'status': 'loaded',
                    'accuracy': 0.92,
                    'latency': 120.5,
                    'memory_usage': 512
                }
            ]
            return jsonify(models)

        @app.route('/api/inference', methods=['POST'])
        def run_inference():
            """Run model inference"""
            try:
                data = request.get_json()

                if not data or 'model_id' not in data:
                    return jsonify({'error': 'model_id required'}), 400

                model_id = data['model_id']
                input_data = data.get('input', [])

                # Mock inference result - in real implementation, call actual model
                result = {
                    'model_id': model_id,
                    'prediction': [0.1, 0.9, 0.0],  # Mock probabilities
                    'confidence': 0.85,
                    'latency': 42.3,
                    'timestamp': datetime.utcnow().isoformat()
                }

                return jsonify(result)

            except Exception as e:
                logger.error(f"Inference error: {e}")
                return jsonify({'error': str(e)}), 500

        @app.route('/api/monitoring/metrics')
        def get_metrics():
            """Get monitoring metrics"""
            return jsonify(self.monitoring_data)

        @app.route('/api/upload', methods=['POST'])
        def upload_file():
            """Handle file uploads"""
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            try:
                # Save file
                upload_path = Path(app.config['UPLOAD_FOLDER']) / file.filename
                file.save(upload_path)

                return jsonify({
                    'filename': file.filename,
                    'size': upload_path.stat().st_size,
                    'path': str(upload_path),
                    'uploaded_at': datetime.utcnow().isoformat()
                })

            except Exception as e:
                logger.error(f"Upload error: {e}")
                return jsonify({'error': str(e)}), 500

    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        socketio = self.socketio

        @socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.active_clients.add(request.sid)
            logger.info(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to Quantum Edge AI Dashboard'})

        @socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.active_clients.discard(request.sid)
            logger.info(f"Client disconnected: {request.sid}")

        @socketio.on('subscribe_metrics')
        def handle_subscribe_metrics(data):
            """Subscribe to real-time metrics"""
            join_room('metrics')
            emit('metrics_subscribed', {'status': 'subscribed'})

        @socketio.on('unsubscribe_metrics')
        def handle_unsubscribe_metrics():
            """Unsubscribe from real-time metrics"""
            leave_room('metrics')
            emit('metrics_unsubscribed', {'status': 'unsubscribed'})

        @socketio.on('request_model_status')
        def handle_model_status_request():
            """Send current model status"""
            # Mock model status - in real implementation, get from model manager
            model_status = {
                'models': [
                    {'id': 'resnet50', 'status': 'active', 'requests_per_second': 15.2},
                    {'id': 'quantum_svm', 'status': 'active', 'requests_per_second': 8.7}
                ],
                'timestamp': datetime.utcnow().isoformat()
            }
            emit('model_status', model_status)

    def _setup_monitoring(self):
        """Setup background monitoring"""
        def monitoring_loop():
            """Background monitoring loop"""
            while True:
                try:
                    # Collect system metrics
                    import psutil

                    self.monitoring_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'cpu_percent': psutil.cpu_percent(interval=0.1),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_percent': psutil.disk_usage('/').percent,
                        'network_connections': len(psutil.net_connections()),
                        'active_clients': len(self.active_clients)
                    }

                    # Send to subscribed clients
                    self.socketio.emit('metrics_update', self.monitoring_data, room='metrics')

                    time.sleep(2)  # Update every 2 seconds

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(5)

        # Start monitoring thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)

def create_app(config: Dict[str, Any] = None) -> Flask:
    """Create Flask application instance"""
    if config is None:
        config = {}

    dashboard = QuantumEdgeDashboard(config)
    return dashboard.app

def create_socketio_app(config: Dict[str, Any] = None):
    """Create SocketIO application instance"""
    if config is None:
        config = {}

    dashboard = QuantumEdgeDashboard(config)
    return dashboard.socketio

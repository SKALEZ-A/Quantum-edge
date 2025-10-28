"""
Quantum Edge AI Platform - REST API Service

Comprehensive REST API service for quantum edge AI operations,
including model inference, training, monitoring, and management endpoints.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
from functools import wraps
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
import re

# Third-party imports (would be installed in production)
try:
    from flask import Flask, request, jsonify, Blueprint, g
    from flask_cors import CORS
    from werkzeug.exceptions import BadRequest, Unauthorized, Forbidden, NotFound
    import jwt
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for development without dependencies
    Flask = Blueprint = None
    CORS = BadRequest = Unauthorized = Forbidden = NotFound = None
    jwt = None
    BaseModel = Field = validator = None

from ..edge_runtime.inference_engine import EdgeInferenceEngine, ModelSpec, Precision
from ..quantum_algorithms.quantum_ml import QuantumMachineLearning
from ..federated_learning.federated_server import FederatedLearningServer
from ..privacy_security.encryption import QuantumEncryption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIRequest:
    """API request model"""
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[bytes]
    query_params: Dict[str, List[str]]
    timestamp: datetime
    client_ip: str
    user_agent: str
    request_id: str

@dataclass
class APIResponse:
    """API response model"""
    status_code: int
    headers: Dict[str, str]
    body: Optional[bytes]
    timestamp: datetime
    processing_time: float
    request_id: str

@dataclass
class UserSession:
    """User session information"""
    user_id: str
    session_id: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str

@dataclass
class APIMetrics:
    """API performance metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    requests_per_minute: float
    error_rate: float
    peak_concurrent_requests: int
    memory_usage_mb: float
    cpu_usage_percent: float

class RequestValidator:
    """Request validation and sanitization"""

    @staticmethod
    def validate_json_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON payload against schema"""
        validated_data = {}

        for field, field_schema in schema.items():
            if field not in payload and field_schema.get('required', False):
                raise BadRequest(f"Missing required field: {field}")

            if field in payload:
                value = payload[field]
                field_type = field_schema.get('type')

                # Type validation
                if field_type == 'string' and not isinstance(value, str):
                    raise BadRequest(f"Field {field} must be string")
                elif field_type == 'number' and not isinstance(value, (int, float)):
                    raise BadRequest(f"Field {field} must be number")
                elif field_type == 'boolean' and not isinstance(value, bool):
                    raise BadRequest(f"Field {field} must be boolean")
                elif field_type == 'array' and not isinstance(value, list):
                    raise BadRequest(f"Field {field} must be array")

                # Range validation
                if 'min' in field_schema and value < field_schema['min']:
                    raise BadRequest(f"Field {field} must be >= {field_schema['min']}")
                if 'max' in field_schema and value > field_schema['max']:
                    raise BadRequest(f"Field {field} must be <= {field_schema['max']}")

                # Custom validation
                if 'pattern' in field_schema:
                    if not re.match(field_schema['pattern'], str(value)):
                        raise BadRequest(f"Field {field} does not match required pattern")

                validated_data[field] = value
            elif 'default' in field_schema:
                validated_data[field] = field_schema['default']

        return validated_data

    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            return str(text)

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>]', '', text)
        return sanitized.strip()

class AuthenticationManager:
    """JWT-based authentication manager"""

    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.sessions: Dict[str, UserSession] = {}
        self.user_permissions: Dict[str, Dict[str, Any]] = {}

    def create_token(self, user_id: str, roles: List[str] = None,
                    permissions: List[str] = None, expires_in: int = 3600) -> str:
        """Create JWT token for user"""
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

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            if datetime.utcfromtimestamp(payload['exp']) < datetime.utcnow():
                return None

            return payload
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> UserSession:
        """Create user session"""
        session_id = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()[:32]

        session = UserSession(
            user_id=user_id,
            session_id=session_id,
            roles=self.user_permissions.get(user_id, {}).get('roles', []),
            permissions=self.user_permissions.get(user_id, {}).get('permissions', []),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        return session

    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate user session"""
        session = self.sessions.get(session_id)
        if session and session.expires_at > datetime.utcnow():
            return session
        return None

class RateLimiter:
    """API rate limiting"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)

        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []

            # Remove old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff
            ]

            if len(self.requests[client_id]) < self.requests_per_minute:
                self.requests[client_id].append(now)
                return True

            return False

class QuantumEdgeAPI:
    """Main REST API service for Quantum Edge AI Platform"""

    def __init__(self, host: str = '0.0.0.0', port: int = 8080,
                 enable_auth: bool = True, enable_rate_limiting: bool = True):
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.enable_rate_limiting = enable_rate_limiting

        # Initialize Flask app
        if Flask:
            self.app = Flask(__name__)
            CORS(self.app)
        else:
            self.app = None

        # Initialize components
        self.auth_manager = AuthenticationManager(
            secret_key='quantum-edge-secret-key-change-in-production'
        )
        self.rate_limiter = RateLimiter(requests_per_minute=100)
        self.validator = RequestValidator()

        # Initialize ML components
        self.inference_engine = None
        self.quantum_ml = QuantumMachineLearning()
        self.federated_server = None
        self.encryption = QuantumEncryption()

        # Metrics and monitoring
        self.metrics = APIMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        self.request_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Request processing
        self._setup_routes()
        self._start_metrics_collector()

    def _setup_routes(self):
        """Setup API routes"""
        if not self.app:
            return

        # Health check
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            })

        # Authentication routes
        @self.app.route('/auth/login', methods=['POST'])
        def login():
            return self._handle_auth_login()

        @self.app.route('/auth/verify', methods=['GET'])
        def verify_token():
            return self._handle_auth_verify()

        # Inference routes
        @self.app.route('/inference/predict', methods=['POST'])
        @self._require_auth
        @self._rate_limit
        def predict():
            return self._handle_inference_predict()

        @self.app.route('/inference/batch', methods=['POST'])
        @self._require_auth
        @self._rate_limit
        def batch_predict():
            return self._handle_inference_batch()

        # Quantum ML routes
        @self.app.route('/quantum/train', methods=['POST'])
        @self._require_auth
        @self._rate_limit
        def train_quantum_model():
            return self._handle_quantum_train()

        @self.app.route('/quantum/predict', methods=['POST'])
        @self._require_auth
        @self._rate_limit
        def quantum_predict():
            return self._handle_quantum_predict()

        # Federated Learning routes
        @self.app.route('/federated/register', methods=['POST'])
        @self._require_auth
        def federated_register():
            return self._handle_federated_register()

        @self.app.route('/federated/update', methods=['POST'])
        @self._require_auth
        @self._rate_limit
        def federated_update():
            return self._handle_federated_update()

        # Model Management routes
        @self.app.route('/models', methods=['GET'])
        @self._require_auth
        def list_models():
            return self._handle_list_models()

        @self.app.route('/models/<model_id>', methods=['GET'])
        @self._require_auth
        def get_model(model_id):
            return self._handle_get_model(model_id)

        @self.app.route('/models', methods=['POST'])
        @self._require_auth
        def create_model():
            return self._handle_create_model()

        # Monitoring routes
        @self.app.route('/metrics', methods=['GET'])
        @self._require_auth
        def get_metrics():
            return self._handle_get_metrics()

        @self.app.route('/logs', methods=['GET'])
        @self._require_auth
        def get_logs():
            return self._handle_get_logs()

    def _require_auth(self, f):
        """Decorator for authentication requirement"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.enable_auth:
                return f(*args, **kwargs)

            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing or invalid authorization header'}), 401

            token = auth_header.split(' ')[1]
            payload = self.auth_manager.verify_token(token)

            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401

            g.user_id = payload['user_id']
            g.roles = payload.get('roles', [])
            g.permissions = payload.get('permissions', [])
            return f(*args, **kwargs)
        return decorated_function

    def _rate_limit(self, f):
        """Decorator for rate limiting"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.enable_rate_limiting:
                return f(*args, **kwargs)

            client_id = request.remote_addr
            if not self.rate_limiter.is_allowed(client_id):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            return f(*args, **kwargs)
        return decorated_function

    def _handle_auth_login(self) -> tuple:
        """Handle user login"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'username': {'type': 'string', 'required': True},
                'password': {'type': 'string', 'required': True}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            # Simplified authentication (in production, verify against database)
            if validated_data['username'] == 'admin' and validated_data['password'] == 'password':
                user_id = 'admin'
                roles = ['admin', 'user']
                permissions = ['read', 'write', 'delete', 'inference', 'training']

                token = self.auth_manager.create_token(user_id, roles, permissions)

                # Create session
                session = self.auth_manager.create_session(
                    user_id, request.remote_addr, request.headers.get('User-Agent', '')
                )

                return jsonify({
                    'token': token,
                    'session_id': session.session_id,
                    'user_id': user_id,
                    'roles': roles,
                    'permissions': permissions,
                    'expires_in': 3600
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500

    def _handle_auth_verify(self) -> tuple:
        """Handle token verification"""
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing token'}), 400

            token = auth_header.split(' ')[1]
            payload = self.auth_manager.verify_token(token)

            if payload:
                return jsonify({
                    'valid': True,
                    'user_id': payload['user_id'],
                    'roles': payload.get('roles', []),
                    'permissions': payload.get('permissions', [])
                }), 200
            else:
                return jsonify({'valid': False, 'error': 'Invalid token'}), 401

        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500

    def _handle_inference_predict(self) -> tuple:
        """Handle single inference request"""
        try:
            if not self.inference_engine:
                return jsonify({'error': 'Inference engine not initialized'}), 503

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'input_data': {'type': 'array', 'required': True},
                'model_id': {'type': 'string', 'required': False, 'default': 'default'},
                'precision': {'type': 'string', 'required': False, 'default': 'fp32'}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            # Convert input to numpy array
            input_array = np.array(validated_data['input_data'])

            # Perform inference
            result = self.inference_engine.predict(input_array)

            return jsonify({
                'prediction': result.output.tolist(),
                'confidence': result.confidence,
                'latency_ms': result.latency,
                'precision_used': result.precision_used.value,
                'memory_used_mb': result.memory_used,
                'power_consumed_mw': result.power_consumed,
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return jsonify({'error': 'Inference failed'}), 500

    def _handle_inference_batch(self) -> tuple:
        """Handle batch inference request"""
        try:
            if not self.inference_engine:
                return jsonify({'error': 'Inference engine not initialized'}), 503

            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'input_batch': {'type': 'array', 'required': True},
                'batch_size': {'type': 'number', 'required': False, 'default': 8, 'min': 1, 'max': 100}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            # Convert batch to numpy array
            batch_array = np.array(validated_data['input_batch'])
            batch_size = validated_data['batch_size']

            # Perform batch inference
            results = self.inference_engine.batch_predict(batch_array, batch_size)

            # Convert results to JSON-serializable format
            predictions = []
            for result in results:
                predictions.append({
                    'prediction': result.output.tolist(),
                    'confidence': result.confidence,
                    'latency_ms': result.latency,
                    'precision_used': result.precision_used.value,
                    'memory_used_mb': result.memory_used,
                    'power_consumed_mw': result.power_consumed
                })

            return jsonify({
                'predictions': predictions,
                'batch_size': len(predictions),
                'total_latency_ms': sum(r.latency for r in results),
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error(f"Batch inference error: {str(e)}")
            return jsonify({'error': 'Batch inference failed'}), 500

    def _handle_quantum_train(self) -> tuple:
        """Handle quantum model training"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'model_type': {'type': 'string', 'required': True, 'pattern': '^(qsvm|qnn)$'},
                'training_data': {'type': 'array', 'required': True},
                'labels': {'type': 'array', 'required': True},
                'hyperparameters': {'type': 'object', 'required': False, 'default': {}}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            X = np.array(validated_data['training_data'])
            y = np.array(validated_data['labels'])
            model_type = validated_data['model_type']
            hyperparameters = validated_data['hyperparameters']

            # Train quantum model
            if model_type == 'qsvm':
                model = self.quantum_ml.create_qsvm(**hyperparameters)
            elif model_type == 'qnn':
                model = self.quantum_ml.create_qnn(**hyperparameters)

            model.fit(X, y)

            return jsonify({
                'model_id': f"{model_type}_{hash(str(X.tobytes()) + str(y.tobytes())):x}",
                'model_type': model_type,
                'training_samples': len(X),
                'features': X.shape[1],
                'hyperparameters': hyperparameters,
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error(f"Quantum training error: {str(e)}")
            return jsonify({'error': 'Training failed'}), 500

    def _handle_quantum_predict(self) -> tuple:
        """Handle quantum model prediction"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'model_id': {'type': 'string', 'required': True},
                'input_data': {'type': 'array', 'required': True}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            # For now, create a new model (in production, load from storage)
            model = self.quantum_ml.create_qsvm()
            input_data = np.array(validated_data['input_data'])

            predictions = model.predict(input_data)

            return jsonify({
                'predictions': predictions.tolist(),
                'model_id': validated_data['model_id'],
                'input_samples': len(input_data),
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error(f"Quantum prediction error: {str(e)}")
            return jsonify({'error': 'Prediction failed'}), 500

    def _handle_federated_register(self) -> tuple:
        """Handle federated learning client registration"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'client_id': {'type': 'string', 'required': True},
                'device_info': {'type': 'object', 'required': False, 'default': {}}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            if not self.federated_server:
                self.federated_server = FederatedLearningServer()

            client_id = self.federated_server.register_client(
                validated_data['client_id'],
                validated_data['device_info']
            )

            return jsonify({
                'client_id': client_id,
                'status': 'registered',
                'global_model_version': 1,
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error(f"Federated registration error: {str(e)}")
            return jsonify({'error': 'Registration failed'}), 500

    def _handle_federated_update(self) -> tuple:
        """Handle federated learning model update"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'client_id': {'type': 'string', 'required': True},
                'model_update': {'type': 'object', 'required': True},
                'local_samples': {'type': 'number', 'required': True, 'min': 1}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            if not self.federated_server:
                return jsonify({'error': 'Federated server not initialized'}), 503

            # Process federated update
            update_result = self.federated_server.process_update(
                validated_data['client_id'],
                validated_data['model_update'],
                validated_data['local_samples']
            )

            return jsonify({
                'client_id': validated_data['client_id'],
                'update_processed': True,
                'global_model_version': update_result.get('version', 1),
                'contribution_score': update_result.get('contribution', 0.0),
                'timestamp': datetime.utcnow().isoformat()
            }), 200

        except Exception as e:
            logger.error(f"Federated update error: {str(e)}")
            return jsonify({'error': 'Update failed'}), 500

    def _handle_list_models(self) -> tuple:
        """Handle list models request"""
        # Simplified model listing
        models = [
            {
                'id': 'quantum_svm_v1',
                'type': 'qsvm',
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active'
            },
            {
                'id': 'quantum_nn_v1',
                'type': 'qnn',
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active'
            }
        ]

        return jsonify({
            'models': models,
            'total': len(models),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    def _handle_get_model(self, model_id: str) -> tuple:
        """Handle get model request"""
        # Simplified model retrieval
        model_info = {
            'id': model_id,
            'type': 'qsvm',
            'created_at': datetime.utcnow().isoformat(),
            'status': 'active',
            'parameters': {
                'num_qubits': 4,
                'layers': 2,
                'C': 1.0
            }
        }

        return jsonify(model_info), 200

    def _handle_create_model(self) -> tuple:
        """Handle create model request"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing JSON payload'}), 400

            schema = {
                'name': {'type': 'string', 'required': True},
                'type': {'type': 'string', 'required': True, 'pattern': '^(qsvm|qnn)$'},
                'parameters': {'type': 'object', 'required': False, 'default': {}}
            }

            validated_data = self.validator.validate_json_payload(data, schema)

            model_id = f"{validated_data['type']}_{hash(validated_data['name'] + str(time.time())):x}"

            return jsonify({
                'model_id': model_id,
                'name': validated_data['name'],
                'type': validated_data['type'],
                'parameters': validated_data['parameters'],
                'status': 'created',
                'created_at': datetime.utcnow().isoformat()
            }), 201

        except Exception as e:
            logger.error(f"Model creation error: {str(e)}")
            return jsonify({'error': 'Model creation failed'}), 500

    def _handle_get_metrics(self) -> tuple:
        """Handle get metrics request"""
        return jsonify(asdict(self.metrics)), 200

    def _handle_get_logs(self) -> tuple:
        """Handle get logs request"""
        # Simplified log retrieval
        logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'INFO',
                'message': 'API server started',
                'module': 'rest_api'
            }
        ]

        return jsonify({
            'logs': logs,
            'total': len(logs),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    def _start_metrics_collector(self):
        """Start background metrics collection"""
        def collect_metrics():
            while True:
                try:
                    # Update metrics
                    self.metrics.memory_usage_mb = 150.0  # Placeholder
                    self.metrics.cpu_usage_percent = 25.0  # Placeholder
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Metrics collection error: {str(e)}")

        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()

    def initialize_inference_engine(self, model_spec: ModelSpec):
        """Initialize the inference engine"""
        self.inference_engine = EdgeInferenceEngine(model_spec)

    def run(self):
        """Run the API server"""
        if self.app:
            logger.info(f"Starting Quantum Edge API server on {self.host}:{self.port}")
            self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
        else:
            logger.error("Flask not available. Install flask and flask-cors to run the API server.")

if __name__ == '__main__':
    # Create and run API server
    api = QuantumEdgeAPI()

    # Initialize with a sample model
    model_spec = ModelSpec(
        input_shape=(784,),
        output_shape=(10,),
        num_parameters=100000,
        model_size_mb=50.0,
        supported_precisions=[Precision.FP32, Precision.FP16, Precision.INT8],
        target_latency_ms=10.0,
        power_budget_mw=500.0
    )

    api.initialize_inference_engine(model_spec)
    api.run()

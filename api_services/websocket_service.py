"""
Quantum Edge AI Platform - WebSocket Service

Real-time WebSocket service for quantum edge AI operations including live inference,
federated learning updates, quantum state monitoring, and collaborative features.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta
import threading
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import base64
import re

# Third-party imports (would be installed in production)
try:
    import websockets
    from websockets.exceptions import ConnectionClosedError, WebSocketException
    import jwt
    import redis
except ImportError:
    # Fallback for development without dependencies
    websockets = ConnectionClosedError = WebSocketException = None
    jwt = redis = None

import numpy as np
from .rest_api import AuthenticationManager, UserSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    AUTHENTICATE = "authenticate"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    INFERENCE_REQUEST = "inference_request"
    BATCH_INFERENCE_REQUEST = "batch_inference_request"
    TRAINING_REQUEST = "training_request"
    FEDERATED_UPDATE = "federated_update"
    QUANTUM_STATE_UPDATE = "quantum_state_update"
    METRICS_UPDATE = "metrics_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"

class SubscriptionType(Enum):
    """Subscription types"""
    INFERENCE_RESULTS = "inference_results"
    METRICS = "metrics"
    FEDERATED_UPDATES = "federated_updates"
    QUANTUM_STATES = "quantum_states"
    MODEL_UPDATES = "model_updates"
    SYSTEM_EVENTS = "system_events"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    id: str
    payload: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert message to JSON"""
        data = {
            'type': self.type.value,
            'id': self.id,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat()
        }
        if self.user_id:
            data['user_id'] = self.user_id
        if self.session_id:
            data['session_id'] = self.session_id
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON"""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data['type']),
            id=data['id'],
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id')
        )

@dataclass
class ConnectedClient:
    """Connected WebSocket client information"""
    websocket: Any  # WebSocketServerProtocol
    user_id: str
    session_id: str
    subscriptions: Set[SubscriptionType]
    connected_at: datetime
    last_heartbeat: datetime
    ip_address: str
    user_agent: str

class WebSocketRateLimiter:
    """Rate limiter for WebSocket connections"""

    def __init__(self, messages_per_minute: int = 120):
        self.messages_per_minute = messages_per_minute
        self.client_messages: Dict[str, List[datetime]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if message is allowed for client"""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)

        with self.lock:
            if client_id not in self.client_messages:
                self.client_messages[client_id] = []

            # Remove old messages
            self.client_messages[client_id] = [
                msg_time for msg_time in self.client_messages[client_id]
                if msg_time > cutoff
            ]

            if len(self.client_messages[client_id]) < self.messages_per_minute:
                self.client_messages[client_id].append(now)
                return True

            return False

class QuantumWebSocketService:
    """Main WebSocket service for real-time quantum edge AI operations"""

    def __init__(self, host: str = '0.0.0.0', port: int = 8081,
                 enable_auth: bool = True, enable_rate_limiting: bool = True,
                 max_connections: int = 1000):
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.enable_rate_limiting = enable_rate_limiting
        self.max_connections = max_connections

        # Initialize components
        self.auth_manager = AuthenticationManager(
            secret_key='quantum-edge-websocket-secret-key-change-in-production'
        )
        self.rate_limiter = WebSocketRateLimiter(messages_per_minute=120)

        # Client management
        self.connected_clients: Dict[str, ConnectedClient] = {}
        self.client_lock = threading.Lock()

        # ML components
        self.inference_engine = None
        self.quantum_ml = None
        self.federated_server = None

        # Background tasks
        self.broadcast_executor = ThreadPoolExecutor(max_workers=4)
        self.monitoring_thread = None
        self.running = False

        # Redis for pub/sub (optional)
        self.redis_client = None
        if redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            except:
                logger.warning("Redis not available, using in-memory pub/sub")

        # In-memory pub/sub for subscriptions
        self.subscribers: Dict[SubscriptionType, Set[str]] = {
            sub_type: set() for sub_type in SubscriptionType
        }

    async def handle_connection(self, websocket: Any, path: str):
        """Handle WebSocket connection"""
        client_id = str(uuid.uuid4())
        client = None

        try:
            # Extract client information
            client_info = await self._extract_client_info(websocket)

            logger.info(f"New WebSocket connection: {client_id} from {client_info['ip_address']}")

            # Authentication handshake
            if self.enable_auth:
                auth_success = await self._handle_authentication(websocket, client_id)
                if not auth_success:
                    await websocket.close(1008, "Authentication failed")
                    return

            # Create connected client
            with self.client_lock:
                client = ConnectedClient(
                    websocket=websocket,
                    user_id=client_info.get('user_id', 'anonymous'),
                    session_id=client_info.get('session_id', client_id),
                    subscriptions=set(),
                    connected_at=datetime.utcnow(),
                    last_heartbeat=datetime.utcnow(),
                    ip_address=client_info['ip_address'],
                    user_agent=client_info.get('user_agent', 'Unknown')
                )
                self.connected_clients[client_id] = client

            # Send welcome message
            welcome_msg = WebSocketMessage(
                type=MessageType.NOTIFICATION,
                id=str(uuid.uuid4()),
                payload={
                    'message': 'Connected to Quantum Edge AI WebSocket',
                    'client_id': client_id,
                    'supported_message_types': [mt.value for mt in MessageType],
                    'supported_subscriptions': [st.value for st in SubscriptionType]
                },
                timestamp=datetime.utcnow(),
                user_id=client.user_id,
                session_id=client.session_id
            )
            await websocket.send(welcome_msg.to_json())

            # Main message handling loop
            async for message in websocket:
                try:
                    # Rate limiting check
                    if self.enable_rate_limiting and not self.rate_limiter.is_allowed(client_id):
                        await self._send_error(websocket, "Rate limit exceeded", client_id)
                        continue

                    # Parse and handle message
                    ws_message = WebSocketMessage.from_json(message)
                    await self._handle_message(websocket, ws_message, client_id)

                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message", client_id)
                except Exception as e:
                    logger.error(f"Message handling error for client {client_id}: {str(e)}")
                    await self._send_error(websocket, f"Message processing error: {str(e)}", client_id)

        except WebSocketException as e:
            logger.warning(f"WebSocket error for client {client_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Connection handling error for client {client_id}: {str(e)}")
        finally:
            # Cleanup
            with self.client_lock:
                if client_id in self.connected_clients:
                    client = self.connected_clients[client_id]
                    # Remove from all subscriptions
                    for sub_type in client.subscriptions:
                        self.subscribers[sub_type].discard(client_id)
                    del self.connected_clients[client_id]

            logger.info(f"WebSocket connection closed: {client_id}")

    async def _extract_client_info(self, websocket: Any) -> Dict[str, Any]:
        """Extract client information from WebSocket connection"""
        # In a real implementation, you'd extract from headers/query params
        return {
            'ip_address': getattr(websocket, 'remote_address', ('unknown', 0))[0],
            'user_agent': 'WebSocket Client',
            'user_id': None,
            'session_id': None
        }

    async def _handle_authentication(self, websocket: Any, client_id: str) -> bool:
        """Handle WebSocket authentication"""
        try:
            # Wait for authentication message with timeout
            auth_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=10.0
            )

            ws_message = WebSocketMessage.from_json(auth_message)

            if ws_message.type != MessageType.AUTHENTICATE:
                await self._send_error(websocket, "Expected authentication message", client_id)
                return False

            token = ws_message.payload.get('token')
            if not token:
                await self._send_error(websocket, "Missing authentication token", client_id)
                return False

            # Verify token
            payload = self.auth_manager.verify_token(token)
            if not payload:
                await self._send_error(websocket, "Invalid authentication token", client_id)
                return False

            # Send authentication success
            success_msg = WebSocketMessage(
                type=MessageType.NOTIFICATION,
                id=str(uuid.uuid4()),
                payload={
                    'message': 'Authentication successful',
                    'user_id': payload['user_id'],
                    'roles': payload.get('roles', []),
                    'permissions': payload.get('permissions', [])
                },
                timestamp=datetime.utcnow()
            )
            await websocket.send(success_msg.to_json())

            # Store user info for later use
            websocket._user_id = payload['user_id']
            websocket._roles = payload.get('roles', [])
            websocket._permissions = payload.get('permissions', [])

            return True

        except asyncio.TimeoutError:
            await self._send_error(websocket, "Authentication timeout", client_id)
            return False
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            await self._send_error(websocket, "Authentication failed", client_id)
            return False

    async def _handle_message(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle incoming WebSocket message"""
        try:
            # Update client heartbeat
            with self.client_lock:
                if client_id in self.connected_clients:
                    self.connected_clients[client_id].last_heartbeat = datetime.utcnow()

            # Route message based on type
            if message.type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(websocket, message, client_id)
            elif message.type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(websocket, message, client_id)
            elif message.type == MessageType.INFERENCE_REQUEST:
                await self._handle_inference_request(websocket, message, client_id)
            elif message.type == MessageType.BATCH_INFERENCE_REQUEST:
                await self._handle_batch_inference_request(websocket, message, client_id)
            elif message.type == MessageType.TRAINING_REQUEST:
                await self._handle_training_request(websocket, message, client_id)
            elif message.type == MessageType.FEDERATED_UPDATE:
                await self._handle_federated_update(websocket, message, client_id)
            elif message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(websocket, message, client_id)
            else:
                await self._send_error(websocket, f"Unknown message type: {message.type.value}", client_id)

        except Exception as e:
            logger.error(f"Message handling error: {str(e)}")
            await self._send_error(websocket, f"Message processing failed: {str(e)}", client_id)

    async def _handle_subscribe(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle subscription request"""
        try:
            subscription_type_str = message.payload.get('subscription_type')
            if not subscription_type_str:
                await self._send_error(websocket, "Missing subscription_type", client_id)
                return

            try:
                subscription_type = SubscriptionType(subscription_type_str)
            except ValueError:
                await self._send_error(websocket, f"Invalid subscription type: {subscription_type_str}", client_id)
                return

            # Add subscription
            with self.client_lock:
                if client_id in self.connected_clients:
                    self.connected_clients[client_id].subscriptions.add(subscription_type)
                    self.subscribers[subscription_type].add(client_id)

            # Send confirmation
            confirm_msg = WebSocketMessage(
                type=MessageType.NOTIFICATION,
                id=str(uuid.uuid4()),
                payload={
                    'message': f'Subscribed to {subscription_type.value}',
                    'subscription_type': subscription_type.value
                },
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )
            await websocket.send(confirm_msg.to_json())

            logger.info(f"Client {client_id} subscribed to {subscription_type.value}")

        except Exception as e:
            logger.error(f"Subscription error: {str(e)}")
            await self._send_error(websocket, "Subscription failed", client_id)

    async def _handle_unsubscribe(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle unsubscription request"""
        try:
            subscription_type_str = message.payload.get('subscription_type')
            if not subscription_type_str:
                await self._send_error(websocket, "Missing subscription_type", client_id)
                return

            try:
                subscription_type = SubscriptionType(subscription_type_str)
            except ValueError:
                await self._send_error(websocket, f"Invalid subscription type: {subscription_type_str}", client_id)
                return

            # Remove subscription
            with self.client_lock:
                if client_id in self.connected_clients:
                    self.connected_clients[client_id].subscriptions.discard(subscription_type)
                    self.subscribers[subscription_type].discard(client_id)

            # Send confirmation
            confirm_msg = WebSocketMessage(
                type=MessageType.NOTIFICATION,
                id=str(uuid.uuid4()),
                payload={
                    'message': f'Unsubscribed from {subscription_type.value}',
                    'subscription_type': subscription_type.value
                },
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )
            await websocket.send(confirm_msg.to_json())

        except Exception as e:
            logger.error(f"Unsubscription error: {str(e)}")
            await self._send_error(websocket, "Unsubscription failed", client_id)

    async def _handle_inference_request(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle real-time inference request"""
        try:
            if not self.inference_engine:
                await self._send_error(websocket, "Inference engine not available", client_id)
                return

            input_data = message.payload.get('input_data')
            model_id = message.payload.get('model_id', 'default')
            precision = message.payload.get('precision', 'fp32')

            if not input_data:
                await self._send_error(websocket, "Missing input_data", client_id)
                return

            # Convert to numpy array
            input_array = np.array(input_data)

            # Perform inference
            result = self.inference_engine.predict(input_array)

            # Send result
            result_msg = WebSocketMessage(
                type=MessageType.INFERENCE_REQUEST,
                id=message.id,  # Use same ID for correlation
                payload={
                    'prediction': result.output.tolist(),
                    'confidence': result.confidence,
                    'latency_ms': result.latency,
                    'precision_used': result.precision_used.value,
                    'memory_used_mb': result.memory_used,
                    'power_consumed_mw': result.power_consumed,
                    'model_id': model_id
                },
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )
            await websocket.send(result_msg.to_json())

            # Broadcast to subscribers
            await self._broadcast_to_subscribers(
                SubscriptionType.INFERENCE_RESULTS,
                result_msg.payload,
                exclude_client=client_id
            )

        except Exception as e:
            logger.error(f"Inference request error: {str(e)}")
            await self._send_error(websocket, f"Inference failed: {str(e)}", client_id)

    async def _handle_batch_inference_request(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle batch inference request"""
        try:
            if not self.inference_engine:
                await self._send_error(websocket, "Inference engine not available", client_id)
                return

            input_batch = message.payload.get('input_batch')
            batch_size = message.payload.get('batch_size', 8)
            model_id = message.payload.get('model_id', 'default')

            if not input_batch:
                await self._send_error(websocket, "Missing input_batch", client_id)
                return

            # Convert to numpy array
            batch_array = np.array(input_batch)

            # Perform batch inference
            results = self.inference_engine.batch_predict(batch_array, batch_size)

            # Format results
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

            # Send result
            result_msg = WebSocketMessage(
                type=MessageType.BATCH_INFERENCE_REQUEST,
                id=message.id,
                payload={
                    'predictions': predictions,
                    'batch_size': len(predictions),
                    'total_latency_ms': sum(r.latency for r in results),
                    'model_id': model_id
                },
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )
            await websocket.send(result_msg.to_json())

        except Exception as e:
            logger.error(f"Batch inference request error: {str(e)}")
            await self._send_error(websocket, f"Batch inference failed: {str(e)}", client_id)

    async def _handle_training_request(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle training request"""
        try:
            if not self.quantum_ml:
                await self._send_error(websocket, "Quantum ML not available", client_id)
                return

            model_type = message.payload.get('model_type')
            training_data = message.payload.get('training_data')
            labels = message.payload.get('labels')
            hyperparameters = message.payload.get('hyperparameters', {})

            if not all([model_type, training_data, labels]):
                await self._send_error(websocket, "Missing required training parameters", client_id)
                return

            # Convert data
            X = np.array(training_data)
            y = np.array(labels)

            # Train model (this should be async in production)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.broadcast_executor,
                self._train_model_sync,
                model_type, X, y, hyperparameters, websocket, message, client_id
            )

        except Exception as e:
            logger.error(f"Training request error: {str(e)}")
            await self._send_error(websocket, f"Training failed: {str(e)}", client_id)

    def _train_model_sync(self, model_type: str, X: np.ndarray, y: np.ndarray,
                         hyperparameters: Dict[str, Any], websocket: Any,
                         message: WebSocketMessage, client_id: str):
        """Synchronous model training (called in executor)"""
        try:
            if model_type == 'qsvm':
                model = self.quantum_ml.create_qsvm(**hyperparameters)
            elif model_type == 'qnn':
                model = self.quantum_ml.create_qnn(**hyperparameters)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.fit(X, y)

            # Send completion message
            result_msg = WebSocketMessage(
                type=MessageType.TRAINING_REQUEST,
                id=message.id,
                payload={
                    'status': 'completed',
                    'model_type': model_type,
                    'training_samples': len(X),
                    'accuracy': model.score(X, y),
                    'model_id': f"{model_type}_{hash(str(X.tobytes()) + str(y.tobytes())):x}"
                },
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )

            # Schedule async send
            asyncio.create_task(self._send_message_async(websocket, result_msg))

        except Exception as e:
            error_msg = WebSocketMessage(
                type=MessageType.ERROR,
                id=message.id,
                payload={'message': f'Training failed: {str(e)}'},
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )
            asyncio.create_task(self._send_message_async(websocket, error_msg))

    async def _handle_federated_update(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle federated learning update"""
        try:
            if not self.federated_server:
                await self._send_error(websocket, "Federated server not available", client_id)
                return

            client_id_param = message.payload.get('client_id')
            model_update = message.payload.get('model_update')
            local_samples = message.payload.get('local_samples')

            if not all([client_id_param, model_update, local_samples]):
                await self._send_error(websocket, "Missing federated update parameters", client_id)
                return

            # Process update
            update_result = self.federated_server.process_update(
                client_id_param, model_update, local_samples
            )

            # Send confirmation
            result_msg = WebSocketMessage(
                type=MessageType.FEDERATED_UPDATE,
                id=message.id,
                payload={
                    'client_id': client_id_param,
                    'processed': True,
                    'version': update_result.get('version', 1),
                    'contribution_score': update_result.get('contribution', 0.0)
                },
                timestamp=datetime.utcnow(),
                user_id=getattr(websocket, '_user_id', None),
                session_id=client_id
            )
            await websocket.send(result_msg.to_json())

            # Broadcast update to subscribers
            await self._broadcast_to_subscribers(
                SubscriptionType.FEDERATED_UPDATES,
                result_msg.payload
            )

        except Exception as e:
            logger.error(f"Federated update error: {str(e)}")
            await self._send_error(websocket, f"Federated update failed: {str(e)}", client_id)

    async def _handle_heartbeat(self, websocket: Any, message: WebSocketMessage, client_id: str):
        """Handle heartbeat message"""
        # Update client heartbeat
        with self.client_lock:
            if client_id in self.connected_clients:
                self.connected_clients[client_id].last_heartbeat = datetime.utcnow()

        # Send heartbeat response
        heartbeat_msg = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            id=message.id,
            payload={'status': 'alive', 'timestamp': datetime.utcnow().isoformat()},
            timestamp=datetime.utcnow(),
            user_id=getattr(websocket, '_user_id', None),
            session_id=client_id
        )
        await websocket.send(heartbeat_msg.to_json())

    async def _send_error(self, websocket: Any, error_message: str, client_id: str):
        """Send error message to client"""
        error_msg = WebSocketMessage(
            type=MessageType.ERROR,
            id=str(uuid.uuid4()),
            payload={'message': error_message},
            timestamp=datetime.utcnow(),
            user_id=getattr(websocket, '_user_id', None),
            session_id=client_id
        )
        try:
            await websocket.send(error_msg.to_json())
        except Exception as e:
            logger.error(f"Failed to send error message: {str(e)}")

    async def _send_message_async(self, websocket: Any, message: WebSocketMessage):
        """Send message asynchronously"""
        try:
            await websocket.send(message.to_json())
        except Exception as e:
            logger.error(f"Failed to send async message: {str(e)}")

    async def _broadcast_to_subscribers(self, subscription_type: SubscriptionType,
                                      payload: Dict[str, Any], exclude_client: Optional[str] = None):
        """Broadcast message to subscribers"""
        if subscription_type not in self.subscribers:
            return

        subscriber_ids = self.subscribers[subscription_type].copy()
        if exclude_client:
            subscriber_ids.discard(exclude_client)

        if not subscriber_ids:
            return

        # Create broadcast message
        broadcast_msg = WebSocketMessage(
            type=MessageType.NOTIFICATION,
            id=str(uuid.uuid4()),
            payload={
                'subscription_type': subscription_type.value,
                'data': payload
            },
            timestamp=datetime.utcnow()
        )

        # Send to all subscribers
        tasks = []
        with self.client_lock:
            for client_id in subscriber_ids:
                if client_id in self.connected_clients:
                    client = self.connected_clients[client_id]
                    tasks.append(client.websocket.send(broadcast_msg.to_json()))

        # Execute broadcasts concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def broadcast_metrics_update(self, metrics: Dict[str, Any]):
        """Broadcast metrics update to subscribers"""
        asyncio.create_task(self._broadcast_to_subscribers(
            SubscriptionType.METRICS,
            metrics
        ))

    def broadcast_quantum_state_update(self, quantum_state: Dict[str, Any]):
        """Broadcast quantum state update"""
        asyncio.create_task(self._broadcast_to_subscribers(
            SubscriptionType.QUANTUM_STATES,
            quantum_state
        ))

    def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Check for dead connections
                now = datetime.utcnow()
                dead_clients = []

                with self.client_lock:
                    for client_id, client in self.connected_clients.items():
                        # Check heartbeat timeout (30 seconds)
                        if (now - client.last_heartbeat).total_seconds() > 30:
                            dead_clients.append(client_id)

                    # Clean up dead clients
                    for client_id in dead_clients:
                        if client_id in self.connected_clients:
                            client = self.connected_clients[client_id]
                            # Remove from subscriptions
                            for sub_type in client.subscriptions:
                                self.subscribers[sub_type].discard(client_id)
                            del self.connected_clients[client_id]
                            logger.info(f"Removed dead client: {client_id}")

                # Simulate metrics updates
                if len(self.connected_clients) > 0:
                    metrics = {
                        'active_connections': len(self.connected_clients),
                        'total_subscriptions': sum(len(subs) for subs in self.subscribers.values()),
                        'memory_usage_mb': 150.0,
                        'cpu_usage_percent': 25.0
                    }
                    self.broadcast_metrics_update(metrics)

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(5)

    async def start_server(self):
        """Start WebSocket server"""
        if not websockets:
            logger.error("websockets library not available")
            return

        self.running = True
        self.start_monitoring()

        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        try:
            server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                max_size=1048576,  # 1MB max message size
                max_queue=1024,     # Max queued connections
                read_limit=65536,   # 64KB read limit
                write_limit=65536   # 64KB write limit
            )

            logger.info("WebSocket server started successfully")
            await server.wait_closed()

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
        finally:
            self.running = False

    def stop_server(self):
        """Stop WebSocket server"""
        self.running = False
        logger.info("WebSocket server stopped")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self.client_lock:
            return {
                'active_connections': len(self.connected_clients),
                'subscription_counts': {
                    sub_type.value: len(subscribers)
                    for sub_type, subscribers in self.subscribers.items()
                },
                'total_subscriptions': sum(len(subs) for subs in self.subscribers.values()),
                'max_connections': self.max_connections
            }

if __name__ == '__main__':
    # Example usage
    import asyncio

    async def main():
        service = QuantumWebSocketService()

        # Initialize with sample components
        from ..edge_runtime.inference_engine import EdgeInferenceEngine, ModelSpec, Precision
        from ..quantum_algorithms.quantum_ml import QuantumMachineLearning

        model_spec = ModelSpec(
            input_shape=(784,),
            output_shape=(10,),
            num_parameters=100000,
            model_size_mb=50.0,
            supported_precisions=[Precision.FP32, Precision.FP16, Precision.INT8],
            target_latency_ms=10.0,
            power_budget_mw=500.0
        )

        service.inference_engine = EdgeInferenceEngine(model_spec)
        service.quantum_ml = QuantumMachineLearning()

        await service.start_server()

    asyncio.run(main())

"""
Quantum Edge AI Platform - API Models

Data models and schemas for API requests, responses, and internal data structures.
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
import uuid

# Third-party imports (would be installed in production)
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic.dataclasses import dataclass as pydantic_dataclass
except ImportError:
    # Fallback for development without dependencies
    BaseModel = Field = validator = ValidationError = None

    def pydantic_dataclass(cls):
        return cls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class APIStatus(Enum):
    """API response status"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PROCESSING = "processing"

class ModelType(Enum):
    """Machine learning model types"""
    QUANTUM_SVM = "quantum_svm"
    QUANTUM_NEURAL_NETWORK = "quantum_neural_network"
    CLASSICAL_NEURAL_NETWORK = "classical_neural_network"
    HYBRID_MODEL = "hybrid_model"

class InferencePrecision(Enum):
    """Inference precision levels"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"

class TaskStatus(Enum):
    """Task execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Base Models
@dataclass
class BaseAPIResponse:
    """Base API response model"""
    status: APIStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

@dataclass
class APIError(BaseAPIResponse):
    """API error response"""
    status: APIStatus = APIStatus.ERROR
    error_code: str = ""
    error_details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None

@dataclass
class APISuccess(BaseAPIResponse):
    """API success response"""
    status: APIStatus = APIStatus.SUCCESS
    data: Optional[Dict[str, Any]] = None

# Request Models
@dataclass
class BaseAPIRequest:
    """Base API request model"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    client_version: str = "1.0.0"
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAPIRequest':
        """Create from dictionary"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class InferenceRequest(BaseAPIRequest):
    """Inference request model"""
    model_id: str
    input_data: Union[List[float], List[List[float]]]
    precision: InferencePrecision = InferencePrecision.FP32
    batch_size: Optional[int] = None
    timeout_seconds: int = 30
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TrainingRequest(BaseAPIRequest):
    """Model training request"""
    model_type: ModelType
    training_data: List[List[float]]
    labels: List[Union[float, int]]
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    epochs: int = 100
    batch_size: int = 32

@dataclass
class ModelCreationRequest(BaseAPIRequest):
    """Model creation request"""
    name: str
    type: ModelType
    parameters: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class FederatedUpdateRequest(BaseAPIRequest):
    """Federated learning update request"""
    client_id: str
    model_update: Dict[str, Any]
    local_samples: int
    client_metadata: Optional[Dict[str, Any]] = None

@dataclass
class QuantumCircuitRequest(BaseAPIRequest):
    """Quantum circuit execution request"""
    circuit_definition: Dict[str, Any]
    shots: int = 1000
    backend: str = "simulator"
    optimization_level: int = 1

# Response Models
@dataclass
class InferenceResult:
    """Inference result model"""
    prediction: Union[List[float], List[List[float]]]
    confidence: Optional[Union[float, List[float]]] = None
    latency_ms: float = 0.0
    precision_used: InferencePrecision = InferencePrecision.FP32
    memory_used_mb: float = 0.0
    power_consumed_mw: float = 0.0
    model_id: str = ""
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TrainingResult:
    """Training result model"""
    model_id: str
    status: TaskStatus
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    training_time_seconds: float = 0.0
    epochs_completed: int = 0
    model_parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class ModelInfo:
    """Model information"""
    id: str
    name: str
    type: ModelType
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    size_mb: float = 0.0
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None

@dataclass
class QuantumStateResult:
    """Quantum state result"""
    amplitudes: List[complex]
    num_qubits: int
    fidelity: float = 0.0
    entropy: float = 0.0
    measurement_counts: Optional[Dict[str, int]] = None
    execution_time_ms: float = 0.0

@dataclass
class FederatedUpdateResult:
    """Federated learning update result"""
    client_id: str
    update_processed: bool
    global_model_version: int = 1
    contribution_score: float = 0.0
    local_model_accuracy: Optional[float] = None
    aggregation_weight: float = 1.0

@dataclass
class MetricsData:
    """System metrics data"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    requests_per_minute: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

# Pydantic Models (if available)
if BaseModel:
    class InferenceRequestModel(BaseModel):
        """Pydantic model for inference requests"""
        model_id: str = Field(..., min_length=1, max_length=100)
        input_data: Union[List[float], List[List[float]]] = Field(..., min_items=1)
        precision: str = Field(default="fp32", regex=r'^(fp32|fp16|int8|int4|binary)$')
        batch_size: Optional[int] = Field(default=None, ge=1, le=1000)
        timeout_seconds: int = Field(default=30, ge=1, le=300)
        metadata: Optional[Dict[str, Any]] = None

        @validator('input_data')
        def validate_input_data(cls, v):
            """Validate input data dimensions"""
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], list):  # 2D array
                    if len(v) > 1000:
                        raise ValueError("Batch size too large")
                    for row in v:
                        if len(row) > 10000:
                            raise ValueError("Input dimension too large")
                else:  # 1D array
                    if len(v) > 10000:
                        raise ValueError("Input dimension too large")
            return v

    class TrainingRequestModel(BaseModel):
        """Pydantic model for training requests"""
        model_type: str = Field(..., regex=r'^(quantum_svm|quantum_neural_network|classical_neural_network|hybrid_model)$')
        training_data: List[List[float]] = Field(..., min_items=1, max_items=10000)
        labels: List[Union[float, int]] = Field(..., min_items=1, max_items=10000)
        hyperparameters: Optional[Dict[str, Any]] = None
        validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
        epochs: int = Field(default=100, ge=1, le=1000)
        batch_size: int = Field(default=32, ge=1, le=1024)

        @validator('training_data', 'labels')
        def validate_data_consistency(cls, v, values):
            """Ensure training data and labels have same length"""
            if 'training_data' in values and len(v) != len(values['training_data']):
                raise ValueError("Labels length must match training data length")
            return v

    class ModelCreationRequestModel(BaseModel):
        """Pydantic model for model creation requests"""
        name: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
        type: str = Field(..., regex=r'^(quantum_svm|quantum_neural_network|classical_neural_network|hybrid_model)$')
        parameters: Optional[Dict[str, Any]] = None
        description: Optional[str] = Field(default=None, max_length=500)
        tags: List[str] = Field(default_factory=list, max_items=10)

        @validator('tags')
        def validate_tags(cls, v):
            """Validate tag format"""
            for tag in v:
                if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                    raise ValueError(f"Invalid tag format: {tag}")
            return v

# Validation and Serialization
class ModelValidator:
    """Model validation utilities"""

    @staticmethod
    def validate_inference_request(data: Dict[str, Any]) -> InferenceRequest:
        """Validate inference request data"""
        try:
            # Basic validation
            if not isinstance(data, dict):
                raise ValueError("Request must be a dictionary")

            required_fields = ['model_id', 'input_data']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Convert precision string to enum
            precision = InferencePrecision.FP32
            if 'precision' in data:
                try:
                    precision = InferencePrecision(data['precision'])
                except ValueError:
                    raise ValueError(f"Invalid precision: {data['precision']}")

            # Create request object
            request = InferenceRequest(
                model_id=data['model_id'],
                input_data=data['input_data'],
                precision=precision,
                batch_size=data.get('batch_size'),
                timeout_seconds=data.get('timeout_seconds', 30),
                metadata=data.get('metadata'),
                user_id=data.get('user_id')
            )

            return request

        except Exception as e:
            raise ValueError(f"Invalid inference request: {str(e)}")

    @staticmethod
    def validate_training_request(data: Dict[str, Any]) -> TrainingRequest:
        """Validate training request data"""
        try:
            required_fields = ['model_type', 'training_data', 'labels']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Convert model type string to enum
            try:
                model_type = ModelType(data['model_type'])
            except ValueError:
                raise ValueError(f"Invalid model type: {data['model_type']}")

            # Validate data consistency
            training_data = data['training_data']
            labels = data['labels']

            if len(training_data) != len(labels):
                raise ValueError("Training data and labels must have same length")

            request = TrainingRequest(
                model_type=model_type,
                training_data=training_data,
                labels=labels,
                hyperparameters=data.get('hyperparameters'),
                validation_split=data.get('validation_split', 0.2),
                epochs=data.get('epochs', 100),
                batch_size=data.get('batch_size', 32),
                user_id=data.get('user_id')
            )

            return request

        except Exception as e:
            raise ValueError(f"Invalid training request: {str(e)}")

class ResponseSerializer:
    """Response serialization utilities"""

    @staticmethod
    def serialize_inference_result(result: InferenceResult) -> Dict[str, Any]:
        """Serialize inference result to JSON-compatible format"""
        data = asdict(result)
        data['precision_used'] = result.precision_used.value

        # Convert complex numbers if present
        if 'prediction' in data:
            data['prediction'] = ResponseSerializer._serialize_complex_data(data['prediction'])

        return data

    @staticmethod
    def serialize_quantum_state(state: QuantumStateResult) -> Dict[str, Any]:
        """Serialize quantum state result"""
        data = asdict(state)
        data['amplitudes'] = [
            [amp.real, amp.imag] for amp in state.amplitudes
        ]
        return data

    @staticmethod
    def _serialize_complex_data(data: Any) -> Any:
        """Recursively serialize complex data structures"""
        if isinstance(data, complex):
            return [data.real, data.imag]
        elif isinstance(data, list):
            return [ResponseSerializer._serialize_complex_data(item) for item in data]
        elif isinstance(data, dict):
            return {
                key: ResponseSerializer._serialize_complex_data(value)
                for key, value in data.items()
            }
        else:
            return data

# API Response Builders
class APIResponseBuilder:
    """Utility class for building API responses"""

    @staticmethod
    def success_response(message: str = "Operation completed successfully",
                        data: Optional[Dict[str, Any]] = None,
                        request_id: Optional[str] = None) -> APISuccess:
        """Build success response"""
        return APISuccess(
            message=message,
            data=data,
            request_id=request_id or str(uuid.uuid4())
        )

    @staticmethod
    def error_response(message: str, error_code: str = "INTERNAL_ERROR",
                      error_details: Optional[Dict[str, Any]] = None,
                      retry_after: Optional[int] = None,
                      request_id: Optional[str] = None) -> APIError:
        """Build error response"""
        return APIError(
            message=message,
            error_code=error_code,
            error_details=error_details,
            retry_after=retry_after,
            request_id=request_id or str(uuid.uuid4())
        )

    @staticmethod
    def inference_response(result: InferenceResult,
                          request_id: Optional[str] = None) -> APISuccess:
        """Build inference response"""
        data = ResponseSerializer.serialize_inference_result(result)
        return APIResponseBuilder.success_response(
            message="Inference completed successfully",
            data=data,
            request_id=request_id
        )

    @staticmethod
    def training_response(result: TrainingResult,
                         request_id: Optional[str] = None) -> APISuccess:
        """Build training response"""
        data = asdict(result)
        data['status'] = result.status.value
        data['model_type'] = result.model_parameters.get('model_type', '') if result.model_parameters else ''

        return APIResponseBuilder.success_response(
            message="Training completed successfully",
            data=data,
            request_id=request_id
        )

    @staticmethod
    def quantum_state_response(state: QuantumStateResult,
                              request_id: Optional[str] = None) -> APISuccess:
        """Build quantum state response"""
        data = ResponseSerializer.serialize_quantum_state(state)
        return APIResponseBuilder.success_response(
            message="Quantum state computed successfully",
            data=data,
            request_id=request_id
        )

# Export all classes
__all__ = [
    'APIStatus', 'ModelType', 'InferencePrecision', 'TaskStatus',
    'BaseAPIResponse', 'APIError', 'APISuccess',
    'BaseAPIRequest', 'InferenceRequest', 'TrainingRequest',
    'ModelCreationRequest', 'FederatedUpdateRequest', 'QuantumCircuitRequest',
    'InferenceResult', 'TrainingResult', 'ModelInfo', 'QuantumStateResult',
    'FederatedUpdateResult', 'MetricsData',
    'ModelValidator', 'ResponseSerializer', 'APIResponseBuilder'
]

# Add Pydantic models if available
if BaseModel:
    __all__.extend([
        'InferenceRequestModel', 'TrainingRequestModel', 'ModelCreationRequestModel'
    ])

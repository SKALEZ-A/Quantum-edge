# API Reference

Complete API documentation for the Quantum Edge AI Platform.

## Overview

The platform provides multiple API interfaces:

- **REST API**: Traditional HTTP-based API for synchronous operations
- **GraphQL API**: Flexible query language for complex data requirements
- **WebSocket API**: Real-time communication for streaming data and events
- **gRPC API**: High-performance binary protocol for edge devices

## Base URLs

```
REST API:    http://localhost:8000/api/v1
GraphQL API: http://localhost:8000/graphql
WebSocket:   ws://localhost:8000/websocket
gRPC:        localhost:50051
```

## Authentication

All APIs require authentication using JWT tokens:

```bash
# Obtain token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:8000/api/v1/models
```

## REST API Reference

### Health Check

**GET** `/health`

Check system health and status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "components": {
    "database": "healthy",
    "quantum_engine": "healthy",
    "edge_runtime": "healthy"
  }
}
```

### Model Management

#### List Models
**GET** `/models`

List all available models.

**Parameters:**
- `category` (optional): Filter by category (`quantum`, `classical`, `hybrid`)
- `status` (optional): Filter by status (`active`, `inactive`, `training`)

**Response:**
```json
{
  "models": [
    {
      "id": "qsvm_001",
      "name": "Quantum SVM Classifier",
      "category": "quantum",
      "status": "active",
      "accuracy": 0.923,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-16T14:20:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "per_page": 10
}
```

#### Get Model Details
**GET** `/models/{model_id}`

Get detailed information about a specific model.

**Response:**
```json
{
  "id": "qsvm_001",
  "name": "Quantum SVM Classifier",
  "description": "Quantum-enhanced support vector machine",
  "category": "quantum",
  "status": "active",
  "metrics": {
    "accuracy": 0.923,
    "precision": 0.918,
    "recall": 0.929,
    "f1_score": 0.923
  },
  "parameters": {
    "n_qubits": 4,
    "kernel": "rbf",
    "gamma": 0.1
  },
  "metadata": {
    "framework": "qiskit",
    "version": "0.25.0",
    "training_time": 120.5
  }
}
```

#### Upload Model
**POST** `/models`

Upload a new model.

**Request Body:**
```json
{
  "name": "Custom Quantum Model",
  "description": "User-trained quantum model",
  "category": "quantum",
  "model_data": "base64_encoded_model_data",
  "parameters": {
    "n_qubits": 6,
    "circuit_depth": 10
  }
}
```

**Response:**
```json
{
  "id": "custom_001",
  "status": "uploaded",
  "message": "Model uploaded successfully"
}
```

#### Delete Model
**DELETE** `/models/{model_id}`

Delete a model.

**Response:**
```json
{
  "message": "Model deleted successfully"
}
```

### Inference

#### Run Inference
**POST** `/inference`

Execute inference on input data.

**Request Body:**
```json
{
  "model_id": "qsvm_001",
  "input_data": [0.1, 0.2, 0.3, 0.4],
  "options": {
    "precision": "FP16",
    "timeout": 30
  }
}
```

**Response:**
```json
{
  "model_id": "qsvm_001",
  "prediction": 1,
  "confidence": 0.87,
  "probabilities": [0.13, 0.87],
  "processing_time": 0.023,
  "edge_optimized": true
}
```

#### Batch Inference
**POST** `/inference/batch`

Execute batch inference on multiple inputs.

**Request Body:**
```json
{
  "model_id": "qsvm_001",
  "input_batch": [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
  ],
  "options": {
    "batch_size": 32,
    "timeout": 60
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "prediction": 1,
      "confidence": 0.87,
      "probabilities": [0.13, 0.87]
    },
    {
      "prediction": 0,
      "confidence": 0.92,
      "probabilities": [0.92, 0.08]
    },
    {
      "prediction": 1,
      "confidence": 0.89,
      "probabilities": [0.11, 0.89]
    }
  ],
  "batch_processing_time": 0.067,
  "average_latency": 0.022
}
```

### Privacy & Security

#### Apply Privacy Protection
**POST** `/privacy/protect`

Apply privacy-preserving transformations to data.

**Request Body:**
```json
{
  "data": [[1, 2, 3], [4, 5, 6]],
  "privacy_level": "high",
  "epsilon": 0.5,
  "method": "differential_privacy"
}
```

**Response:**
```json
{
  "protected_data": [[1.02, 2.01, 2.98], [4.03, 5.02, 5.97]],
  "privacy_report": {
    "epsilon_used": 0.5,
    "privacy_loss": 0.45,
    "utility_preserved": 0.92
  }
}
```

#### Verify Compliance
**GET** `/compliance/{framework}`

Check compliance status for a specific framework.

**Parameters:**
- `framework`: Compliance framework (`gdpr`, `hipaa`, `ccpa`)

**Response:**
```json
{
  "framework": "gdpr",
  "status": "compliant",
  "last_audit": "2024-01-15T10:00:00Z",
  "violations": [],
  "recommendations": [
    "Consider implementing additional access controls"
  ]
}
```

### Federated Learning

#### Register Client
**POST** `/federated/register`

Register a new federated learning client.

**Request Body:**
```json
{
  "client_id": "client_001",
  "client_type": "edge_device",
  "capabilities": {
    "max_data_size": 1000,
    "supported_models": ["quantum", "classical"],
    "privacy_budget": 1.0
  }
}
```

**Response:**
```json
{
  "registration_id": "reg_001",
  "global_model_hash": "abc123...",
  "training_config": {
    "rounds": 10,
    "epochs_per_round": 5,
    "learning_rate": 0.01
  }
}
```

#### Submit Model Update
**POST** `/federated/submit`

Submit model updates from a client.

**Request Body:**
```json
{
  "client_id": "client_001",
  "round_id": 5,
  "model_updates": {
    "weights": "compressed_model_weights",
    "gradients": "quantized_gradients"
  },
  "metadata": {
    "local_accuracy": 0.89,
    "training_time": 120.5,
    "data_size": 500
  }
}
```

**Response:**
```json
{
  "status": "accepted",
  "round_id": 5,
  "aggregation_eta": 300,
  "next_round_config": {
    "learning_rate": 0.009,
    "epochs": 6
  }
}
```

### Monitoring & Metrics

#### Get System Metrics
**GET** `/monitoring/metrics`

Get current system metrics.

**Parameters:**
- `period` (optional): Time period (`1h`, `24h`, `7d`)
- `metrics` (optional): Comma-separated list of metrics

**Response:**
```json
{
  "timestamp": "2024-01-15T12:00:00Z",
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "disk_usage": 78.5
  },
  "models": {
    "active_models": 12,
    "total_inferences": 15420,
    "average_latency": 0.023
  },
  "quantum": {
    "circuits_executed": 8920,
    "average_fidelity": 0.945,
    "error_rate": 0.0023
  }
}
```

#### Get Performance History
**GET** `/monitoring/history`

Get historical performance data.

**Parameters:**
- `start_time`: ISO 8601 start time
- `end_time`: ISO 8601 end time
- `resolution`: Data resolution (`1m`, `5m`, `1h`)

**Response:**
```json
{
  "period": {
    "start": "2024-01-15T00:00:00Z",
    "end": "2024-01-15T12:00:00Z",
    "resolution": "5m"
  },
  "metrics": [
    {
      "timestamp": "2024-01-15T00:00:00Z",
      "cpu_percent": 42.1,
      "memory_percent": 58.9,
      "inference_count": 145
    },
    {
      "timestamp": "2024-01-15T00:05:00Z",
      "cpu_percent": 44.3,
      "memory_percent": 61.2,
      "inference_count": 152
    }
  ]
}
```

### Configuration

#### Get Configuration
**GET** `/config`

Get current system configuration.

**Response:**
```json
{
  "edge_runtime": {
    "max_memory_mb": 512,
    "default_precision": "FP16",
    "enable_quantization": true
  },
  "quantum_engine": {
    "n_qubits": 4,
    "simulator": "qiskit_aer",
    "optimization_level": 2
  },
  "privacy": {
    "default_epsilon": 0.5,
    "enable_differential_privacy": true,
    "compliance_frameworks": ["gdpr", "ccpa"]
  }
}
```

#### Update Configuration
**PUT** `/config`

Update system configuration.

**Request Body:**
```json
{
  "edge_runtime": {
    "max_memory_mb": 1024
  },
  "quantum_engine": {
    "optimization_level": 3
  }
}
```

**Response:**
```json
{
  "message": "Configuration updated successfully",
  "restart_required": false
}
```

## GraphQL API

### Schema

```graphql
type Query {
  health: HealthStatus!
  models(category: String, status: String): [Model!]!
  model(id: ID!): Model
  metrics(period: String): SystemMetrics!
  compliance(framework: String!): ComplianceStatus!
}

type Mutation {
  runInference(modelId: ID!, input: [Float!]!): InferenceResult!
  uploadModel(input: UploadModelInput!): Model!
  applyPrivacy(input: PrivacyInput!): PrivacyResult!
  updateConfig(input: ConfigInput!): ConfigResult!
}

type Subscription {
  metricsUpdate: SystemMetrics!
  inferenceComplete: InferenceResult!
}
```

### Example Queries

#### Get Model Information
```graphql
query GetModel($id: ID!) {
  model(id: $id) {
    id
    name
    metrics {
      accuracy
      precision
      recall
    }
    parameters {
      nQubits
      kernel
    }
  }
}
```

#### Run Inference
```graphql
mutation RunInference($modelId: ID!, $input: [Float!]!) {
  runInference(modelId: $modelId, input: $input) {
    prediction
    confidence
    processingTime
  }
}
```

#### Subscribe to Metrics
```graphql
subscription MetricsUpdate {
  metricsUpdate {
    system {
      cpuPercent
      memoryPercent
    }
    models {
      activeModels
      averageLatency
    }
  }
}
```

## WebSocket API

### Connection

```javascript
import io from 'socket.io-client';

const socket = io('ws://localhost:8000/websocket', {
  auth: {
    token: 'your_jwt_token'
  }
});
```

### Events

#### Subscribe to Metrics
```javascript
socket.emit('subscribe_metrics', {
  interval: '5s',
  metrics: ['cpu', 'memory', 'inference']
});
```

#### Listen for Updates
```javascript
socket.on('metrics_update', (data) => {
  console.log('Metrics:', data);
  // {
  //   timestamp: "2024-01-15T12:00:00Z",
  //   cpu_percent: 45.2,
  //   memory_percent: 62.1,
  //   inference_count: 15420
  // }
});
```

#### Real-time Inference
```javascript
// Send inference request
socket.emit('run_inference', {
  model_id: 'qsvm_001',
  input_data: [0.1, 0.2, 0.3, 0.4]
});

// Listen for result
socket.on('inference_result', (result) => {
  console.log('Prediction:', result.prediction);
  console.log('Confidence:', result.confidence);
});
```

#### Federated Learning Updates
```javascript
// Join federated learning room
socket.emit('join_federated', {
  client_id: 'client_001'
});

// Listen for global model updates
socket.on('global_model_update', (update) => {
  console.log('New global model:', update.model_hash);
  // Download and apply new model
});
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Unprocessable Entity
- `429`: Too Many Requests
- `500`: Internal Server Error
- `503`: Service Unavailable

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "model_id",
      "reason": "Model not found"
    },
    "timestamp": "2024-01-15T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid input data
- `AUTHENTICATION_ERROR`: Invalid or missing credentials
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `RESOURCE_LIMIT_EXCEEDED`: Rate limit or quota exceeded
- `QUANTUM_ENGINE_ERROR`: Quantum computation failed
- `PRIVACY_VIOLATION`: Privacy constraints not met
- `MODEL_INFERENCE_ERROR`: Inference execution failed

## Rate Limiting

API endpoints are rate limited to prevent abuse:

- **REST API**: 1000 requests per hour per user
- **GraphQL**: 5000 operations per hour per user
- **WebSocket**: 1000 messages per minute per connection

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1642185600
```

## Versioning

API versions follow semantic versioning:

- **v1**: Current stable version
- **v2**: Next major version (breaking changes)
- **beta**: Preview features

Specify version in request headers:

```
Accept: application/vnd.quantum-edge-ai.v1+json
```

## SDKs and Libraries

### Python SDK

```python
from quantum_edge_ai import Client

client = Client(api_key='your_key')

# Run inference
result = client.inference.run('qsvm_001', [0.1, 0.2, 0.3, 0.4])

# Upload model
model = client.models.upload('model.pkl', name='My Model')

# Get metrics
metrics = client.monitoring.get_metrics()
```

### JavaScript SDK

```javascript
import { QuantumEdgeAI } from 'quantum-edge-ai-sdk';

const client = new QuantumEdgeAI({ apiKey: 'your_key' });

// Run inference
const result = await client.inference.run('qsvm_001', [0.1, 0.2, 0.3, 0.4]);

// Real-time updates
client.websocket.subscribe('metrics', (data) => {
  console.log('Live metrics:', data);
});
```

### Go SDK

```go
package main

import (
    "github.com/quantum-edge-ai/go-sdk"
)

func main() {
    client := sdk.NewClient("your_api_key")

    // Run inference
    result, err := client.Inference.Run("qsvm_001", []float64{0.1, 0.2, 0.3, 0.4})

    // Handle result
    if err == nil {
        fmt.Printf("Prediction: %v, Confidence: %.2f\n",
            result.Prediction, result.Confidence)
    }
}
```

---

For more detailed examples and tutorials, see the [Getting Started Guide](../tutorials/getting_started.md).

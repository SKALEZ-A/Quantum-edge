# Getting Started with Quantum Edge AI Platform

Welcome to the Quantum Edge AI Platform! This tutorial will guide you through the initial setup and basic usage of the platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Platform Overview](#platform-overview)
4. [Your First Quantum ML Model](#first-model)
5. [Web Dashboard](#dashboard)
6. [API Usage](#api-usage)
7. [Next Steps](#next-steps)

## Prerequisites

Before getting started, ensure you have:

- **Python 3.8+** installed
- **pip** package manager
- **Git** for cloning repositories
- **Web browser** for the dashboard
- **Optional**: CUDA-compatible GPU for accelerated quantum simulations

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores, 10GB storage
- **Recommended**: 8GB RAM, 4+ CPU cores, 50GB storage
- **For Quantum Simulations**: 16GB+ RAM recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/quantum-edge-ai-platform.git
cd quantum-edge-ai-platform
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv quantum_env

# Activate virtual environment
# On Windows:
quantum_env\Scripts\activate
# On macOS/Linux:
source quantum_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install the platform
pip install -e .

# For development and examples:
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```python
import quantum_edge_ai_platform
print(f"Version: {quantum_edge_ai_platform.__version__}")
print("Installation successful!")
```

## Platform Overview

The Quantum Edge AI Platform consists of several key components:

### Core Components

- **Edge Runtime**: Optimized inference engine for edge devices
- **Quantum Algorithms**: Quantum-enhanced machine learning algorithms
- **Federated Learning**: Privacy-preserving distributed learning
- **Privacy & Security**: Advanced privacy protection and security
- **Web Interface**: Dashboard for monitoring and management
- **API Services**: REST, GraphQL, and WebSocket APIs

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │    API Gateway  │    │   Edge Devices  │
│                 │    │                 │    │                 │
│ - Real-time     │◄──►│ - REST/GraphQL  │◄──►│ - Inference     │
│   Monitoring    │    │ - WebSocket     │    │ - Local ML      │
│ - Model Mgmt    │    │ - Load Balance  │    │ - Sensor Fusion │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Core Platform  │
                    │                 │
                    │ - Quantum ML    │
                    │ - Privacy Ctrl  │
                    │ - Fed Learning  │
                    │ - Edge Runtime  │
                    └─────────────────┘
```

## Your First Quantum ML Model

Let's create and train your first quantum machine learning model!

### 1. Import Required Components

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import platform components
from quantum_algorithms.quantum_ml import QuantumMachineLearning
from utils.data_processing import DataProcessor
from utils.monitoring import PerformanceProfiler
```

### 2. Prepare Data

```python
# Generate sample classification data
print("Generating sample data...")
X, y = make_classification(
    n_samples=300,
    n_features=4,
    n_classes=2,
    n_redundant=0,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
```

### 3. Initialize Quantum ML

```python
# Initialize the Quantum ML system
print("Initializing Quantum ML...")
qml = QuantumMachineLearning(n_qubits=4)

# Initialize performance profiler
profiler = PerformanceProfiler()
```

### 4. Train a Quantum SVM

```python
# Train Quantum Support Vector Machine
print("Training Quantum SVM...")
profiler.start_profile("quantum_svm_training")

qml.train_classifier(X_train, y_train, method='qsvm')

training_time = profiler.end_profile("quantum_svm_training")
print(f"Training completed in {training_time:.4f} seconds")
```

### 5. Make Predictions

```python
# Make predictions
print("Making predictions...")
profiler.start_profile("quantum_svm_prediction")

predictions = qml.classify(X_test)

prediction_time = profiler.end_profile("quantum_svm_prediction")

# Calculate accuracy
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Prediction time: {prediction_time:.4f} seconds")
print("\\nClassification Report:")
print(classification_report(y_test, predictions))
```

### 6. Compare with Classical SVM

```python
# Compare with classical SVM
print("\\nComparing with Classical SVM...")
from sklearn.svm import SVC

classical_svm = SVC(kernel='rbf', C=1.0, random_state=42)

# Train classical SVM
profiler.start_profile("classical_svm_training")
classical_svm.fit(X_train, y_train)
classical_training_time = profiler.end_profile("classical_svm_training")

# Predict with classical SVM
profiler.start_profile("classical_svm_prediction")
classical_predictions = classical_svm.predict(X_test)
classical_prediction_time = profiler.end_profile("classical_svm_prediction")

classical_accuracy = accuracy_score(y_test, classical_predictions)

print(f"\\nClassical SVM Results:")
print(f"Accuracy: {classical_accuracy:.4f}")
print(f"Training time: {classical_training_time:.4f} seconds")
print(f"Prediction time: {classical_prediction_time:.4f} seconds")

print(f"\\nPerformance Comparison:")
print(f"Accuracy improvement: {(accuracy/classical_accuracy - 1)*100:.2f}%")
print(f"Training speedup: {classical_training_time/training_time:.2f}x")
```

## Web Dashboard

The platform includes a comprehensive web dashboard for monitoring and management.

### Starting the Dashboard

```python
from web_interface import run_dashboard

# Run dashboard on default port (5000)
run_dashboard(debug=True)
```

Then open your browser to `http://localhost:5000`

### Dashboard Features

- **Real-time System Monitoring**: CPU, memory, disk usage
- **Model Management**: Upload, train, and deploy models
- **Performance Analytics**: View metrics and analytics
- **API Testing**: Test API endpoints directly
- **Log Viewer**: Monitor system logs in real-time

### Using the Dashboard

1. **Overview Tab**: See system health and key metrics
2. **Models Tab**: Manage your AI models
3. **Monitoring Tab**: Detailed system and performance metrics
4. **Analytics Tab**: Advanced analytics and insights

## API Usage

The platform provides comprehensive APIs for integration.

### REST API Examples

```python
import requests

# Base URL
BASE_URL = "http://localhost:5000"

# Health check
response = requests.get(f"{BASE_URL}/api/health")
print(f"API Status: {response.json()}")

# Model inference
inference_data = {
    "model_id": "quantum_svm",
    "input": [[0.1, 0.2, 0.3, 0.4]]
}

response = requests.post(
    f"{BASE_URL}/api/inference",
    json=inference_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"Error: {response.text}")
```

### WebSocket API

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Received: {data}")

def on_open(ws):
    print("Connected to WebSocket")
    # Subscribe to metrics
    ws.send(json.dumps({
        'type': 'subscribe_metrics'
    }))

ws = websocket.WebSocketApp(
    "ws://localhost:5000/websocket",
    on_message=on_message,
    on_open=on_open
)

ws.run_forever()
```

## Next Steps

Now that you've completed the basics, here are some next steps:

### 1. Explore Advanced Features

- **Federated Learning**: Privacy-preserving distributed learning
- **Edge Deployment**: Deploy models on edge devices
- **Quantum Neural Networks**: More complex quantum models
- **Custom Quantum Algorithms**: Implement your own quantum algorithms

### 2. Application-Specific Tutorials

- [Autonomous Vehicles](./autonomous_vehicles_tutorial.md)
- [Healthcare AI](./healthcare_tutorial.md)
- [Industrial IoT](./industrial_iot_tutorial.md)
- [Smart City](./smart_city_tutorial.md)

### 3. Advanced Topics

- [Privacy & Security](./privacy_security_guide.md)
- [Performance Optimization](./performance_optimization.md)
- [Custom Model Development](./custom_models.md)
- [Edge Runtime Configuration](./edge_runtime_config.md)

### 4. API Reference

- [REST API Documentation](./api/rest_api.md)
- [GraphQL API](./api/graphql_api.md)
- [WebSocket API](./api/websocket_api.md)

### 5. Contributing

Want to contribute to the platform?

- [Development Setup](./development_setup.md)
- [Code Style Guide](./code_style.md)
- [Testing Guidelines](./testing_guide.md)
- [Contributing Guide](./contributing.md)

## Support

If you encounter issues or have questions:

- **Documentation**: Check the full [documentation](../README.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-org/quantum-edge-ai-platform/issues)
- **Discussions**: Join the [GitHub Discussions](https://github.com/your-org/quantum-edge-ai-platform/discussions)

## What's Next?

You've successfully set up and used the Quantum Edge AI Platform! The platform offers:

- **Quantum-Enhanced ML**: Leverage quantum computing for machine learning
- **Edge Optimization**: Deploy AI models on resource-constrained devices
- **Privacy Protection**: Advanced privacy-preserving techniques
- **Distributed Learning**: Federated learning capabilities
- **Real-time Monitoring**: Comprehensive system monitoring and management

Continue exploring the tutorials and documentation to unlock the full potential of quantum edge AI!

---

*Happy coding with Quantum Edge AI! 🚀*

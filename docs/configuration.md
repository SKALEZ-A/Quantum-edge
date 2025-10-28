# Configuration Guide

Comprehensive guide for configuring the Quantum Edge AI Platform.

## Configuration Overview

The platform uses a hierarchical configuration system that supports multiple environments and deployment scenarios:

- **Environment Variables**: Runtime configuration
- **Configuration Files**: Structured YAML/JSON files
- **Database Configs**: Dynamic runtime configuration
- **CLI Overrides**: Command-line configuration options

## Configuration Hierarchy

Configuration values are resolved in this order (later sources override earlier ones):

1. **Default Values** (hardcoded in code)
2. **Configuration Files** (`config.yaml`, `config.json`)
3. **Environment Variables** (`QUANTUM_EDGE_*`)
4. **Runtime Overrides** (programmatic changes)
5. **CLI Arguments** (highest priority)

## Core Configuration

### Basic Configuration File

Create `config.yaml` in your project root:

```yaml
# Quantum Edge AI Platform Configuration
version: "1.0"
environment: "development"

# System Settings
system:
  log_level: "INFO"
  max_workers: 4
  enable_debug: false
  temp_directory: "/tmp/quantum_edge"

# Edge Runtime Configuration
edge_runtime:
  max_memory_mb: 512
  default_precision: "FP16"
  enable_quantization: true
  adaptive_precision: true
  compression_level: "medium"
  cache_enabled: true
  cache_size_mb: 256

# Quantum Engine Configuration
quantum_engine:
  n_qubits: 4
  simulator: "qiskit_aer"
  optimization_level: 2
  noise_model: "none"
  coupling_map: "linear"
  basis_gates: ["u1", "u2", "u3", "cx"]
  max_circuit_depth: 100
  enable_error_mitigation: false

# Privacy & Security Configuration
privacy:
  default_epsilon: 0.5
  enable_differential_privacy: true
  privacy_budget_per_user: 1.0
  compliance_frameworks: ["gdpr", "ccpa"]
  audit_log_enabled: true
  audit_log_retention_days: 365
  encryption_algorithm: "AES-256-GCM"
  key_rotation_days: 90

# Federated Learning Configuration
federated_learning:
  min_clients_per_round: 3
  max_clients_per_round: 10
  aggregation_algorithm: "fedavg"
  secure_aggregation: true
  client_selection_strategy: "random"
  max_training_rounds: 100
  convergence_threshold: 0.001
  privacy_guarantee: "differential_privacy"

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
  cors_origins: ["http://localhost:3000", "https://dashboard.quantum-edge-ai.com"]
  rate_limit_requests: 1000
  rate_limit_window_seconds: 3600
  enable_graphql: true
  enable_websocket: true
  enable_grpc: true
  jwt_secret_key: "your-secret-key-here"
  jwt_expiration_hours: 24

# Database Configuration
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  database: "quantum_edge"
  username: "quantum_user"
  password: "secure_password"
  connection_pool_size: 10
  connection_timeout_seconds: 30
  enable_ssl: true

# Monitoring Configuration
monitoring:
  enable_prometheus: true
  prometheus_port: 9090
  enable_grafana: true
  grafana_port: 3000
  metrics_retention_days: 30
  alert_email_enabled: true
  alert_email_recipients: ["admin@quantum-edge-ai.com"]
  log_aggregation_enabled: true
  log_retention_days: 90

# Deployment Configuration
deployment:
  docker_enabled: true
  kubernetes_enabled: false
  auto_scaling_enabled: true
  min_replicas: 2
  max_replicas: 10
  cpu_threshold: 70
  memory_threshold: 80
  health_check_interval_seconds: 30
  graceful_shutdown_timeout_seconds: 30
```

## Environment-Specific Configurations

### Development Environment

```yaml
# config.dev.yaml
environment: "development"

system:
  log_level: "DEBUG"
  enable_debug: true

edge_runtime:
  max_memory_mb: 1024

quantum_engine:
  simulator: "qiskit_aer"
  enable_error_mitigation: false

api:
  host: "localhost"
  port: 8000

database:
  type: "sqlite"
  database: ":memory:"
```

### Production Environment

```yaml
# config.prod.yaml
environment: "production"

system:
  log_level: "WARNING"
  enable_debug: false
  max_workers: 8

edge_runtime:
  max_memory_mb: 2048
  compression_level: "high"

quantum_engine:
  optimization_level: 3
  enable_error_mitigation: true

privacy:
  default_epsilon: 0.1
  privacy_budget_per_user: 0.5

api:
  host: "0.0.0.0"
  port: 443
  enable_ssl: true

monitoring:
  enable_prometheus: true
  enable_grafana: true
  alert_email_enabled: true

deployment:
  kubernetes_enabled: true
  auto_scaling_enabled: true
```

### Edge Device Environment

```yaml
# config.edge.yaml
environment: "edge"

system:
  max_workers: 1

edge_runtime:
  max_memory_mb: 256
  default_precision: "INT8"
  compression_level: "high"
  cache_enabled: false

quantum_engine:
  n_qubits: 2
  simulator: "qiskit_basic_aer"
  max_circuit_depth: 20

federated_learning:
  client_selection_strategy: "edge_optimized"

api:
  port: 8080

monitoring:
  metrics_retention_days: 7
  alert_email_enabled: false
```

## Environment Variables

Override any configuration value using environment variables with the prefix `QUANTUM_EDGE_`:

```bash
# System settings
export QUANTUM_EDGE_SYSTEM_LOG_LEVEL=DEBUG
export QUANTUM_EDGE_SYSTEM_MAX_WORKERS=8

# Edge runtime
export QUANTUM_EDGE_EDGE_RUNTIME_MAX_MEMORY_MB=1024
export QUANTUM_EDGE_EDGE_RUNTIME_DEFAULT_PRECISION=FP16

# Quantum engine
export QUANTUM_EDGE_QUANTUM_ENGINE_N_QUBITS=6
export QUANTUM_EDGE_QUANTUM_ENGINE_SIMULATOR=qiskit_aer

# API settings
export QUANTUM_EDGE_API_HOST=0.0.0.0
export QUANTUM_EDGE_API_PORT=8000
export QUANTUM_EDGE_API_JWT_SECRET_KEY=your-super-secret-key

# Database
export QUANTUM_EDGE_DATABASE_HOST=localhost
export QUANTUM_EDGE_DATABASE_PASSWORD=secure_password

# Privacy settings
export QUANTUM_EDGE_PRIVACY_DEFAULT_EPSILON=0.5
export QUANTUM_EDGE_PRIVACY_ENABLE_DIFFERENTIAL_PRIVACY=true
```

### Docker Environment Variables

```dockerfile
# Dockerfile
ENV QUANTUM_EDGE_SYSTEM_LOG_LEVEL=INFO
ENV QUANTUM_EDGE_API_HOST=0.0.0.0
ENV QUANTUM_EDGE_DATABASE_HOST=db
ENV QUANTUM_EDGE_DATABASE_PASSWORD=${DB_PASSWORD}
```

### Kubernetes Secrets

```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: quantum-edge-secrets
type: Opaque
data:
  jwt-secret: bXktc3VwZXItc2VjcmV0LWtleQ==
  db-password: c2VjdXJlLWRhdGFiYXNlLXBhc3N3b3Jk
  encryption-key: bXktZW5jcnlwdGlvbi1rZXk=
```

```yaml
# kubernetes/deployment.yaml
env:
  - name: QUANTUM_EDGE_API_JWT_SECRET_KEY
    valueFrom:
      secretKeyRef:
        name: quantum-edge-secrets
        key: jwt-secret
  - name: QUANTUM_EDGE_DATABASE_PASSWORD
    valueFrom:
      secretKeyRef:
        name: quantum-edge-secrets
        key: db-password
```

## Configuration Validation

The platform validates configuration on startup:

```python
from config.config_manager import ConfigManager

# Load and validate configuration
config_manager = ConfigManager()
config = config_manager.load_config('config.yaml')

# Validate configuration
errors = config_manager.validate_config(config)
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
    exit(1)

# Configuration is valid, proceed with initialization
```

### Validation Rules

```python
# config/validators.py
VALIDATION_RULES = {
    'system.max_workers': {
        'type': 'integer',
        'min': 1,
        'max': 32
    },
    'edge_runtime.max_memory_mb': {
        'type': 'integer',
        'min': 64,
        'max': 8192
    },
    'quantum_engine.n_qubits': {
        'type': 'integer',
        'min': 1,
        'max': 50
    },
    'privacy.default_epsilon': {
        'type': 'number',
        'min': 0.01,
        'max': 10.0
    },
    'api.port': {
        'type': 'integer',
        'min': 1024,
        'max': 65535
    }
}
```

## Dynamic Configuration

### Runtime Configuration Updates

```python
from config.config_manager import ConfigManager

config_manager = ConfigManager()

# Update configuration at runtime
config_manager.update_config({
    'edge_runtime.max_memory_mb': 1024,
    'quantum_engine.optimization_level': 3
})

# Configuration changes take effect immediately
# (may require restart for some settings)
```

### Feature Flags

```python
# Enable/disable features dynamically
config_manager.set_feature_flag('quantum_acceleration', True)
config_manager.set_feature_flag('federated_learning', False)

# Check feature status
if config_manager.is_feature_enabled('quantum_acceleration'):
    # Use quantum acceleration
    pass
```

## Security Configuration

### Encryption Keys

```yaml
# Secure key management
security:
  encryption_keys:
    primary:
      algorithm: "AES-256-GCM"
      key_id: "primary-key-2024"
      rotation_schedule: "90d"
    backup:
      algorithm: "AES-256-GCM"
      key_id: "backup-key-2024"
      rotation_schedule: "90d"
```

### Access Control

```yaml
security:
  access_control:
    enable_rbac: true
    default_role: "user"
    roles:
      admin:
        permissions: ["*"]
      user:
        permissions: ["inference.run", "models.list"]
      edge_device:
        permissions: ["inference.run", "federated.submit"]
```

### Audit Configuration

```yaml
security:
  audit:
    enabled: true
    log_level: "detailed"
    retention_days: 365
    storage_backend: "encrypted_s3"
    alert_on_suspicious_activity: true
    suspicious_patterns:
      - "unauthorized_access"
      - "data_exfiltration"
      - "configuration_tampering"
```

## Monitoring Configuration

### Prometheus Metrics

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: "/metrics"
    collect_system_metrics: true
    collect_application_metrics: true
    histogram_buckets: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
```

### Alerting Rules

```yaml
monitoring:
  alerting:
    rules:
      - name: "High CPU Usage"
        condition: "cpu_usage_percent > 85"
        duration: "5m"
        severity: "warning"
        channels: ["email", "slack"]
      - name: "Memory Pressure"
        condition: "memory_usage_percent > 90"
        duration: "2m"
        severity: "critical"
        channels: ["email", "pagerduty"]
      - name: "Quantum Engine Errors"
        condition: "quantum_errors_total > 10"
        duration: "1m"
        severity: "error"
        channels: ["email"]
```

## Deployment Configurations

### Docker Configuration

```yaml
deployment:
  docker:
    base_image: "python:3.9-slim"
    build_context: "."
    dockerfile: "Dockerfile"
    ports:
      - "8000:8000"
    volumes:
      - "./models:/app/models"
      - "./logs:/app/logs"
    environment:
      - "QUANTUM_EDGE_ENVIRONMENT=production"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: "30s"
      timeout: "10s"
      retries: 3
```

### Kubernetes Configuration

```yaml
deployment:
  kubernetes:
    namespace: "quantum-edge"
    replicas: 3
    image: "quantum-edge-ai/platform:v1.0.0"
    resources:
      requests:
        memory: "512Mi"
        cpu: "500m"
      limits:
        memory: "2Gi"
        cpu: "2000m"
    service:
      type: "LoadBalancer"
      ports:
        - port: 80
          targetPort: 8000
    ingress:
      enabled: true
      host: "api.quantum-edge-ai.com"
      tls:
        enabled: true
        secretName: "quantum-edge-tls"
```

## Configuration Examples

### Minimal Configuration

```yaml
version: "1.0"
environment: "development"

edge_runtime:
  max_memory_mb: 512

quantum_engine:
  n_qubits: 4

api:
  port: 8000
```

### High-Performance Configuration

```yaml
version: "1.0"
environment: "production"

system:
  max_workers: 16

edge_runtime:
  max_memory_mb: 4096
  default_precision: "FP32"
  compression_level: "none"

quantum_engine:
  n_qubits: 20
  optimization_level: 3
  enable_error_mitigation: true

api:
  rate_limit_requests: 10000

deployment:
  kubernetes_enabled: true
  min_replicas: 5
  max_replicas: 20
```

### Edge Device Configuration

```yaml
version: "1.0"
environment: "edge"

system:
  max_workers: 1

edge_runtime:
  max_memory_mb: 128
  default_precision: "INT8"
  cache_enabled: false

quantum_engine:
  n_qubits: 2
  max_circuit_depth: 10

federated_learning:
  client_selection_strategy: "edge_priority"
```

## Configuration Management

### Configuration Backup

```python
from config.config_manager import ConfigManager

config_manager = ConfigManager()

# Create backup
backup_path = config_manager.create_backup()
print(f"Configuration backed up to: {backup_path}")

# Restore from backup
config_manager.restore_backup(backup_path)
```

### Configuration Migration

```python
# Migrate from old configuration format
from config.migration import ConfigMigrator

migrator = ConfigMigrator()
migrator.migrate_config('config_old.yaml', 'config_new.yaml')
```

### Configuration Templates

```python
from config.templates import ConfigTemplates

templates = ConfigTemplates()

# Generate configuration for specific use case
edge_config = templates.generate_edge_config()
prod_config = templates.generate_production_config()
dev_config = templates.generate_development_config()
```

## Troubleshooting Configuration Issues

### Common Configuration Errors

1. **Invalid YAML/JSON Syntax**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

2. **Missing Required Fields**
   ```python
   # Check for missing required fields
   from config.validators import ConfigValidator
   validator = ConfigValidator()
   missing = validator.check_required_fields(config)
   ```

3. **Invalid Value Ranges**
   ```python
   # Validate value ranges
   errors = validator.validate_ranges(config)
   ```

4. **Environment Variable Conflicts**
   ```bash
   # List all QUANTUM_EDGE environment variables
   env | grep QUANTUM_EDGE
   ```

### Debug Configuration Loading

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from config.config_manager import ConfigManager

# Enable debug logging for configuration loading
config_manager = ConfigManager(debug=True)
config = config_manager.load_config()

# Print resolved configuration values
config_manager.print_resolved_config()
```

## Best Practices

1. **Use Environment Variables for Secrets**: Never store sensitive information in config files
2. **Validate Configurations**: Always validate configurations before deployment
3. **Use Configuration Templates**: Start with templates and customize as needed
4. **Document Custom Configurations**: Keep track of non-standard configurations
5. **Version Control Configurations**: Store configurations in version control (excluding secrets)
6. **Test Configurations**: Test configurations in staging before production deployment
7. **Monitor Configuration Changes**: Track who changed what and when
8. **Backup Configurations**: Regularly backup working configurations

## CLI Configuration

Override any configuration from command line:

```bash
# Run with custom configuration
python -m quantum_edge_ai --config config.prod.yaml --log-level DEBUG

# Override specific values
python -m quantum_edge_ai --quantum-engine.n-qubits 8 --api.port 9000

# Use environment-specific config
python -m quantum_edge_ai --env production
```

### Available CLI Options

```
--config FILE              Configuration file path
--env ENV                  Environment (development, production, edge)
--log-level LEVEL          Logging level (DEBUG, INFO, WARNING, ERROR)
--host HOST                API server host
--port PORT                API server port
--max-memory INT           Maximum memory in MB
--n-qubits INT             Number of qubits for quantum computations
--enable-debug             Enable debug mode
--disable-privacy          Disable privacy features (not recommended)
--enable-gpu               Enable GPU acceleration
```

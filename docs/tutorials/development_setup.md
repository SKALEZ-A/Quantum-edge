# Development Setup Guide

Complete guide for setting up a development environment for the Quantum Edge AI Platform.

## Prerequisites

Before setting up the development environment, ensure you have the following:

### System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL2)
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: 10GB free space for dependencies and data
- **Network**: Stable internet connection for downloading dependencies

### Required Software

```bash
# Python 3.8+
python --version

# pip (Python package manager)
pip --version

# git (version control)
git --version

# Docker (optional, for containerized development)
docker --version

# Node.js (optional, for frontend development)
node --version
npm --version
```

## Quick Start Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/quantum-edge-ai-platform.git
cd quantum-edge-ai-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

**Essential environment variables:**

```bash
# Database
QUANTUM_EDGE_DATABASE_URL=sqlite:///dev.db

# API Keys (get from respective services)
QUANTUM_EDGE_OPENAI_API_KEY=your_openai_key
QUANTUM_EDGE_IBM_QUANTUM_TOKEN=your_ibm_token

# Development settings
QUANTUM_EDGE_DEBUG=true
QUANTUM_EDGE_LOG_LEVEL=DEBUG
```

### 3. Initialize Database

```bash
# Initialize database
python -m quantum_edge_ai db init
python -m quantum_edge_ai db migrate

# (Optional) Load sample data
python -m quantum_edge_ai db seed
```

### 4. Run Development Server

```bash
# Start the development server
python -m quantum_edge_ai run --host 0.0.0.0 --port 8000 --reload

# Open browser to http://localhost:8000
```

## Detailed Setup Instructions

### Python Environment Setup

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Verify activation
which python  # Should point to venv/bin/python
```

#### Using conda

```bash
# Create conda environment
conda create -n quantum-edge python=3.9
conda activate quantum-edge

# Install pip in conda environment
conda install pip
```

### Dependency Installation

#### Core Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install optional dependencies
pip install -r requirements-optional.txt
```

#### Manual Installation (Advanced)

```bash
# Install quantum computing libraries
pip install qiskit qiskit-aer qiskit-ibm-runtime

# Install machine learning libraries
pip install scikit-learn tensorflow torch

# Install web framework
pip install fastapi uvicorn

# Install database libraries
pip install sqlalchemy alembic

# Install testing libraries
pip install pytest pytest-cov pytest-asyncio

# Install development tools
pip install black isort flake8 mypy pre-commit
```

### Configuration Files

#### Environment Configuration

Create `.env` file in the project root:

```bash
# Application Settings
QUANTUM_EDGE_DEBUG=true
QUANTUM_EDGE_ENVIRONMENT=development
QUANTUM_EDGE_LOG_LEVEL=DEBUG
QUANTUM_EDGE_SECRET_KEY=your-super-secret-key-change-this

# API Configuration
QUANTUM_EDGE_API_HOST=localhost
QUANTUM_EDGE_API_PORT=8000
QUANTUM_EDGE_API_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Database Configuration
QUANTUM_EDGE_DATABASE_URL=sqlite:///dev.db
QUANTUM_EDGE_DATABASE_ECHO=false

# Quantum Computing
QUANTUM_EDGE_IBMQ_TOKEN=your_ibm_quantum_token
QUANTUM_EDGE_QISKIT_BACKEND=ibmq_qasm_simulator

# External Services
QUANTUM_EDGE_REDIS_URL=redis://localhost:6379
QUANTUM_EDGE_PROMETHEUS_URL=http://localhost:9090

# Security
QUANTUM_EDGE_JWT_SECRET_KEY=your-jwt-secret-key
QUANTUM_EDGE_JWT_ALGORITHM=HS256
QUANTUM_EDGE_JWT_EXPIRATION_HOURS=24

# Privacy & Security
QUANTUM_EDGE_PRIVACY_EPSILON=0.5
QUANTUM_EDGE_DIFFERENTIAL_PRIVACY_ENABLED=true
QUANTUM_EDGE_ENCRYPTION_KEY=your-encryption-key-32-chars

# Federated Learning
QUANTUM_EDGE_FEDERATED_SERVER_URL=http://localhost:8001
QUANTUM_EDGE_MIN_CLIENTS_PER_ROUND=3
QUANTUM_EDGE_MAX_TRAINING_ROUNDS=100
```

#### Configuration Validation

```bash
# Validate configuration
python -c "from config.config_manager import ConfigManager; cm = ConfigManager(); cm.validate_config(cm.load_config())"

# Check for missing environment variables
python -c "import os; required = ['QUANTUM_EDGE_SECRET_KEY', 'QUANTUM_EDGE_DATABASE_URL']; missing = [k for k in required if not os.getenv(k)]; print('Missing:', missing if missing else 'None')"
```

### Database Setup

#### SQLite (Development)

```bash
# Initialize SQLite database
python -c "
from quantum_edge_ai.db import init_db
init_db()
print('Database initialized')
"
```

#### PostgreSQL (Production-like)

```bash
# Install PostgreSQL
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL service
# Ubuntu/Debian
sudo systemctl start postgresql

# macOS
brew services start postgresql

# Create database and user
sudo -u postgres psql
```

```sql
CREATE DATABASE quantum_edge_dev;
CREATE USER quantum_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE quantum_edge_dev TO quantum_user;
\q
```

Update `.env`:
```bash
QUANTUM_EDGE_DATABASE_URL=postgresql://quantum_user:secure_password@localhost/quantum_edge_dev
```

#### Database Migration

```bash
# Create migration
python -m quantum_edge_ai db revision --autogenerate -m "Initial migration"

# Run migrations
python -m quantum_edge_ai db upgrade

# Check migration status
python -m quantum_edge_ai db current
python -m quantum_edge_ai db history
```

### External Services Setup

#### Redis (Caching & Sessions)

```bash
# Install Redis
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
# Ubuntu/Debian
sudo systemctl start redis-server

# macOS
brew services start redis

# Verify Redis is running
redis-cli ping  # Should respond with PONG
```

#### Prometheus & Grafana (Monitoring)

```bash
# Install Prometheus
# Ubuntu/Debian
sudo apt-get install prometheus

# macOS
brew install prometheus

# Install Grafana
# Ubuntu/Debian
sudo apt-get install grafana

# macOS
brew install grafana

# Start services
# Ubuntu/Debian
sudo systemctl start prometheus
sudo systemctl start grafana-server

# macOS
brew services start prometheus
brew services start grafana
```

### Quantum Computing Setup

#### IBM Quantum Experience

1. **Create IBM Quantum Account**
   - Go to [IBM Quantum Experience](https://quantum-computing.ibm.com/)
   - Create a free account
   - Generate an API token

2. **Configure Qiskit**
   ```bash
   # Save IBM Quantum token
   echo "your_ibm_quantum_token" > ~/.qiskit/ibmq_token

   # Or set environment variable
   export QUANTUM_EDGE_IBMQ_TOKEN=your_ibm_quantum_token
   ```

3. **Test Quantum Connection**
   ```python
   from qiskit import IBMQ
   IBMQ.load_account()
   provider = IBMQ.get_provider()
   print("Available backends:", provider.backends())
   ```

#### Local Quantum Simulator

```bash
# Install Qiskit Aer (local simulator)
pip install qiskit-aer

# Test simulator
python -c "
from qiskit import QuantumCircuit, Aer, execute
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend, shots=1024).result()
print(result.get_counts())
"
```

### Development Tools Setup

#### Code Formatting & Linting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install development tools
pip install black isort flake8 mypy

# Format code
black .
isort .

# Lint code
flake8 .

# Type check
mypy .
```

#### Testing Setup

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock

# Run tests
pytest

# Run tests with coverage
pytest --cov=quantum_edge_ai --cov-report=html

# Run specific test file
pytest tests/test_quantum_algorithms.py

# Run tests in verbose mode
pytest -v

# Run tests with debugging
pytest --pdb
```

#### Documentation Setup

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html

# Serve documentation locally
cd build/html
python -m http.server 8080
# Open http://localhost:8080
```

### IDE Setup

#### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml"
    }
}
```

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Quantum Edge AI",
            "type": "python",
            "request": "launch",
            "module": "quantum_edge_ai",
            "args": ["run", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            "env": {
                "QUANTUM_EDGE_DEBUG": "true"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

#### PyCharm Configuration

1. **Set Project Interpreter**
   - File → Settings → Project → Python Interpreter
   - Add Interpreter → Existing → Select `venv/bin/python`

2. **Configure Testing**
   - File → Settings → Tools → Python Integrated Tools
   - Testing → Default test runner → pytest

3. **Enable Code Formatting**
   - File → Settings → Tools → External Tools
   - Add tools for black, isort, flake8

### Docker Development Setup

#### Development with Docker Compose

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - /app/venv
    environment:
      - QUANTUM_EDGE_DEBUG=true
      - QUANTUM_EDGE_DATABASE_URL=postgresql://user:pass@db:5432/quantum_edge_dev
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: quantum_edge_dev
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin

volumes:
  postgres_data:
```

Create `Dockerfile.dev`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

EXPOSE 8000

CMD ["python", "-m", "quantum_edge_ai", "run", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**Run development environment:**

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f app

# Run tests in container
docker-compose -f docker-compose.dev.yml exec app pytest

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Troubleshooting Setup Issues

#### Common Issues

1. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"

   # Reinstall package in development mode
   pip install -e .
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connection
   python -c "from quantum_edge_ai.db import get_db; db = get_db(); print('Connected')"

   # Check database URL
   echo $QUANTUM_EDGE_DATABASE_URL
   ```

3. **Quantum Backend Issues**
   ```bash
   # Check Qiskit installation
   python -c "import qiskit; print(qiskit.__version__)"

   # Test IBM Quantum connection
   python -c "from qiskit import IBMQ; IBMQ.load_account(); print('Connected')"
   ```

4. **Memory Issues**
   ```bash
   # Check available memory
   free -h  # Linux
   vm_stat  # macOS

   # Reduce quantum circuit size in config
   QUANTUM_EDGE_QUANTUM_ENGINE_N_QUBITS=2
   ```

5. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :8000  # Linux/macOS
   netstat -ano | findstr :8000  # Windows

   # Change port
   export QUANTUM_EDGE_API_PORT=8001
   ```

### Performance Optimization

#### Development Performance Tips

1. **Use lightweight quantum simulator for development**
   ```python
   # In config
   QUANTUM_EDGE_QISKIT_BACKEND=qasm_simulator  # Instead of ibmq backend
   ```

2. **Enable development caching**
   ```python
   # In config
   QUANTUM_EDGE_CACHE_ENABLED=true
   QUANTUM_EDGE_CACHE_BACKEND=redis
   ```

3. **Use smaller datasets for testing**
   ```python
   # Reduce dataset size for development
   QUANTUM_EDGE_MAX_TRAINING_SAMPLES=1000
   ```

4. **Enable async processing**
   ```python
   # Enable async for better performance
   QUANTUM_EDGE_ASYNC_PROCESSING=true
   ```

### Security Considerations

#### Development Security

1. **Never commit secrets**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo "secrets/" >> .gitignore
   ```

2. **Use development certificates**
   ```bash
   # Generate self-signed certificate for HTTPS development
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   ```

3. **Enable security headers in development**
   ```python
   # In config
   QUANTUM_EDGE_SECURITY_HEADERS_ENABLED=true
   QUANTUM_EDGE_CSP_ENABLED=false  # Disable CSP for development
   ```

### Next Steps

After completing the setup:

1. **Run the test suite** to ensure everything works
2. **Read the API documentation** to understand the platform
3. **Try the example notebooks** in `docs/notebooks/`
4. **Check the monitoring dashboard** at http://localhost:3000
5. **Join the development community** and contribute

For additional help, check the [troubleshooting guide](../troubleshooting.md) or create an issue on GitHub.

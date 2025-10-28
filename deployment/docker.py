"""
Quantum Edge AI Platform - Docker Deployment

Docker containerization for quantum edge AI platform including
multi-stage builds, optimization, and orchestration.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DockerBuildContext:
    """Docker build context configuration"""
    build_path: str = "."
    dockerfile_path: str = "Dockerfile"
    build_args: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    target_stage: Optional[str] = None
    no_cache: bool = False
    pull: bool = True

@dataclass
class DockerImage:
    """Docker image configuration"""
    name: str
    tag: str = "latest"
    registry: Optional[str] = None
    architecture: str = "amd64"
    os: str = "linux"

    @property
    def full_name(self) -> str:
        """Get full image name with registry"""
        registry_prefix = f"{self.registry}/" if self.registry else ""
        return f"{registry_prefix}{self.name}:{self.tag}"

    @property
    def manifest_name(self) -> str:
        """Get manifest name for multi-arch images"""
        registry_prefix = f"{self.registry}/" if self.registry else ""
        return f"{registry_prefix}{self.name}:{self.tag}"

@dataclass
class DockerContainer:
    """Docker container configuration"""
    name: str
    image: DockerImage
    ports: Dict[int, int] = field(default_factory=dict)  # host:container
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)  # host:container
    networks: List[str] = field(default_factory=list)
    restart_policy: str = "unless-stopped"
    memory_limit: Optional[str] = None  # e.g., "512m"
    cpu_limit: Optional[float] = None
    health_check: Optional[Dict[str, Any]] = None
    labels: Dict[str, str] = field(default_factory=dict)
    command: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None

@dataclass
class DockerComposeService:
    """Docker Compose service configuration"""
    name: str
    container: DockerContainer
    depends_on: List[str] = field(default_factory=list)
    profiles: List[str] = field(default_factory=list)
    deploy: Optional[Dict[str, Any]] = None

class DockerDeployment:
    """Docker deployment manager"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)

    def create_dockerfile(self, output_path: str = "Dockerfile",
                         base_image: str = "python:3.9-slim",
                         multi_stage: bool = True,
                         gpu_support: bool = False) -> str:
        """Create optimized Dockerfile"""

        dockerfile_content = []

        if multi_stage:
            # Multi-stage build for optimization
            dockerfile_content.extend(self._create_multi_stage_dockerfile(base_image, gpu_support))
        else:
            # Single-stage build
            dockerfile_content.extend(self._create_single_stage_dockerfile(base_image, gpu_support))

        dockerfile_path = self.project_root / output_path
        with open(dockerfile_path, 'w') as f:
            f.write('\n'.join(dockerfile_content))

        logger.info(f"Created Dockerfile at {dockerfile_path}")
        return str(dockerfile_path)

    def _create_multi_stage_dockerfile(self, base_image: str, gpu_support: bool) -> List[str]:
        """Create multi-stage Dockerfile"""
        lines = []

        # Build stage
        lines.extend([
            "# Multi-stage build for Quantum Edge AI Platform",
            f"FROM {base_image} AS builder",
            "",
            "# Install build dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    build-essential \\",
            "    libssl-dev \\",
            "    libffi-dev \\",
            "    python3-dev \\",
            "    git \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Create virtual environment",
            "RUN python -m venv /opt/venv",
            "ENV PATH=\"/opt/venv/bin:$PATH\"",
            "",
            "# Copy requirements and install Python dependencies",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir --upgrade pip && \\",
            "    pip install --no-cache-dir -r requirements.txt",
            "",
            "# Production stage",
            f"FROM {'nvidia/cuda:11.8-runtime-ubuntu20.04' if gpu_support else 'python:3.9-slim'} AS production",
            "",
            "# Install runtime dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    libgomp1 \\",
            "    libssl1.1 \\",
            "    ca-certificates \\",
            "    curl \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Create non-root user",
            "RUN groupadd -r quantum && useradd -r -g quantum quantum",
            "",
            "# Copy virtual environment from builder",
            "COPY --from=builder /opt/venv /opt/venv",
            "ENV PATH=\"/opt/venv/bin:$PATH\"",
            "",
            "# Set working directory",
            "WORKDIR /app",
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "# Change ownership to non-root user",
            "RUN chown -R quantum:quantum /app",
            "USER quantum",
            "",
            "# Health check",
            "HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\",
            "    CMD python -c \"import requests; requests.get('http://localhost:8080/health')\" 2>/dev/null || exit 1",
            "",
            "# Expose port",
            "EXPOSE 8080",
            "",
            "# Set environment variables",
            "ENV PYTHONUNBUFFERED=1",
            "ENV QUANTUM_EDGE_ENV=production",
            "",
            "# Run application",
            "CMD [\"python\", \"-m\", \"quantum_edge_ai.main\"]"
        ])

        return lines

    def _create_single_stage_dockerfile(self, base_image: str, gpu_support: bool) -> List[str]:
        """Create single-stage Dockerfile"""
        gpu_base = "nvidia/cuda:11.8-runtime-ubuntu20.04" if gpu_support else base_image

        lines = [
            f"FROM {gpu_base}",
            "",
            "# Install system dependencies",
            "RUN apt-get update && apt-get install -y \\",
            "    build-essential \\",
            "    libssl-dev \\",
            "    libffi-dev \\",
            "    python3-dev \\",
            "    python3-pip \\",
            "    libgomp1 \\",
            "    curl \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "# Create non-root user",
            "RUN groupadd -r quantum && useradd -r -g quantum quantum",
            "",
            "# Set working directory",
            "WORKDIR /app",
            "",
            "# Copy requirements first for better caching",
            "COPY requirements.txt .",
            "RUN pip install --no-cache-dir --upgrade pip && \\",
            "    pip install --no-cache-dir -r requirements.txt",
            "",
            "# Copy application code",
            "COPY . .",
            "",
            "# Change ownership",
            "RUN chown -R quantum:quantum /app",
            "USER quantum",
            "",
            "# Health check",
            "HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\",
            "    CMD curl -f http://localhost:8080/health || exit 1",
            "",
            "# Expose port",
            "EXPOSE 8080",
            "",
            "# Set environment",
            "ENV PYTHONUNBUFFERED=1",
            "ENV QUANTUM_EDGE_ENV=production",
            "",
            "# Run application",
            "CMD [\"python\", \"-m\", \"quantum_edge_ai.main\"]"
        ]

        return lines

    def create_requirements_file(self, output_path: str = "requirements.txt",
                               include_dev: bool = False) -> str:
        """Create requirements.txt file"""

        core_requirements = [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "requests>=2.25.0",
            "aiohttp>=3.8.0",
            "websockets>=10.0",
            "pydantic>=1.8.0",
            "python-dotenv>=0.19.0",
            "pyyaml>=6.0",
            "prometheus-client>=0.14.0",
            "psutil>=5.8.0",
            "cryptography>=36.0.0",
            "PyJWT>=2.0.0",
            "bcrypt>=3.2.0",
            "sqlalchemy>=1.4.0",
            "alembic>=1.7.0",
            "redis>=4.0.0",
            "celery>=5.2.0",
            "flower>=1.0.0"
        ]

        quantum_requirements = [
            "qiskit>=0.36.0",
            "qiskit-aer>=0.11.0",
            "qiskit-ibmq-provider>=0.19.0",
            "qiskit-optimization>=0.4.0",
            "pennylane>=0.24.0",
            "pennylane-qiskit>=0.24.0",
            "cirq>=1.0.0",
            "tensorflow-quantum>=0.7.0"
        ]

        edge_ai_requirements = [
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
            "torchvision>=0.12.0",
            "torchaudio>=0.11.0",
            "onnx>=1.11.0",
            "onnxruntime>=1.11.0",
            "tflite-runtime>=2.8.0",
            "openvino>=2022.1.0"
        ]

        federated_requirements = [
            "torchvision>=0.12.0",
            "syft>=0.7.0",
            "opacus>=1.0.0"
        ]

        dev_requirements = [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.9.0"
        ] if include_dev else []

        all_requirements = (
            core_requirements +
            quantum_requirements +
            edge_ai_requirements +
            federated_requirements +
            dev_requirements
        )

        requirements_path = self.project_root / output_path
        with open(requirements_path, 'w') as f:
            f.write('# Quantum Edge AI Platform Requirements\n')
            f.write('# Generated automatically\n\n')
            for req in sorted(all_requirements):
                f.write(f'{req}\n')

        logger.info(f"Created requirements.txt at {requirements_path}")
        return str(requirements_path)

    def create_dockerignore(self, output_path: str = ".dockerignore") -> str:
        """Create .dockerignore file"""

        ignore_patterns = [
            "# Version control",
            ".git",
            ".gitignore",
            ".gitmodules",
            "",
            "# Python",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "env",
            "venv",
            ".venv",
            "ENV",
            "env.bak",
            "venv.bak",
            "pip-log.txt",
            "pip-delete-this-directory.txt",
            "",
            "# Testing",
            ".coverage",
            ".pytest_cache",
            ".tox",
            "htmlcov",
            ".mypy_cache",
            "",
            "# IDE",
            ".vscode",
            ".idea",
            "*.swp",
            "*.swo",
            "",
            "# OS",
            ".DS_Store",
            "Thumbs.db",
            "",
            "# Documentation",
            "docs/_build",
            "*.md",
            "",
            "# Logs",
            "*.log",
            "logs",
            "",
            "# Temporary files",
            "*.tmp",
            "*.temp",
            "",
            "# Build artifacts",
            "build",
            "dist",
            "*.egg-info",
            "",
            "# Node.js (if any)",
            "node_modules",
            "npm-debug.log*",
            "yarn-debug.log*",
            "yarn-error.log*",
            "",
            "# Quantum Edge specific",
            ".quantum_cache",
            "models/cache",
            "data/raw",
            "data/processed",
            "experiments",
            "",
            "# Deployment",
            "docker-compose*.yml",
            "kubernetes",
            ".env*"
        ]

        dockerignore_path = self.project_root / output_path
        with open(dockerignore_path, 'w') as f:
            f.write('\n'.join(ignore_patterns))

        logger.info(f"Created .dockerignore at {dockerignore_path}")
        return str(dockerignore_path)

    def build_image(self, image: DockerImage, context: DockerBuildContext,
                   push: bool = False) -> bool:
        """Build Docker image"""
        try:
            cmd = ["docker", "build"]

            # Add build arguments
            for key, value in context.build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add labels
            for key, value in context.labels.items():
                cmd.extend(["--label", f"{key}={value}"])

            # Add target stage
            if context.target_stage:
                cmd.extend(["--target", context.target_stage])

            # Add other options
            if context.no_cache:
                cmd.append("--no-cache")
            if context.pull:
                cmd.append("--pull")

            # Add tag and context
            cmd.extend(["-t", image.full_name, str(context.build_path)])

            logger.info(f"Building Docker image: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully built image: {image.full_name}")

                if push:
                    return self.push_image(image)

                return True
            else:
                logger.error(f"Failed to build image: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error building image: {str(e)}")
            return False

    def push_image(self, image: DockerImage) -> bool:
        """Push Docker image to registry"""
        try:
            cmd = ["docker", "push", image.full_name]

            logger.info(f"Pushing Docker image: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully pushed image: {image.full_name}")
                return True
            else:
                logger.error(f"Failed to push image: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error pushing image: {str(e)}")
            return False

    def run_container(self, container: DockerContainer, detached: bool = True) -> Optional[str]:
        """Run Docker container"""
        try:
            cmd = ["docker", "run"]

            if detached:
                cmd.append("-d")

            # Add name
            cmd.extend(["--name", container.name])

            # Add ports
            for host_port, container_port in container.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])

            # Add environment variables
            for key, value in container.environment.items():
                cmd.extend(["-e", f"{key}={value}"])

            # Add volumes
            for host_path, container_path in container.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

            # Add networks
            for network in container.networks:
                cmd.extend(["--network", network])

            # Add restart policy
            cmd.extend(["--restart", container.restart_policy])

            # Add resource limits
            if container.memory_limit:
                cmd.extend(["-m", container.memory_limit])
            if container.cpu_limit:
                cmd.extend(["--cpus", str(container.cpu_limit)])

            # Add labels
            for key, value in container.labels.items():
                cmd.extend(["-l", f"{key}={value}"])

            # Add health check
            if container.health_check:
                health_cmd = ["--health-cmd", container.health_check["test"]]
                if "interval" in container.health_check:
                    health_cmd.extend(["--health-interval", container.health_check["interval"]])
                if "timeout" in container.health_check:
                    health_cmd.extend(["--health-timeout", container.health_check["timeout"]])
                if "retries" in container.health_check:
                    health_cmd.extend(["--health-retries", str(container.health_check["retries"])])
                cmd.extend(health_cmd)

            # Add command/entrypoint
            if container.entrypoint:
                cmd.extend(["--entrypoint", ' '.join(container.entrypoint)])

            # Add image
            cmd.append(container.image.full_name)

            # Add command
            if container.command:
                cmd.extend(container.command)

            logger.info(f"Running Docker container: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                container_id = result.stdout.strip()
                logger.info(f"Successfully started container: {container_id}")
                return container_id
            else:
                logger.error(f"Failed to start container: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error running container: {str(e)}")
            return None

class DockerComposeDeployment:
    """Docker Compose deployment manager"""

    def __init__(self, project_root: str = ".", project_name: str = "quantum-edge-ai"):
        self.project_root = Path(project_root)
        self.project_name = project_name
        self.services: Dict[str, DockerComposeService] = {}
        self.logger = logging.getLogger(__name__)

    def add_service(self, service: DockerComposeService):
        """Add service to compose deployment"""
        self.services[service.name] = service
        logger.info(f"Added service: {service.name}")

    def create_compose_file(self, output_path: str = "docker-compose.yml",
                           version: str = "3.8") -> str:
        """Create docker-compose.yml file"""

        compose_data = {
            "version": version,
            "services": {},
            "volumes": {},
            "networks": {}
        }

        # Add services
        for service_name, service in self.services.items():
            service_config = {
                "image": service.container.image.full_name,
                "container_name": f"{self.project_name}_{service_name}",
                "restart": service.container.restart_policy
            }

            # Add ports
            if service.container.ports:
                service_config["ports"] = [
                    f"{host}:{container}" for host, container in service.container.ports.items()
                ]

            # Add environment
            if service.container.environment:
                service_config["environment"] = service.container.environment

            # Add volumes
            if service.container.volumes:
                service_config["volumes"] = [
                    f"{host}:{container}" for host, container in service.container.volumes.items()
                ]

            # Add networks
            if service.container.networks:
                service_config["networks"] = service.container.networks

            # Add health check
            if service.container.health_check:
                service_config["healthcheck"] = service.container.health_check

            # Add resource limits
            if service.container.memory_limit or service.container.cpu_limit:
                deploy_config = {}
                limits = {}

                if service.container.memory_limit:
                    limits["memory"] = service.container.memory_limit
                if service.container.cpu_limit:
                    limits["cpus"] = str(service.container.cpu_limit)

                if limits:
                    deploy_config["resources"] = {"limits": limits}
                    service_config["deploy"] = deploy_config

            # Add depends_on
            if service.depends_on:
                service_config["depends_on"] = service.depends_on

            # Add profiles
            if service.profiles:
                service_config["profiles"] = service.profiles

            # Add deploy configuration
            if service.deploy:
                service_config.setdefault("deploy", {}).update(service.deploy)

            compose_data["services"][service_name] = service_config

        # Add default volumes
        compose_data["volumes"] = {
            "quantum-edge-data": {
                "driver": "local"
            },
            "quantum-edge-models": {
                "driver": "local"
            }
        }

        # Add networks
        compose_data["networks"] = {
            "quantum-edge-network": {
                "driver": "bridge"
            },
            "federated-network": {
                "driver": "bridge",
                "internal": True
            }
        }

        # Write compose file
        compose_path = self.project_root / output_path
        with open(compose_path, 'w') as f:
            import yaml
            yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created docker-compose.yml at {compose_path}")
        return str(compose_path)

    def create_override_file(self, output_path: str = "docker-compose.override.yml",
                           environment: str = "development") -> str:
        """Create docker-compose override file"""

        override_data = {
            "version": "3.8",
            "services": {}
        }

        # Add development overrides
        for service_name in self.services.keys():
            service_override = {}

            if environment == "development":
                # Development-specific settings
                service_override["environment"] = [
                    "QUANTUM_EDGE_LOG_LEVEL=DEBUG",
                    "QUANTUM_EDGE_DEBUG=true"
                ]
                service_override["volumes"] = [
                    "./:/app:ro",  # Mount source code for development
                    "/app/__pycache__",  # Exclude cache
                    "/app/.pytest_cache"
                ]
            elif environment == "testing":
                service_override["environment"] = [
                    "QUANTUM_EDGE_ENV=testing",
                    "QUANTUM_EDGE_DATABASE_URL=sqlite:///test.db"
                ]
                service_override["command"] = ["pytest", "/app/tests"]

            override_data["services"][service_name] = service_override

        # Write override file
        override_path = self.project_root / output_path
        with open(override_path, 'w') as f:
            import yaml
            yaml.dump(override_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created docker-compose override file at {override_path}")
        return str(override_path)

    def up(self, services: Optional[List[str]] = None, detached: bool = True) -> bool:
        """Start docker-compose services"""
        try:
            cmd = ["docker-compose", "up"]
            if detached:
                cmd.append("-d")
            if services:
                cmd.extend(services)

            logger.info(f"Starting docker-compose: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Successfully started docker-compose services")
                return True
            else:
                logger.error(f"Failed to start services: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error starting docker-compose: {str(e)}")
            return False

    def down(self, remove_volumes: bool = False, remove_images: bool = False) -> bool:
        """Stop docker-compose services"""
        try:
            cmd = ["docker-compose", "down"]
            if remove_volumes:
                cmd.append("-v")
            if remove_images:
                cmd.extend(["--rmi", "all"])

            logger.info(f"Stopping docker-compose: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Successfully stopped docker-compose services")
                return True
            else:
                logger.error(f"Failed to stop services: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error stopping docker-compose: {str(e)}")
            return False

    def build(self, services: Optional[List[str]] = None) -> bool:
        """Build docker-compose services"""
        try:
            cmd = ["docker-compose", "build"]
            if services:
                cmd.extend(services)

            logger.info(f"Building docker-compose services: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Successfully built docker-compose services")
                return True
            else:
                logger.error(f"Failed to build services: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error building docker-compose: {str(e)}")
            return False

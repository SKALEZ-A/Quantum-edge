"""
Quantum Edge AI Platform - Deployment Configuration

Comprehensive deployment configurations for Docker, Kubernetes, cloud platforms,
and edge device deployment strategies.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
import hashlib
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentType(Enum):
    """Deployment types"""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES_HELM = "kubernetes_helm"
    AWS_ECS = "aws_ecs"
    AWS_EKS = "aws_eks"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    GCP_GKE = "gcp_gke"
    AZURE_CONTAINER_INSTANCES = "azure_container_instances"
    AZURE_KUBERNETES = "azure_kubernetes"

class ScalingStrategy(Enum):
    """Auto-scaling strategies"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    CUSTOM_METRICS = "custom_metrics"
    SCHEDULED = "scheduled"

@dataclass
class ResourceLimits:
    """Resource limits for containers"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    storage_gb: int = 10
    network_mbps: int = 100

@dataclass
class HealthCheck:
    """Health check configuration"""
    path: str = "/health"
    port: int = 8080
    interval_seconds: int = 30
    timeout_seconds: int = 10
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    initial_delay_seconds: int = 30

@dataclass
class DockerConfig:
    """Docker deployment configuration"""

    # Basic settings
    image_name: str = "quantum-edge-ai"
    image_tag: str = "latest"
    registry: Optional[str] = None
    namespace: str = "quantum-edge"

    # Build configuration
    dockerfile_path: str = "Dockerfile"
    build_context: str = "."
    build_args: Dict[str, str] = field(default_factory=dict)

    # Runtime configuration
    ports: Dict[int, int] = field(default_factory=lambda: {8080: 8080})  # host:container
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)  # host:container
    networks: List[str] = field(default_factory=lambda: ["quantum-edge-network"])

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Health checks
    health_check: HealthCheck = field(default_factory=HealthCheck)

    # Security
    user: str = "1000:1000"
    security_opts: List[str] = field(default_factory=list)
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=list)

    # Logging
    log_driver: str = "json-file"
    log_options: Dict[str, str] = field(default_factory=lambda: {
        "max-size": "10m",
        "max-file": "3"
    })

    # Restart policy
    restart_policy: str = "unless-stopped"

    def to_dockerfile(self) -> str:
        """Generate Dockerfile content"""
        dockerfile = f"""FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libssl-dev \\
    libffi-dev \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE {list(self.ports.keys())[0]}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:{list(self.ports.keys())[0]}{self.health_check.path} || exit 1

# Run application
CMD ["python", "-m", "quantum_edge_ai.main"]
"""
        return dockerfile

    def to_docker_compose(self) -> Dict[str, Any]:
        """Generate docker-compose configuration"""
        services = {
            'quantum-edge-api': {
                'image': f"{self.image_name}:{self.image_tag}",
                'ports': [f"{host}:{container}" for host, container in self.ports.items()],
                'environment': self.environment,
                'volumes': [f"{host}:{container}" for host, container in self.volumes.items()],
                'networks': self.networks,
                'deploy': {
                    'resources': {
                        'limits': {
                            'cpus': str(self.resource_limits.cpu_cores),
                            'memory': f"{self.resource_limits.memory_mb}M"
                        }
                    },
                    'restart_policy': {
                        'condition': self.restart_policy
                    }
                },
                'healthcheck': {
                    'test': ["CMD", "curl", "-f", f"http://localhost:{list(self.ports.keys())[0]}{self.health_check.path}"],
                    'interval': f"{self.health_check.interval_seconds}s",
                    'timeout': f"{self.health_check.timeout_seconds}s",
                    'retries': self.health_check.unhealthy_threshold,
                    'start_period': f"{self.health_check.initial_delay_seconds}s"
                }
            }
        }

        # Add database service
        services['quantum-edge-db'] = {
            'image': 'postgres:13',
            'environment': {
                'POSTGRES_DB': 'quantum_edge',
                'POSTGRES_USER': 'quantum_user',
                'POSTGRES_PASSWORD': 'change_in_production'
            },
            'volumes': ['quantum-edge-data:/var/lib/postgresql/data'],
            'networks': self.networks
        }

        # Add Redis service
        services['quantum-edge-redis'] = {
            'image': 'redis:7-alpine',
            'networks': self.networks
        }

        return {
            'version': '3.8',
            'services': services,
            'volumes': {
                'quantum-edge-data': {}
            },
            'networks': {
                network: {} for network in self.networks
            }
        }

@dataclass
class KubernetesConfig:
    """Kubernetes deployment configuration"""

    # Basic settings
    name: str = "quantum-edge-ai"
    namespace: str = "quantum-edge"
    replicas: int = 1

    # Container configuration
    image: str = "quantum-edge-ai:latest"
    image_pull_policy: str = "Always"

    # Ports
    ports: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "http", "containerPort": 8080, "protocol": "TCP"}
    ])

    # Environment variables
    env: Dict[str, str] = field(default_factory=dict)

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Health checks
    liveness_probe: HealthCheck = field(default_factory=HealthCheck)
    readiness_probe: HealthCheck = field(default_factory=lambda: HealthCheck(path="/ready"))

    # Volumes
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    volume_mounts: List[Dict[str, Any]] = field(default_factory=list)

    # Service configuration
    service_type: str = "ClusterIP"
    service_ports: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "http", "port": 80, "targetPort": 8080, "protocol": "TCP"}
    ])

    # Ingress configuration
    ingress_enabled: bool = False
    ingress_host: Optional[str] = None
    ingress_tls: bool = False
    ingress_class: str = "nginx"

    # Auto-scaling
    autoscaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    scaling_strategy: ScalingStrategy = ScalingStrategy.CPU_UTILIZATION
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80

    # ConfigMaps and Secrets
    config_maps: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)

    # Node selectors and tolerations
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None

    def to_deployment_yaml(self) -> str:
        """Generate Kubernetes Deployment YAML"""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.name,
                'namespace': self.namespace,
                'labels': {
                    'app': self.name,
                    'component': 'api'
                }
            },
            'spec': {
                'replicas': self.replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.name,
                            'component': 'api'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.name,
                            'image': self.image,
                            'imagePullPolicy': self.image_pull_policy,
                            'ports': self.ports,
                            'env': [
                                {'name': k, 'value': v} for k, v in self.env.items()
                            ],
                            'resources': {
                                'limits': {
                                    'cpu': f"{self.resource_limits.cpu_cores}",
                                    'memory': f"{self.resource_limits.memory_mb}Mi"
                                },
                                'requests': {
                                    'cpu': f"{self.resource_limits.cpu_cores * 0.5}",
                                    'memory': f"{int(self.resource_limits.memory_mb * 0.5)}Mi"
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': self.liveness_probe.path,
                                    'port': self.liveness_probe.port
                                },
                                'initialDelaySeconds': self.liveness_probe.initial_delay_seconds,
                                'periodSeconds': self.liveness_probe.interval_seconds,
                                'timeoutSeconds': self.liveness_probe.timeout_seconds,
                                'failureThreshold': self.liveness_probe.unhealthy_threshold
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': self.readiness_probe.path,
                                    'port': self.readiness_probe.port
                                },
                                'initialDelaySeconds': self.readiness_probe.initial_delay_seconds,
                                'periodSeconds': self.readiness_probe.interval_seconds,
                                'timeoutSeconds': self.readiness_probe.timeout_seconds,
                                'failureThreshold': self.readiness_probe.unhealthy_threshold
                            }
                        }]
                    }
                }
            }
        }

        # Add volumes if specified
        if self.volumes or self.volume_mounts:
            deployment['spec']['template']['spec']['volumes'] = self.volumes
            deployment['spec']['template']['spec']['containers'][0]['volumeMounts'] = self.volume_mounts

        # Add node selector
        if self.node_selector:
            deployment['spec']['template']['spec']['nodeSelector'] = self.node_selector

        # Add tolerations
        if self.tolerations:
            deployment['spec']['template']['spec']['tolerations'] = self.tolerations

        # Add affinity
        if self.affinity:
            deployment['spec']['template']['spec']['affinity'] = self.affinity

        return yaml.dump(deployment, default_flow_style=False)

    def to_service_yaml(self) -> str:
        """Generate Kubernetes Service YAML"""
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.name}-service",
                'namespace': self.namespace,
                'labels': {
                    'app': self.name
                }
            },
            'spec': {
                'type': self.service_type,
                'selector': {
                    'app': self.name
                },
                'ports': self.service_ports
            }
        }

        return yaml.dump(service, default_flow_style=False)

    def to_hpa_yaml(self) -> str:
        """Generate HorizontalPodAutoscaler YAML"""
        if not self.autoscaling_enabled:
            return ""

        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.name}-hpa",
                'namespace': self.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.name
                },
                'minReplicas': self.min_replicas,
                'maxReplicas': self.max_replicas,
                'metrics': []
            }
        }

        # Add CPU utilization metric
        if self.scaling_strategy == ScalingStrategy.CPU_UTILIZATION:
            hpa['spec']['metrics'].append({
                'type': 'Resource',
                'resource': {
                    'name': 'cpu',
                    'target': {
                        'type': 'Utilization',
                        'averageUtilization': self.target_cpu_utilization
                    }
                }
            })

        # Add memory utilization metric
        if self.scaling_strategy == ScalingStrategy.MEMORY_UTILIZATION:
            hpa['spec']['metrics'].append({
                'type': 'Resource',
                'resource': {
                    'name': 'memory',
                    'target': {
                        'type': 'Utilization',
                        'averageUtilization': self.target_memory_utilization
                    }
                }
            })

        return yaml.dump(hpa, default_flow_style=False)

    def to_ingress_yaml(self) -> str:
        """Generate Ingress YAML"""
        if not self.ingress_enabled:
            return ""

        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{self.name}-ingress",
                'namespace': self.namespace,
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/'
                }
            },
            'spec': {
                'ingressClassName': self.ingress_class,
                'rules': [{
                    'host': self.ingress_host,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f"{self.name}-service",
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }

        # Add TLS if enabled
        if self.ingress_tls:
            ingress['spec']['tls'] = [{
                'hosts': [self.ingress_host],
                'secretName': f"{self.name}-tls"
            }]

        return yaml.dump(ingress, default_flow_style=False)

@dataclass
class DeploymentConfig:
    """Main deployment configuration"""

    # Deployment type
    deployment_type: DeploymentType = DeploymentType.STANDALONE

    # Environment
    environment: str = "development"

    # Docker configuration
    docker: DockerConfig = field(default_factory=DockerConfig)

    # Kubernetes configuration
    kubernetes: KubernetesConfig = field(default_factory=KubernetesConfig)

    # Cloud-specific configurations
    cloud_config: Dict[str, Any] = field(default_factory=dict)

    # Deployment metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Deployment settings
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_tracing: bool = False
    enable_metrics: bool = True

    # Backup and recovery
    enable_backups: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30

    # Rollback configuration
    enable_rollback: bool = True
    max_rollback_versions: int = 5

    def generate_deployment_files(self, output_dir: str = "deployment"):
        """Generate all deployment files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if self.deployment_type == DeploymentType.DOCKER:
            self._generate_docker_deployment(output_path)
        elif self.deployment_type == DeploymentType.KUBERNETES:
            self._generate_kubernetes_deployment(output_path)
        elif self.deployment_type == DeploymentType.DOCKER_COMPOSE:
            self._generate_docker_compose_deployment(output_path)

    def _generate_docker_deployment(self, output_path: Path):
        """Generate Docker deployment files"""
        # Dockerfile
        dockerfile_path = output_path / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(self.docker.to_dockerfile())

        # .dockerignore
        dockerignore_path = output_path / ".dockerignore"
        with open(dockerignore_path, 'w') as f:
            f.write(self._generate_dockerignore())

        # Build script
        build_script = output_path / "build.sh"
        with open(build_script, 'w') as f:
            f.write(self._generate_docker_build_script())

        build_script.chmod(0o755)

    def _generate_kubernetes_deployment(self, output_path: Path):
        """Generate Kubernetes deployment files"""
        # Deployment
        deployment_path = output_path / "deployment.yaml"
        with open(deployment_path, 'w') as f:
            f.write(self.kubernetes.to_deployment_yaml())

        # Service
        service_path = output_path / "service.yaml"
        with open(service_path, 'w') as f:
            f.write(self.kubernetes.to_service_yaml())

        # HPA
        if self.kubernetes.autoscaling_enabled:
            hpa_path = output_path / "hpa.yaml"
            with open(hpa_path, 'w') as f:
                f.write(self.kubernetes.to_hpa_yaml())

        # Ingress
        if self.kubernetes.ingress_enabled:
            ingress_path = output_path / "ingress.yaml"
            with open(ingress_path, 'w') as f:
                f.write(self.kubernetes.to_ingress_yaml())

        # ConfigMap
        configmap_path = output_path / "configmap.yaml"
        with open(configmap_path, 'w') as f:
            f.write(self._generate_configmap_yaml())

    def _generate_docker_compose_deployment(self, output_path: Path):
        """Generate Docker Compose deployment files"""
        compose_data = self.docker.to_docker_compose()

        compose_path = output_path / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False)

        # Environment file
        env_path = output_path / ".env"
        with open(env_path, 'w') as f:
            for key, value in self.docker.environment.items():
                f.write(f"{key}={value}\n")

    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore content"""
        return """.git
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt
.coverage
.pytest_cache
.mypy_cache
.vscode
.idea
*.egg-info
dist
build
.DS_Store
*.log
"""

    def _generate_docker_build_script(self) -> str:
        """Generate Docker build script"""
        return f"""#!/bin/bash

# Quantum Edge AI Platform - Docker Build Script

set -e

IMAGE_NAME={self.docker.image_name}
IMAGE_TAG={self.docker.image_tag}
REGISTRY={self.docker.registry or ''}

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Tag for registry if specified
if [ -n "$REGISTRY" ]; then
    docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY/$IMAGE_NAME:$IMAGE_TAG
fi

# Push to registry if specified
if [ -n "$REGISTRY" ]; then
    echo "Pushing to registry..."
    docker push $REGISTRY/$IMAGE_NAME:$IMAGE_TAG
fi

echo "Build completed successfully!"
"""

    def _generate_configmap_yaml(self) -> str:
        """Generate ConfigMap YAML"""
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.kubernetes.name}-config",
                'namespace': self.kubernetes.namespace
            },
            'data': {
                'config.json': json.dumps({
                    'version': self.version,
                    'environment': self.environment,
                    'enable_monitoring': self.enable_monitoring,
                    'enable_logging': self.enable_logging
                }, indent=2)
            }
        }

        return yaml.dump(configmap, default_flow_style=False)

    def validate_deployment(self) -> List[str]:
        """Validate deployment configuration"""
        errors = []

        # Validate Docker config
        if self.deployment_type in [DeploymentType.DOCKER, DeploymentType.DOCKER_COMPOSE]:
            if not self.docker.image_name:
                errors.append("Docker image name is required")
            if self.docker.resource_limits.memory_mb < 128:
                errors.append("Docker memory limit too low")

        # Validate Kubernetes config
        if self.deployment_type == DeploymentType.KUBERNETES:
            if not self.kubernetes.name:
                errors.append("Kubernetes deployment name is required")
            if self.kubernetes.replicas < 1:
                errors.append("Kubernetes replicas must be at least 1")

        return errors

class DeploymentManager:
    """Deployment manager for handling deployments"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def deploy(self, target: str = "local") -> bool:
        """Deploy the application"""
        try:
            if self.config.deployment_type == DeploymentType.DOCKER:
                return self._deploy_docker(target)
            elif self.config.deployment_type == DeploymentType.KUBERNETES:
                return self._deploy_kubernetes(target)
            elif self.config.deployment_type == DeploymentType.DOCKER_COMPOSE:
                return self._deploy_docker_compose(target)
            else:
                self.logger.error(f"Unsupported deployment type: {self.config.deployment_type}")
                return False
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            return False

    def _deploy_docker(self, target: str) -> bool:
        """Deploy using Docker"""
        self.logger.info("Deploying with Docker...")
        # Implementation would use Docker SDK
        return True

    def _deploy_kubernetes(self, target: str) -> bool:
        """Deploy to Kubernetes"""
        self.logger.info("Deploying to Kubernetes...")
        # Implementation would use Kubernetes Python client
        return True

    def _deploy_docker_compose(self, target: str) -> bool:
        """Deploy using Docker Compose"""
        self.logger.info("Deploying with Docker Compose...")
        # Implementation would use Docker Compose API
        return True

    def rollback(self, version: Optional[str] = None) -> bool:
        """Rollback deployment"""
        self.logger.info(f"Rolling back deployment to version: {version}")
        # Implementation would handle rollback logic
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            'deployment_type': self.config.deployment_type.value,
            'status': 'running',
            'version': self.config.version,
            'last_updated': self.config.updated_at.isoformat()
        }

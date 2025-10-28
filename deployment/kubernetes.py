"""
Quantum Edge AI Platform - Kubernetes Deployment

Kubernetes orchestration for quantum edge AI platform including
deployments, services, ingress, and Helm charts.
"""

import os
import json
import time
import base64
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging
import subprocess
import tempfile
import shutil

# Third-party imports (would be installed in production)
try:
    import yaml
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
except ImportError:
    yaml = None
    client = config = ApiException = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KubernetesResource:
    """Base Kubernetes resource"""
    api_version: str
    kind: str
    metadata: Dict[str, Any]
    spec: Dict[str, Any]

    def to_yaml(self) -> str:
        """Convert to YAML"""
        if not yaml:
            raise ImportError("PyYAML is required for YAML serialization")

        data = asdict(self)
        return yaml.dump(data, default_flow_style=False)

@dataclass
class KubernetesDeployment(KubernetesResource):
    """Kubernetes Deployment resource"""

    def __init__(self, name: str, namespace: str = "default",
                 replicas: int = 1, image: str = "", labels: Dict[str, str] = None):
        super().__init__(
            api_version="apps/v1",
            kind="Deployment",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={}
        )

        self.spec = {
            "replicas": replicas,
            "selector": {
                "matchLabels": {"app": name}
            },
            "template": {
                "metadata": {
                    "labels": {"app": name}
                },
                "spec": {
                    "containers": [{
                        "name": name,
                        "image": image,
                        "ports": [{"containerPort": 8080}],
                        "resources": {
                            "requests": {"memory": "256Mi", "cpu": "250m"},
                            "limits": {"memory": "512Mi", "cpu": "500m"}
                        }
                    }]
                }
            }
        }

@dataclass
class KubernetesService(KubernetesResource):
    """Kubernetes Service resource"""

    def __init__(self, name: str, namespace: str = "default",
                 service_type: str = "ClusterIP", ports: List[Dict[str, Any]] = None,
                 labels: Dict[str, str] = None):
        super().__init__(
            api_version="v1",
            kind="Service",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={
                "type": service_type,
                "selector": {"app": name},
                "ports": ports or [{"port": 80, "targetPort": 8080}]
            }
        )

@dataclass
class KubernetesIngress(KubernetesResource):
    """Kubernetes Ingress resource"""

    def __init__(self, name: str, namespace: str = "default",
                 host: str = "", paths: List[Dict[str, Any]] = None,
                 tls: List[Dict[str, Any]] = None, labels: Dict[str, str] = None):
        super().__init__(
            api_version="networking.k8s.io/v1",
            kind="Ingress",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={}
        )

        self.spec = {
            "rules": [{
                "host": host,
                "http": {
                    "paths": paths or [{
                        "path": "/",
                        "pathType": "Prefix",
                        "backend": {
                            "service": {
                                "name": name,
                                "port": {"number": 80}
                            }
                        }
                    }]
                }
            }]
        }

        if tls:
            self.spec["tls"] = tls

@dataclass
class KubernetesConfigMap(KubernetesResource):
    """Kubernetes ConfigMap resource"""

    def __init__(self, name: str, namespace: str = "default",
                 data: Dict[str, str] = None, labels: Dict[str, str] = None):
        super().__init__(
            api_version="v1",
            kind="ConfigMap",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={}
        )

        self.data = data or {}

@dataclass
class KubernetesSecret(KubernetesResource):
    """Kubernetes Secret resource"""

    def __init__(self, name: str, namespace: str = "default",
                 data: Dict[str, str] = None, secret_type: str = "Opaque",
                 labels: Dict[str, str] = None):
        super().__init__(
            api_version="v1",
            kind="Secret",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={}
        )

        self.type = secret_type
        self.data = data or {}

@dataclass
class KubernetesHPA(KubernetesResource):
    """Kubernetes HorizontalPodAutoscaler resource"""

    def __init__(self, name: str, namespace: str = "default",
                 target_deployment: str = "", min_replicas: int = 1,
                 max_replicas: int = 10, cpu_target: int = 70,
                 memory_target: int = 80, labels: Dict[str, str] = None):
        super().__init__(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={}
        )

        self.spec = {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": target_deployment
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": cpu_target
                        }
                    }
                },
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": memory_target
                        }
                    }
                }
            ]
        }

@dataclass
class KubernetesPVC(KubernetesResource):
    """Kubernetes PersistentVolumeClaim resource"""

    def __init__(self, name: str, namespace: str = "default",
                 storage_class: str = "", access_modes: List[str] = None,
                 storage_size: str = "10Gi", labels: Dict[str, str] = None):
        super().__init__(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata={
                "name": name,
                "namespace": namespace,
                "labels": labels or {"app": name}
            },
            spec={}
        )

        self.spec = {
            "accessModes": access_modes or ["ReadWriteOnce"],
            "resources": {
                "requests": {
                    "storage": storage_size
                }
            }
        }

        if storage_class:
            self.spec["storageClassName"] = storage_class

@dataclass
class HelmChart:
    """Helm chart configuration"""

    name: str
    version: str = "0.1.0"
    description: str = ""
    app_version: str = "1.0.0"
    keywords: List[str] = field(default_factory=list)
    home: str = ""
    sources: List[str] = field(default_factory=list)
    maintainers: List[Dict[str, str]] = field(default_factory=list)

    # Chart files
    chart_yaml: Dict[str, Any] = field(default_factory=dict)
    values_yaml: Dict[str, Any] = field(default_factory=dict)
    templates: Dict[str, str] = field(default_factory=dict)

class KubernetesDeploymentManager:
    """Kubernetes deployment manager"""

    def __init__(self, namespace: str = "quantum-edge",
                 kubeconfig: Optional[str] = None):
        self.namespace = namespace
        self.kubeconfig = kubeconfig or os.getenv('KUBECONFIG')
        self.client = None
        self.apps_client = None
        self.logger = logging.getLogger(__name__)

        if client and config:
            try:
                if self.kubeconfig:
                    config.load_kube_config(self.kubeconfig)
                else:
                    config.load_incluster_config()

                self.client = client.CoreV1Api()
                self.apps_client = client.AppsV1Api()
                self.logger.info("Connected to Kubernetes cluster")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Kubernetes: {str(e)}")

    def create_namespace(self, name: str) -> bool:
        """Create Kubernetes namespace"""
        if not self.client:
            return False

        try:
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=name)
            )
            self.client.create_namespace(namespace)
            self.logger.info(f"Created namespace: {name}")
            return True
        except ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.info(f"Namespace already exists: {name}")
                return True
            else:
                self.logger.error(f"Failed to create namespace: {e}")
                return False

    def deploy_resource(self, resource: KubernetesResource) -> bool:
        """Deploy Kubernetes resource"""
        if not self.client:
            return False

        try:
            # Convert resource to Kubernetes API object
            k8s_object = self._resource_to_k8s_object(resource)

            # Deploy based on resource type
            if resource.kind == "Deployment":
                self.apps_client.create_namespaced_deployment(
                    namespace=resource.metadata["namespace"],
                    body=k8s_object
                )
            elif resource.kind == "Service":
                self.client.create_namespaced_service(
                    namespace=resource.metadata["namespace"],
                    body=k8s_object
                )
            elif resource.kind == "ConfigMap":
                self.client.create_namespaced_config_map(
                    namespace=resource.metadata["namespace"],
                    body=k8s_object
                )
            elif resource.kind == "Secret":
                self.client.create_namespaced_secret(
                    namespace=resource.metadata["namespace"],
                    body=k8s_object
                )
            elif resource.kind == "Ingress":
                networking_client = client.NetworkingV1Api()
                networking_client.create_namespaced_ingress(
                    namespace=resource.metadata["namespace"],
                    body=k8s_object
                )
            elif resource.kind == "PersistentVolumeClaim":
                self.client.create_namespaced_persistent_volume_claim(
                    namespace=resource.metadata["namespace"],
                    body=k8s_object
                )

            self.logger.info(f"Deployed {resource.kind}: {resource.metadata['name']}")
            return True

        except ApiException as e:
            self.logger.error(f"Failed to deploy {resource.kind}: {e}")
            return False

    def _resource_to_k8s_object(self, resource: KubernetesResource) -> Any:
        """Convert custom resource to Kubernetes API object"""
        if resource.kind == "Deployment":
            return client.V1Deployment(
                api_version=resource.api_version,
                kind=resource.kind,
                metadata=client.V1ObjectMeta(**resource.metadata),
                spec=client.V1DeploymentSpec(**resource.spec)
            )
        elif resource.kind == "Service":
            return client.V1Service(
                api_version=resource.api_version,
                kind=resource.kind,
                metadata=client.V1ObjectMeta(**resource.metadata),
                spec=client.V1ServiceSpec(**resource.spec)
            )
        elif resource.kind == "ConfigMap":
            return client.V1ConfigMap(
                api_version=resource.api_version,
                kind=resource.kind,
                metadata=client.V1ObjectMeta(**resource.metadata),
                data=getattr(resource, 'data', {})
            )
        elif resource.kind == "Secret":
            return client.V1Secret(
                api_version=resource.api_version,
                kind=resource.kind,
                metadata=client.V1ObjectMeta(**resource.metadata),
                type=getattr(resource, 'type', 'Opaque'),
                data=getattr(resource, 'data', {})
            )
        elif resource.kind == "Ingress":
            return client.V1Ingress(
                api_version=resource.api_version,
                kind=resource.kind,
                metadata=client.V1ObjectMeta(**resource.metadata),
                spec=client.V1IngressSpec(**resource.spec)
            )
        elif resource.kind == "PersistentVolumeClaim":
            return client.V1PersistentVolumeClaim(
                api_version=resource.api_version,
                kind=resource.kind,
                metadata=client.V1ObjectMeta(**resource.metadata),
                spec=client.V1PersistentVolumeClaimSpec(**resource.spec)
            )

        raise ValueError(f"Unsupported resource type: {resource.kind}")

    def get_resource_status(self, kind: str, name: str) -> Optional[Dict[str, Any]]:
        """Get resource status"""
        if not self.client:
            return None

        try:
            if kind == "Deployment":
                resource = self.apps_client.read_namespaced_deployment(
                    name=name, namespace=self.namespace
                )
            elif kind == "Service":
                resource = self.client.read_namespaced_service(
                    name=name, namespace=self.namespace
                )
            elif kind == "Pod":
                resource = self.client.read_namespaced_pod(
                    name=name, namespace=self.namespace
                )

            return {
                "name": resource.metadata.name,
                "status": getattr(resource.status, 'phase', 'Unknown') if hasattr(resource, 'status') else 'Unknown',
                "created": resource.metadata.creation_timestamp.isoformat() if resource.metadata.creation_timestamp else None
            }

        except ApiException as e:
            self.logger.error(f"Failed to get resource status: {e}")
            return None

    def scale_deployment(self, name: str, replicas: int) -> bool:
        """Scale deployment"""
        if not self.apps_client:
            return False

        try:
            # Get current deployment
            deployment = self.apps_client.read_namespaced_deployment(
                name=name, namespace=self.namespace
            )

            # Update replicas
            deployment.spec.replicas = replicas

            # Update deployment
            self.apps_client.replace_namespaced_deployment(
                name=name, namespace=self.namespace, body=deployment
            )

            self.logger.info(f"Scaled deployment {name} to {replicas} replicas")
            return True

        except ApiException as e:
            self.logger.error(f"Failed to scale deployment: {e}")
            return False

    def delete_resource(self, kind: str, name: str) -> bool:
        """Delete Kubernetes resource"""
        if not self.client:
            return False

        try:
            if kind == "Deployment":
                self.apps_client.delete_namespaced_deployment(
                    name=name, namespace=self.namespace
                )
            elif kind == "Service":
                self.client.delete_namespaced_service(
                    name=name, namespace=self.namespace
                )
            elif kind == "ConfigMap":
                self.client.delete_namespaced_config_map(
                    name=name, namespace=self.namespace
                )
            elif kind == "Secret":
                self.client.delete_namespaced_secret(
                    name=name, namespace=self.namespace
                )

            self.logger.info(f"Deleted {kind}: {name}")
            return True

        except ApiException as e:
            self.logger.error(f"Failed to delete {kind}: {e}")
            return False

class HelmDeployment:
    """Helm chart deployment manager"""

    def __init__(self, chart_path: str = ".", release_name: str = "quantum-edge",
                 namespace: str = "quantum-edge"):
        self.chart_path = Path(chart_path)
        self.release_name = release_name
        self.namespace = namespace
        self.logger = logging.getLogger(__name__)

    def create_chart(self, chart: HelmChart) -> str:
        """Create Helm chart structure"""

        # Create chart directory
        chart_dir = self.chart_path / chart.name
        chart_dir.mkdir(exist_ok=True)

        # Create Chart.yaml
        chart_yaml = {
            "apiVersion": "v2",
            "name": chart.name,
            "description": chart.description,
            "type": "application",
            "version": chart.version,
            "appVersion": chart.app_version
        }

        if chart.keywords:
            chart_yaml["keywords"] = chart.keywords
        if chart.home:
            chart_yaml["home"] = chart.home
        if chart.sources:
            chart_yaml["sources"] = chart.sources
        if chart.maintainers:
            chart_yaml["maintainers"] = chart.maintainers

        with open(chart_dir / "Chart.yaml", 'w') as f:
            yaml.dump(chart_yaml, f, default_flow_style=False)

        # Create values.yaml
        with open(chart_dir / "values.yaml", 'w') as f:
            yaml.dump(chart.values_yaml, f, default_flow_style=False)

        # Create templates directory
        templates_dir = chart_dir / "templates"
        templates_dir.mkdir(exist_ok=True)

        # Create template files
        for template_name, template_content in chart.templates.items():
            with open(templates_dir / template_name, 'w') as f:
                f.write(template_content)

        self.logger.info(f"Created Helm chart: {chart.name}")
        return str(chart_dir)

    def package_chart(self, chart_path: str) -> Optional[str]:
        """Package Helm chart"""
        try:
            cmd = ["helm", "package", chart_path]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.chart_path)

            if result.returncode == 0:
                # Extract package name from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith("Successfully packaged chart and saved it to:"):
                        package_path = line.split(": ")[1]
                        self.logger.info(f"Packaged Helm chart: {package_path}")
                        return package_path

                # Fallback: find .tgz file
                chart_name = Path(chart_path).name
                for file in Path(self.chart_path).glob(f"{chart_name}-*.tgz"):
                    return str(file)

            else:
                self.logger.error(f"Failed to package chart: {result.stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Error packaging chart: {str(e)}")
            return None

    def install_chart(self, chart_path: str, values: Dict[str, Any] = None) -> bool:
        """Install Helm chart"""
        try:
            cmd = ["helm", "install", self.release_name, chart_path,
                   "--namespace", self.namespace, "--create-namespace"]

            # Add values
            if values:
                values_file = self._create_temp_values_file(values)
                cmd.extend(["-f", values_file])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"Installed Helm chart: {self.release_name}")
                return True
            else:
                self.logger.error(f"Failed to install chart: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error installing chart: {str(e)}")
            return False

    def upgrade_chart(self, chart_path: str, values: Dict[str, Any] = None) -> bool:
        """Upgrade Helm chart"""
        try:
            cmd = ["helm", "upgrade", self.release_name, chart_path,
                   "--namespace", self.namespace]

            # Add values
            if values:
                values_file = self._create_temp_values_file(values)
                cmd.extend(["-f", values_file])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"Upgraded Helm chart: {self.release_name}")
                return True
            else:
                self.logger.error(f"Failed to upgrade chart: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error upgrading chart: {str(e)}")
            return False

    def uninstall_chart(self) -> bool:
        """Uninstall Helm chart"""
        try:
            cmd = ["helm", "uninstall", self.release_name, "--namespace", self.namespace]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(f"Uninstalled Helm chart: {self.release_name}")
                return True
            else:
                self.logger.error(f"Failed to uninstall chart: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error uninstalling chart: {str(e)}")
            return False

    def _create_temp_values_file(self, values: Dict[str, Any]) -> str:
        """Create temporary values file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(values, f, default_flow_style=False)
            return f.name

class K3sDeployment:
    """K3s deployment for edge devices"""

    def __init__(self, master_ip: str = "", token: str = "",
                 node_labels: Dict[str, str] = None):
        self.master_ip = master_ip
        self.token = token
        self.node_labels = node_labels or {}
        self.logger = logging.getLogger(__name__)

    def generate_master_config(self) -> str:
        """Generate K3s master configuration"""
        config = f"""# K3s Master Configuration
# Generated for Quantum Edge AI Platform

# Disable components not needed for edge AI
disable:
  - traefik
  - servicelb

# Enable useful components
enable:
  - kubelet
  - kube-apiserver

# TLS configuration
tls-san:
  - {self.master_ip}

# Data directory
data-dir: /var/lib/rancher/k3s

# Node labels for edge AI workloads
node-label:
"""

        for key, value in self.node_labels.items():
            config += f"  - {key}={value}\n"

        config += """
# Kubelet configuration
kubelet-arg:
  - "max-pods=50"
  - "cpu-manager-policy=static"
  - "topology-manager-policy=single-numa-node"

# API server configuration
kube-apiserver-arg:
  - "enable-admission-plugins=NodeRestriction,PodSecurityPolicy"
  - "audit-log-path=/var/log/k3s-audit.log"
  - "audit-log-maxage=30"
"""

        return config

    def generate_worker_config(self) -> str:
        """Generate K3s worker configuration"""
        config = f"""# K3s Worker Configuration
# Generated for Quantum Edge AI Platform

# Master server
server: https://{self.master_ip}:6443

# Join token
token: {self.token}

# Node labels for edge AI workloads
node-label:
"""

        for key, value in self.node_labels.items():
            config += f"  - {key}={value}\n"

        config += """
# Kubelet configuration for edge devices
kubelet-arg:
  - "max-pods=20"
  - "cpu-manager-policy=none"
  - "topology-manager-policy=none"
  - "serialize-image-pulls=false"

# Disable unused components
disable:
  - traefik
  - servicelb
"""

        return config

    def create_edge_deployment_script(self, master: bool = False) -> str:
        """Create deployment script for edge devices"""
        script = """#!/bin/bash

# Quantum Edge AI Platform - K3s Deployment Script
# This script deploys K3s on edge devices for distributed AI workloads

set -e

# Configuration
K3S_VERSION="v1.24.4+k3s1"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/rancher/k3s"
DATA_DIR="/var/lib/rancher/k3s"
LOG_DIR="/var/log/k3s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root"
   exit 1
fi

# Install dependencies
log_info "Installing dependencies..."
apt-get update
apt-get install -y curl wget jq iptables

# Create directories
mkdir -p $CONFIG_DIR
mkdir -p $DATA_DIR
mkdir -p $LOG_DIR

# Install K3s
log_info "Installing K3s..."
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=$K3S_VERSION sh -

# Wait for K3s to start
log_info "Waiting for K3s to start..."
sleep 30

# Verify installation
if systemctl is-active --quiet k3s; then
    log_info "K3s installed and running successfully"
else
    log_error "K3s failed to start"
    exit 1
fi

# Configure firewall (allow necessary ports)
log_info "Configuring firewall..."
iptables -A INPUT -p tcp --dport 6443 -j ACCEPT  # K3s API
iptables -A INPUT -p tcp --dport 8472 -j ACCEPT  # Flannel VXLAN
iptables -A INPUT -p udp --dport 8472 -j ACCEPT  # Flannel VXLAN
iptables -A INPUT -p tcp --dport 10250 -j ACCEPT # Kubelet

# Save iptables rules
iptables-save > /etc/iptables/rules.v4

log_info "K3s deployment completed successfully!"
log_info "You can now deploy Quantum Edge AI workloads to this cluster."
"""

        return script

    def create_k3s_systemd_override(self) -> str:
        """Create systemd override for K3s"""
        override = """[Service]
# Override K3s systemd service for edge optimization

# Limit CPU usage
CPUQuota=80%

# Limit memory usage
MemoryLimit=1G

# Restart on failure
Restart=always
RestartSec=10

# Environment variables for edge AI
Environment=K3S_NODE_NAME=edge-device
Environment=K3S_KUBECONFIG_MODE=644
"""

        return override

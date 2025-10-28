"""
Quantum Edge AI Platform - Monitoring Configuration

Comprehensive monitoring configurations including metrics collection,
logging, alerting, and observability for quantum edge AI systems.
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
from pathlib import Path
import statistics

# Third-party imports (would be installed in production)
try:
    import psutil
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
except ImportError:
    # Fallback for development without dependencies
    psutil = prometheus_client = None
    Counter = Gauge = Histogram = Summary = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    for_duration: str = "5m"  # How long condition must be true
    enabled: bool = True

@dataclass
class LogConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    json_format: bool = False

    # Structured logging
    include_request_id: bool = True
    include_user_id: bool = True
    include_trace_id: bool = False

    # Log aggregation
    aggregation_enabled: bool = False
    aggregation_endpoint: Optional[str] = None

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""

    # Metrics collection
    metrics_enabled: bool = True
    metrics_interval: int = 60  # seconds
    metrics_retention_days: int = 30

    # Prometheus configuration
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"

    # Custom metrics
    custom_metrics: List[MetricDefinition] = field(default_factory=list)

    # Alerting
    alerting_enabled: bool = True
    alert_rules: List[AlertRule] = field(default_factory=list)
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = field(default_factory=list)

    # Health checks
    health_check_enabled: bool = True
    health_check_path: str = "/health"
    health_check_interval: int = 30

    # Tracing
    tracing_enabled: bool = False
    tracing_sample_rate: float = 0.1
    tracing_exporter: str = "jaeger"  # jaeger, zipkin, otlp

    # Dashboards
    dashboard_enabled: bool = True
    dashboard_port: int = 3000
    dashboard_config: Dict[str, Any] = field(default_factory=dict)

    def initialize_default_metrics(self):
        """Initialize default metrics definitions"""
        self.custom_metrics = [
            MetricDefinition(
                name="quantum_edge_requests_total",
                type=MetricType.COUNTER,
                description="Total number of API requests",
                labels=["method", "endpoint", "status"]
            ),
            MetricDefinition(
                name="quantum_edge_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Request duration in seconds",
                labels=["method", "endpoint"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            MetricDefinition(
                name="quantum_edge_active_connections",
                type=MetricType.GAUGE,
                description="Number of active connections",
                labels=["type"]
            ),
            MetricDefinition(
                name="quantum_edge_model_inference_count",
                type=MetricType.COUNTER,
                description="Number of model inferences",
                labels=["model_id", "precision"]
            ),
            MetricDefinition(
                name="quantum_edge_memory_usage_bytes",
                type=MetricType.GAUGE,
                description="Memory usage in bytes",
                labels=["type"]
            ),
            MetricDefinition(
                name="quantum_edge_cpu_usage_percent",
                type=MetricType.GAUGE,
                description="CPU usage percentage",
                labels=["core"]
            ),
            MetricDefinition(
                name="quantum_edge_quantum_circuit_depth",
                type=MetricType.HISTOGRAM,
                description="Quantum circuit depth distribution",
                labels=["circuit_type"],
                buckets=[1, 5, 10, 20, 50, 100]
            ),
            MetricDefinition(
                name="quantum_edge_federated_clients",
                type=MetricType.GAUGE,
                description="Number of federated learning clients",
                labels=["status"]
            )
        ]

    def initialize_default_alerts(self):
        """Initialize default alert rules"""
        self.alert_rules = [
            AlertRule(
                name="HighErrorRate",
                description="Error rate is too high",
                condition="rate(quantum_edge_requests_total{status=~\"5..\"}[5m]) / rate(quantum_edge_requests_total[5m]) > 0.05",
                severity=AlertSeverity.ERROR,
                labels={"service": "quantum-edge"},
                annotations={
                    "summary": "High error rate detected",
                    "description": "Error rate is {{ $value }}% which is above the threshold of 5%"
                }
            ),
            AlertRule(
                name="HighMemoryUsage",
                description="Memory usage is too high",
                condition="quantum_edge_memory_usage_bytes > 0.9 * 1024*1024*1024",  # 900MB
                severity=AlertSeverity.WARNING,
                labels={"service": "quantum-edge"},
                annotations={
                    "summary": "High memory usage",
                    "description": "Memory usage is at {{ $value }} bytes"
                }
            ),
            AlertRule(
                name="HighCPUUsage",
                description="CPU usage is too high",
                condition="rate(quantum_edge_cpu_usage_percent[5m]) > 80",
                severity=AlertSeverity.WARNING,
                labels={"service": "quantum-edge"},
                annotations={
                    "summary": "High CPU usage",
                    "description": "CPU usage is at {{ $value }}%"
                }
            ),
            AlertRule(
                name="FederatedClientDisconnect",
                description="Federated learning client disconnected",
                condition="quantum_edge_federated_clients{status=\"disconnected\"} > 0",
                severity=AlertSeverity.INFO,
                labels={"service": "quantum-edge"},
                annotations={
                    "summary": "Federated client disconnected",
                    "description": "{{ $value }} federated clients are disconnected"
                }
            ),
            AlertRule(
                name="QuantumCircuitTimeout",
                description="Quantum circuit execution timeout",
                condition="increase(quantum_edge_quantum_circuit_depth[10m]) == 0",
                severity=AlertSeverity.WARNING,
                labels={"service": "quantum-edge"},
                annotations={
                    "summary": "Quantum circuit timeout",
                    "description": "No quantum circuits executed in the last 10 minutes"
                }
            )
        ]

class MetricsCollector:
    """Metrics collector for system monitoring"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Prometheus metrics registry
        self.registry = None
        if prometheus_client:
            self.registry = prometheus_client.CollectorRegistry()

        # Custom metrics storage
        self._metrics_store = {}
        self._collection_thread = None
        self._collecting = False

        # Performance data
        self._performance_history = []
        self._max_history_size = 1000

    def initialize_metrics(self):
        """Initialize metrics collection"""
        if not self.config.metrics_enabled:
            return

        # Initialize default metrics if not set
        if not self.config.custom_metrics:
            self.config.initialize_default_metrics()

        # Create Prometheus metrics
        if self.registry:
            for metric_def in self.config.custom_metrics:
                self._create_prometheus_metric(metric_def)

        # Start collection thread
        self._start_collection()

        self.logger.info("Metrics collection initialized")

    def _create_prometheus_metric(self, metric_def: MetricDefinition):
        """Create Prometheus metric"""
        if not self.registry:
            return

        metric_name = metric_def.name.replace('_', '')

        if metric_def.type == MetricType.COUNTER:
            metric = Counter(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        elif metric_def.type == MetricType.GAUGE:
            metric = Gauge(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )
        elif metric_def.type == MetricType.HISTOGRAM:
            metric = Histogram(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                buckets=metric_def.buckets or prometheus_client.Histogram.DEFAULT_BUCKETS,
                registry=self.registry
            )
        elif metric_def.type == MetricType.SUMMARY:
            metric = Summary(
                metric_name,
                metric_def.description,
                labelnames=metric_def.labels,
                registry=self.registry
            )

        self._metrics_store[metric_def.name] = metric

    def _start_collection(self):
        """Start metrics collection thread"""
        if self._collection_thread and self._collection_thread.is_alive():
            return

        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self._collection_thread.start()

    def _collect_metrics_loop(self):
        """Metrics collection loop"""
        while self._collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.config.metrics_interval)
            except Exception as e:
                self.logger.error(f"Metrics collection error: {str(e)}")
                time.sleep(5)

    def _collect_system_metrics(self):
        """Collect system metrics"""
        if not psutil:
            return

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_gauge("quantum_edge_cpu_usage_percent", cpu_percent, {"core": "all"})

            # Memory metrics
            memory = psutil.virtual_memory()
            self.update_gauge("quantum_edge_memory_usage_bytes", memory.used, {"type": "system"})
            self.update_gauge("quantum_edge_memory_usage_bytes", memory.available, {"type": "available"})

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.update_gauge("quantum_edge_disk_usage_bytes", disk.used, {"mount": "/"})

            # Network metrics
            network = psutil.net_io_counters()
            if network:
                self.update_counter("quantum_edge_network_bytes_total", network.bytes_sent, {"direction": "tx"})
                self.update_counter("quantum_edge_network_bytes_total", network.bytes_recv, {"direction": "rx"})

            # Store performance history
            performance_data = {
                'timestamp': datetime.utcnow(),
                'cpu_percent': cpu_percent,
                'memory_used': memory.used,
                'memory_available': memory.available,
                'disk_used': disk.used,
                'network_sent': network.bytes_sent if network else 0,
                'network_recv': network.bytes_recv if network else 0
            }

            self._performance_history.append(performance_data)
            if len(self._performance_history) > self._max_history_size:
                self._performance_history = self._performance_history[-self._max_history_size:]

        except Exception as e:
            self.logger.error(f"System metrics collection error: {str(e)}")

    def update_counter(self, name: str, value: float, labels: Dict[str, str] = None):
        """Update counter metric"""
        if name in self._metrics_store:
            metric = self._metrics_store[name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def update_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Update gauge metric"""
        if name in self._metrics_store:
            metric = self._metrics_store[name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe histogram metric"""
        if name in self._metrics_store:
            metric = self._metrics_store[name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    def get_metrics_data(self) -> Dict[str, Any]:
        """Get current metrics data"""
        return {
            'system_metrics': self._get_latest_performance_data(),
            'custom_metrics': {name: str(metric) for name, metric in self._metrics_store.items()},
            'collection_status': 'active' if self._collecting else 'inactive',
            'metrics_count': len(self._metrics_store)
        }

    def _get_latest_performance_data(self) -> Dict[str, Any]:
        """Get latest performance data"""
        if not self._performance_history:
            return {}

        latest = self._performance_history[-1]

        # Calculate some statistics
        if len(self._performance_history) > 1:
            cpu_history = [p['cpu_percent'] for p in self._performance_history[-10:]]
            memory_history = [p['memory_used'] for p in self._performance_history[-10:]]

            return {
                'timestamp': latest['timestamp'].isoformat(),
                'cpu_percent': latest['cpu_percent'],
                'cpu_avg_10min': statistics.mean(cpu_history) if cpu_history else 0,
                'memory_used': latest['memory_used'],
                'memory_available': latest['memory_available'],
                'memory_avg_10min': statistics.mean(memory_history) if memory_history else 0,
                'disk_used': latest['disk_used'],
                'network_sent': latest['network_sent'],
                'network_recv': latest['network_recv']
            }

        return asdict(latest)

class AlertManager:
    """Alert manager for monitoring alerts"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Alert state
        self._active_alerts = {}
        self._alert_history = []

        # Initialize default alerts if not set
        if not self.config.alert_rules:
            self.config.initialize_default_alerts()

    def evaluate_alerts(self, metrics_data: Dict[str, Any]):
        """Evaluate alert rules against metrics"""
        if not self.config.alerting_enabled:
            return

        for rule in self.config.alert_rules:
            if not rule.enabled:
                continue

            try:
                if self._evaluate_rule(rule, metrics_data):
                    self._trigger_alert(rule)
                else:
                    self._resolve_alert(rule)
            except Exception as e:
                self.logger.error(f"Alert evaluation error for {rule.name}: {str(e)}")

    def _evaluate_rule(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> bool:
        """Evaluate alert rule condition"""
        # Simplified condition evaluation
        # In production, this would use a proper expression evaluator
        condition = rule.condition

        # Simple threshold checks
        if ">" in condition:
            parts = condition.split(">")
            metric_name = parts[0].strip()
            threshold = float(parts[1].strip())

            metric_value = self._extract_metric_value(metric_name, metrics_data)
            return metric_value > threshold

        elif "<" in condition:
            parts = condition.split("<")
            metric_name = parts[0].strip()
            threshold = float(parts[1].strip())

            metric_value = self._extract_metric_value(metric_name, metrics_data)
            return metric_value < threshold

        return False

    def _extract_metric_value(self, metric_name: str, metrics_data: Dict[str, Any]) -> float:
        """Extract metric value from metrics data"""
        # Simplified metric extraction
        system_metrics = metrics_data.get('system_metrics', {})

        if 'cpu_percent' in metric_name:
            return system_metrics.get('cpu_percent', 0)
        elif 'memory_usage_bytes' in metric_name:
            return system_metrics.get('memory_used', 0)
        elif 'error_rate' in metric_name:
            return 0.02  # Placeholder
        elif 'federated_clients' in metric_name:
            return 5  # Placeholder

        return 0

    def _trigger_alert(self, rule: AlertRule):
        """Trigger alert"""
        if rule.name in self._active_alerts:
            return  # Already active

        alert = {
            'name': rule.name,
            'description': rule.description,
            'severity': rule.severity.value,
            'triggered_at': datetime.utcnow(),
            'labels': rule.labels,
            'annotations': rule.annotations
        }

        self._active_alerts[rule.name] = alert
        self._alert_history.append(alert)

        self.logger.warning(f"ALERT TRIGGERED: {rule.name} - {rule.description}")

        # Send notifications
        self._send_alert_notifications(alert)

    def _resolve_alert(self, rule: AlertRule):
        """Resolve alert"""
        if rule.name not in self._active_alerts:
            return

        alert = self._active_alerts[rule.name]
        alert['resolved_at'] = datetime.utcnow()

        self.logger.info(f"ALERT RESOLVED: {rule.name}")

        # Send resolution notifications
        self._send_resolution_notifications(alert)

        del self._active_alerts[rule.name]

    def _send_alert_notifications(self, alert: Dict[str, Any]):
        """Send alert notifications"""
        # Webhook notification
        if self.config.alert_webhook_url:
            try:
                # In production, send HTTP request to webhook
                self.logger.info(f"Sending alert to webhook: {alert['name']}")
            except Exception as e:
                self.logger.error(f"Webhook notification failed: {str(e)}")

        # Email notification
        if self.config.alert_email_recipients:
            try:
                # In production, send email
                self.logger.info(f"Sending alert email for: {alert['name']}")
            except Exception as e:
                self.logger.error(f"Email notification failed: {str(e)}")

    def _send_resolution_notifications(self, alert: Dict[str, Any]):
        """Send alert resolution notifications"""
        # Similar to alert notifications but for resolution
        pass

    def get_alert_status(self) -> Dict[str, Any]:
        """Get alert status"""
        return {
            'active_alerts': list(self._active_alerts.values()),
            'alert_history': self._alert_history[-50:],  # Last 50 alerts
            'total_alerts': len(self._alert_history),
            'alert_rules': len(self.config.alert_rules)
        }

class LogManager:
    """Log manager for structured logging"""

    def __init__(self, config: LogConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        logger = logging.getLogger('quantum_edge')
        logger.setLevel(getattr(logging, self.config.level.value))

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatter
        if self.config.json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(self.config.format)

        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if self.config.file_path:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def log_request(self, request_data: Dict[str, Any], response_data: Dict[str, Any]):
        """Log HTTP request"""
        extra = {}

        if self.config.include_request_id:
            extra['request_id'] = request_data.get('request_id', 'unknown')

        if self.config.include_user_id:
            extra['user_id'] = request_data.get('user_id', 'anonymous')

        if self.config.include_trace_id:
            extra['trace_id'] = request_data.get('trace_id', 'unknown')

        status_code = response_data.get('status_code', 200)
        processing_time = response_data.get('_processing_time', 0)

        message = f"{request_data.get('method')} {request_data.get('path')} {status_code} {processing_time:.3f}s"

        if status_code >= 400:
            self.logger.error(message, extra=extra)
        else:
            self.logger.info(message, extra=extra)

    def log_model_inference(self, model_id: str, input_shape: tuple, latency: float):
        """Log model inference"""
        self.logger.info(
            f"Model inference: {model_id}, input_shape={input_shape}, latency={latency:.3f}s"
        )

    def log_quantum_execution(self, circuit_depth: int, execution_time: float, shots: int):
        """Log quantum circuit execution"""
        self.logger.info(
            f"Quantum execution: depth={circuit_depth}, time={execution_time:.3f}s, shots={shots}"
        )

    def log_security_event(self, event_type: str, user: str, resource: str, result: str):
        """Log security event"""
        self.logger.warning(
            f"Security event: {event_type}, user={user}, resource={resource}, result={result}"
        )

    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'log_level': self.config.level.value,
            'json_format': self.config.json_format,
            'file_logging': self.config.file_path is not None,
            'console_logging': self.config.console_output,
            'structured_logging': self.config.include_request_id or self.config.include_user_id
        }

#!/usr/bin/env python3
"""
Comprehensive Metrics Collection System for Quantum Edge AI Platform

This module provides extensive monitoring capabilities including:
- System resource metrics (CPU, memory, disk, network)
- Application performance metrics
- Quantum computing metrics
- Privacy and security metrics
- Federated learning metrics
- Custom business metrics
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import csv
import sqlite3
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric measurement."""
    timestamp: datetime
    name: str
    value: Union[int, float, bool, str]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series data for a metric."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    aggregation_window: timedelta = field(default_factory=lambda: timedelta(minutes=1))

    def add_point(self, point: MetricPoint):
        """Add a metric point to the series."""
        self.points.append(point)

    def get_recent_points(self, duration: timedelta) -> List[MetricPoint]:
        """Get points within the specified duration."""
        cutoff = datetime.now() - duration
        return [p for p in self.points if p.timestamp >= cutoff]

    def aggregate(self, aggregator: str = 'mean') -> Optional[float]:
        """Aggregate recent points."""
        recent = self.get_recent_points(self.aggregation_window)
        if not recent:
            return None

        values = [p.value for p in recent if isinstance(p.value, (int, float))]
        if not values:
            return None

        if aggregator == 'mean':
            return np.mean(values)
        elif aggregator == 'sum':
            return np.sum(values)
        elif aggregator == 'max':
            return np.max(values)
        elif aggregator == 'min':
            return np.min(values)
        elif aggregator == 'count':
            return len(values)
        else:
            return np.mean(values)


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors."""

    def __init__(self, name: str, collection_interval: float = 60.0):
        self.name = name
        self.collection_interval = collection_interval
        self.is_running = False
        self.collection_thread: Optional[threading.Thread] = None

    @abstractmethod
    def collect_metrics(self) -> List[MetricPoint]:
        """Collect metrics and return as MetricPoint objects."""
        pass

    def start_collection(self):
        """Start metric collection in a background thread."""
        if self.is_running:
            return

        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info(f"Started metrics collection for {self.name}")

    def stop_collection(self):
        """Stop metric collection."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info(f"Stopped metrics collection for {self.name}")

    def _collection_loop(self):
        """Main collection loop."""
        while self.is_running:
            try:
                metrics = self.collect_metrics()
                # In practice, these would be sent to a metrics storage system
                for metric in metrics:
                    logger.debug(f"Collected metric: {metric.name} = {metric.value}")

            except Exception as e:
                logger.error(f"Error collecting metrics for {self.name}: {e}")

            time.sleep(self.collection_interval)


class SystemMetricsCollector(MetricsCollector):
    """Collects system-level metrics (CPU, memory, disk, network)."""

    def __init__(self, collection_interval: float = 30.0):
        super().__init__("system_metrics", collection_interval)
        self.prev_net_io = psutil.net_io_counters()
        self.prev_disk_io = psutil.disk_io_counters()

    def collect_metrics(self) -> List[MetricPoint]:
        """Collect comprehensive system metrics."""
        timestamp = datetime.now()
        metrics = []

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_times = psutil.cpu_times_percent()

        metrics.extend([
            MetricPoint(timestamp, "system.cpu.percent", cpu_percent, {"type": "overall"}),
            MetricPoint(timestamp, "system.cpu.user", cpu_times.user, {"type": "user"}),
            MetricPoint(timestamp, "system.cpu.system", cpu_times.system, {"type": "system"}),
            MetricPoint(timestamp, "system.cpu.idle", cpu_times.idle, {"type": "idle"}),
        ])

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics.extend([
            MetricPoint(timestamp, "system.memory.percent", memory.percent, {"type": "virtual"}),
            MetricPoint(timestamp, "system.memory.used", memory.used, {"type": "virtual", "unit": "bytes"}),
            MetricPoint(timestamp, "system.memory.available", memory.available, {"type": "virtual", "unit": "bytes"}),
            MetricPoint(timestamp, "system.memory.swap_percent", swap.percent, {"type": "swap"}),
            MetricPoint(timestamp, "system.memory.swap_used", swap.used, {"type": "swap", "unit": "bytes"}),
        ])

        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        current_disk_io = psutil.disk_io_counters()

        metrics.extend([
            MetricPoint(timestamp, "system.disk.percent", disk_usage.percent, {"mount": "/"}),
            MetricPoint(timestamp, "system.disk.used", disk_usage.used, {"mount": "/", "unit": "bytes"}),
            MetricPoint(timestamp, "system.disk.free", disk_usage.free, {"mount": "/", "unit": "bytes"}),
        ])

        if current_disk_io and self.prev_disk_io:
            read_bytes = current_disk_io.read_bytes - self.prev_disk_io.read_bytes
            write_bytes = current_disk_io.write_bytes - self.prev_disk_io.write_bytes

            metrics.extend([
                MetricPoint(timestamp, "system.disk.read_bytes", read_bytes, {"unit": "bytes"}),
                MetricPoint(timestamp, "system.disk.write_bytes", write_bytes, {"unit": "bytes"}),
            ])

        self.prev_disk_io = current_disk_io

        # Network metrics
        current_net_io = psutil.net_io_counters()
        if current_net_io and self.prev_net_io:
            bytes_sent = current_net_io.bytes_sent - self.prev_net_io.bytes_sent
            bytes_recv = current_net_io.bytes_recv - self.prev_net_io.bytes_recv
            packets_sent = current_net_io.packets_sent - self.prev_net_io.packets_sent
            packets_recv = current_net_io.packets_recv - self.prev_net_io.packets_recv

            metrics.extend([
                MetricPoint(timestamp, "system.network.bytes_sent", bytes_sent, {"unit": "bytes"}),
                MetricPoint(timestamp, "system.network.bytes_recv", bytes_recv, {"unit": "bytes"}),
                MetricPoint(timestamp, "system.network.packets_sent", packets_sent),
                MetricPoint(timestamp, "system.network.packets_recv", packets_recv),
            ])

        self.prev_net_io = current_net_io

        # System load
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

        metrics.extend([
            MetricPoint(timestamp, "system.load.1min", load_avg[0]),
            MetricPoint(timestamp, "system.load.5min", load_avg[1]),
            MetricPoint(timestamp, "system.load.15min", load_avg[2]),
        ])

        # Process information
        process = psutil.Process()
        with process.oneshot():
            metrics.extend([
                MetricPoint(timestamp, "process.cpu_percent", process.cpu_percent()),
                MetricPoint(timestamp, "process.memory_percent", process.memory_percent()),
                MetricPoint(timestamp, "process.memory_rss", process.memory_info().rss, {"unit": "bytes"}),
                MetricPoint(timestamp, "process.memory_vms", process.memory_info().vms, {"unit": "bytes"}),
                MetricPoint(timestamp, "process.num_threads", process.num_threads()),
                MetricPoint(timestamp, "process.num_fds", process.num_fds() if hasattr(process, 'num_fds') else 0),
            ])

        return metrics


class ApplicationMetricsCollector(MetricsCollector):
    """Collects application-level performance metrics."""

    def __init__(self, collection_interval: float = 10.0):
        super().__init__("application_metrics", collection_interval)
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
        self.active_requests = 0

    def collect_metrics(self) -> List[MetricPoint]:
        """Collect application performance metrics."""
        timestamp = datetime.now()
        metrics = []

        # Request metrics
        metrics.extend([
            MetricPoint(timestamp, "app.requests.total", self.request_count),
            MetricPoint(timestamp, "app.requests.active", self.active_requests),
            MetricPoint(timestamp, "app.requests.errors", self.error_count),
        ])

        # Response time metrics
        if self.response_times:
            response_times = list(self.response_times)
            metrics.extend([
                MetricPoint(timestamp, "app.response_time.avg", np.mean(response_times), {"unit": "ms"}),
                MetricPoint(timestamp, "app.response_time.median", np.median(response_times), {"unit": "ms"}),
                MetricPoint(timestamp, "app.response_time.p95", np.percentile(response_times, 95), {"unit": "ms"}),
                MetricPoint(timestamp, "app.response_time.p99", np.percentile(response_times, 99), {"unit": "ms"}),
                MetricPoint(timestamp, "app.response_time.min", np.min(response_times), {"unit": "ms"}),
                MetricPoint(timestamp, "app.response_time.max", np.max(response_times), {"unit": "ms"}),
            ])

        # Error rate
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count
            metrics.append(MetricPoint(timestamp, "app.error_rate", error_rate))

        # Throughput (requests per second)
        # This is a simplified calculation - in practice you'd use a sliding window
        metrics.append(MetricPoint(timestamp, "app.throughput_rps", self.request_count / max(1, time.time() - psutil.boot_time())))

        return metrics

    def record_request(self, response_time: float, success: bool = True):
        """Record a request for metrics collection."""
        self.request_count += 1
        self.response_times.append(response_time)

        if not success:
            self.error_count += 1

    def increment_active_requests(self):
        """Increment active request count."""
        self.active_requests += 1

    def decrement_active_requests(self):
        """Decrement active request count."""
        self.active_requests = max(0, self.active_requests - 1)


class QuantumMetricsCollector(MetricsCollector):
    """Collects quantum computing performance metrics."""

    def __init__(self, collection_interval: float = 60.0):
        super().__init__("quantum_metrics", collection_interval)
        self.circuit_count = 0
        self.quantum_errors = 0
        self.execution_times = deque(maxlen=1000)
        self.fidelity_measurements = deque(maxlen=100)

    def collect_metrics(self) -> List[MetricPoint]:
        """Collect quantum computing metrics."""
        timestamp = datetime.now()
        metrics = []

        # Circuit execution metrics
        metrics.extend([
            MetricPoint(timestamp, "quantum.circuits.executed", self.circuit_count),
            MetricPoint(timestamp, "quantum.errors.total", self.quantum_errors),
        ])

        # Execution time metrics
        if self.execution_times:
            exec_times = list(self.execution_times)
            metrics.extend([
                MetricPoint(timestamp, "quantum.execution_time.avg", np.mean(exec_times), {"unit": "ms"}),
                MetricPoint(timestamp, "quantum.execution_time.median", np.median(exec_times), {"unit": "ms"}),
                MetricPoint(timestamp, "quantum.execution_time.p95", np.percentile(exec_times, 95), {"unit": "ms"}),
            ])

        # Fidelity metrics
        if self.fidelity_measurements:
            fidelities = list(self.fidelity_measurements)
            metrics.extend([
                MetricPoint(timestamp, "quantum.fidelity.avg", np.mean(fidelities)),
                MetricPoint(timestamp, "quantum.fidelity.median", np.median(fidelities)),
                MetricPoint(timestamp, "quantum.fidelity.min", np.min(fidelities)),
                MetricPoint(timestamp, "quantum.fidelity.max", np.max(fidelities)),
            ])

        # Error rate
        if self.circuit_count > 0:
            error_rate = self.quantum_errors / self.circuit_count
            metrics.append(MetricPoint(timestamp, "quantum.error_rate", error_rate))

        # Circuit complexity metrics (mock data for demonstration)
        metrics.extend([
            MetricPoint(timestamp, "quantum.circuit_depth.avg", np.random.normal(50, 10)),
            MetricPoint(timestamp, "quantum.gate_count.avg", np.random.normal(200, 50)),
            MetricPoint(timestamp, "quantum.qubit_count.avg", np.random.normal(4, 1)),
        ])

        return metrics

    def record_circuit_execution(self, execution_time: float, fidelity: float = None, success: bool = True):
        """Record a quantum circuit execution."""
        self.circuit_count += 1
        self.execution_times.append(execution_time)

        if fidelity is not None:
            self.fidelity_measurements.append(fidelity)

        if not success:
            self.quantum_errors += 1


class PrivacyMetricsCollector(MetricsCollector):
    """Collects privacy and security metrics."""

    def __init__(self, collection_interval: float = 300.0):  # 5 minutes
        super().__init__("privacy_metrics", collection_interval)
        self.privacy_queries = 0
        self.privacy_violations = 0
        self.epsilon_usage = deque(maxlen=1000)
        self.audit_events = 0

    def collect_metrics(self) -> List[MetricPoint]:
        """Collect privacy and security metrics."""
        timestamp = datetime.now()
        metrics = []

        # Privacy query metrics
        metrics.extend([
            MetricPoint(timestamp, "privacy.queries.total", self.privacy_queries),
            MetricPoint(timestamp, "privacy.violations.total", self.privacy_violations),
            MetricPoint(timestamp, "privacy.audit_events", self.audit_events),
        ])

        # Privacy budget usage
        if self.epsilon_usage:
            epsilons = list(self.epsilon_usage)
            metrics.extend([
                MetricPoint(timestamp, "privacy.epsilon.avg", np.mean(epsilons)),
                MetricPoint(timestamp, "privacy.epsilon.total", np.sum(epsilons)),
                MetricPoint(timestamp, "privacy.epsilon.max", np.max(epsilons)),
            ])

        # Privacy violation rate
        if self.privacy_queries > 0:
            violation_rate = self.privacy_violations / self.privacy_queries
            metrics.append(MetricPoint(timestamp, "privacy.violation_rate", violation_rate))

        # Compliance metrics (mock data for demonstration)
        metrics.extend([
            MetricPoint(timestamp, "privacy.gdpr_compliance", 0.98),  # 98% compliant
            MetricPoint(timestamp, "privacy.ccpa_compliance", 0.95),  # 95% compliant
            MetricPoint(timestamp, "privacy.hipaa_compliance", 0.97),  # 97% compliant
        ])

        return metrics

    def record_privacy_query(self, epsilon_used: float, violation: bool = False):
        """Record a privacy-preserving query."""
        self.privacy_queries += 1
        self.epsilon_usage.append(epsilon_used)

        if violation:
            self.privacy_violations += 1

    def record_audit_event(self):
        """Record an audit event."""
        self.audit_events += 1


class FederatedMetricsCollector(MetricsCollector):
    """Collects federated learning performance metrics."""

    def __init__(self, collection_interval: float = 60.0):
        super().__init__("federated_metrics", collection_interval)
        self.rounds_completed = 0
        self.clients_participated = 0
        self.model_updates_received = 0
        self.communication_cost = 0.0
        self.round_times = deque(maxlen=100)

    def collect_metrics(self) -> List[MetricPoint]:
        """Collect federated learning metrics."""
        timestamp = datetime.now()
        metrics = []

        # Round metrics
        metrics.extend([
            MetricPoint(timestamp, "federated.rounds_completed", self.rounds_completed),
            MetricPoint(timestamp, "federated.clients_participated", self.clients_participated),
            MetricPoint(timestamp, "federated.model_updates_received", self.model_updates_received),
            MetricPoint(timestamp, "federated.communication_cost", self.communication_cost, {"unit": "MB"}),
        ])

        # Round time metrics
        if self.round_times:
            round_times = list(self.round_times)
            metrics.extend([
                MetricPoint(timestamp, "federated.round_time.avg", np.mean(round_times), {"unit": "seconds"}),
                MetricPoint(timestamp, "federated.round_time.median", np.median(round_times), {"unit": "seconds"}),
                MetricPoint(timestamp, "federated.round_time.p95", np.percentile(round_times, 95), {"unit": "seconds"}),
            ])

        # Participation rate
        if self.rounds_completed > 0:
            avg_clients_per_round = self.clients_participated / self.rounds_completed
            metrics.append(MetricPoint(timestamp, "federated.participation_rate", avg_clients_per_round))

        # Communication efficiency
        if self.model_updates_received > 0:
            avg_comm_per_update = self.communication_cost / self.model_updates_received
            metrics.append(MetricPoint(timestamp, "federated.comm_per_update", avg_comm_per_update, {"unit": "MB"}))

        # Convergence metrics (mock data for demonstration)
        metrics.extend([
            MetricPoint(timestamp, "federated.model_convergence", np.random.uniform(0.8, 0.99)),
            MetricPoint(timestamp, "federated.client_drift", np.random.uniform(0.01, 0.1)),
        ])

        return metrics

    def record_round_completion(self, round_time: float, n_clients: int, comm_cost: float):
        """Record completion of a federated learning round."""
        self.rounds_completed += 1
        self.clients_participated += n_clients
        self.round_times.append(round_time)
        self.communication_cost += comm_cost

    def record_model_update(self):
        """Record receipt of a model update."""
        self.model_updates_received += 1


class MetricsStorage:
    """Storage system for metrics data."""

    def __init__(self, storage_type: str = "memory", config: Dict[str, Any] = None):
        self.storage_type = storage_type
        self.config = config or {}
        self.series: Dict[str, MetricSeries] = {}
        self._lock = threading.Lock()

        if storage_type == "sqlite":
            self._init_sqlite()
        elif storage_type == "csv":
            self._init_csv()

    def _init_sqlite(self):
        """Initialize SQLite storage."""
        self.db_path = self.config.get("db_path", "metrics.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    name TEXT,
                    value REAL,
                    tags TEXT,
                    metadata TEXT
                )
            ''')

    def _init_csv(self):
        """Initialize CSV storage."""
        self.csv_path = self.config.get("csv_path", "metrics.csv")
        # CSV will be created when first metric is stored

    def store_metric(self, metric: MetricPoint):
        """Store a metric point."""
        with self._lock:
            # Store in memory series
            if metric.name not in self.series:
                self.series[metric.name] = MetricSeries(metric.name)
            self.series[metric.name].add_point(metric)

            # Store in persistent storage
            if self.storage_type == "sqlite":
                self._store_sqlite(metric)
            elif self.storage_type == "csv":
                self._store_csv(metric)

    def _store_sqlite(self, metric: MetricPoint):
        """Store metric in SQLite database."""
        with self.conn:
            self.conn.execute('''
                INSERT INTO metrics (timestamp, name, value, tags, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.name,
                metric.value if isinstance(metric.value, (int, float)) else str(metric.value),
                json.dumps(metric.tags),
                json.dumps(metric.metadata)
            ))

    def _store_csv(self, metric: MetricPoint):
        """Store metric in CSV file."""
        # This is a simplified implementation
        # In practice, you'd want proper CSV handling with headers
        pass

    def query_metrics(self, name: str, start_time: datetime = None, end_time: datetime = None) -> List[MetricPoint]:
        """Query metrics by name and time range."""
        if name not in self.series:
            return []

        series = self.series[name]
        points = list(series.points)

        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]

        return points

    def get_aggregated_metric(self, name: str, aggregator: str = "mean",
                            duration: timedelta = None) -> Optional[float]:
        """Get aggregated metric value."""
        if name not in self.series:
            return None

        series = self.series[name]

        if duration:
            points = series.get_recent_points(duration)
        else:
            points = list(series.points)

        if not points:
            return None

        values = [p.value for p in points if isinstance(p.value, (int, float))]
        if not values:
            return None

        if aggregator == "mean":
            return np.mean(values)
        elif aggregator == "sum":
            return np.sum(values)
        elif aggregator == "max":
            return np.max(values)
        elif aggregator == "min":
            return np.min(values)
        elif aggregator == "count":
            return len(values)
        else:
            return np.mean(values)


class MetricsDashboard:
    """Real-time metrics dashboard."""

    def __init__(self, collectors: List[MetricsCollector], storage: MetricsStorage):
        self.collectors = collectors
        self.storage = storage
        self.is_running = False
        self.dashboard_thread: Optional[threading.Thread] = None

    def start_dashboard(self, update_interval: float = 5.0):
        """Start the metrics dashboard."""
        if self.is_running:
            return

        self.is_running = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop,
                                               args=(update_interval,), daemon=True)
        self.dashboard_thread.start()
        logger.info("Started metrics dashboard")

    def stop_dashboard(self):
        """Stop the metrics dashboard."""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5.0)
        logger.info("Stopped metrics dashboard")

    def _dashboard_loop(self, update_interval: float):
        """Main dashboard update loop."""
        while self.is_running:
            try:
                self._update_dashboard()
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")

            time.sleep(update_interval)

    def _update_dashboard(self):
        """Update dashboard with current metrics."""
        # This would typically update a web dashboard or display
        # For now, we'll just log key metrics

        # System metrics
        cpu_percent = self.storage.get_aggregated_metric("system.cpu.percent")
        memory_percent = self.storage.get_aggregated_metric("system.memory.percent")

        # Application metrics
        request_count = self.storage.get_aggregated_metric("app.requests.total", aggregator="sum")
        error_rate = self.storage.get_aggregated_metric("app.error_rate")

        # Quantum metrics
        circuits_executed = self.storage.get_aggregated_metric("quantum.circuits.executed", aggregator="sum")
        quantum_error_rate = self.storage.get_aggregated_metric("quantum.error_rate")

        # Privacy metrics
        privacy_queries = self.storage.get_aggregated_metric("privacy.queries.total", aggregator="sum")
        violation_rate = self.storage.get_aggregated_metric("privacy.violation_rate")

        # Federated metrics
        rounds_completed = self.storage.get_aggregated_metric("federated.rounds_completed", aggregator="sum")

        # Log dashboard summary
        logger.info("📊 Metrics Dashboard Update:")
        logger.info(f"   System: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
        logger.info(f"   App: {request_count:.0f} requests, Error rate {error_rate:.3f}")
        logger.info(f"   Quantum: {circuits_executed:.0f} circuits, Error rate {quantum_error_rate:.3f}")
        logger.info(f"   Privacy: {privacy_queries:.0f} queries, Violation rate {violation_rate:.3f}")
        logger.info(f"   Federated: {rounds_completed:.0f} rounds completed")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return {
            "system": {
                "cpu_percent": self.storage.get_aggregated_metric("system.cpu.percent"),
                "memory_percent": self.storage.get_aggregated_metric("system.memory.percent"),
                "disk_percent": self.storage.get_aggregated_metric("system.disk.percent"),
            },
            "application": {
                "total_requests": self.storage.get_aggregated_metric("app.requests.total", aggregator="sum"),
                "active_requests": self.storage.get_aggregated_metric("app.requests.active"),
                "error_rate": self.storage.get_aggregated_metric("app.error_rate"),
                "avg_response_time": self.storage.get_aggregated_metric("app.response_time.avg"),
            },
            "quantum": {
                "circuits_executed": self.storage.get_aggregated_metric("quantum.circuits.executed", aggregator="sum"),
                "error_rate": self.storage.get_aggregated_metric("quantum.error_rate"),
                "avg_fidelity": self.storage.get_aggregated_metric("quantum.fidelity.avg"),
            },
            "privacy": {
                "total_queries": self.storage.get_aggregated_metric("privacy.queries.total", aggregator="sum"),
                "violation_rate": self.storage.get_aggregated_metric("privacy.violation_rate"),
                "epsilon_usage": self.storage.get_aggregated_metric("privacy.epsilon.avg"),
            },
            "federated": {
                "rounds_completed": self.storage.get_aggregated_metric("federated.rounds_completed", aggregator="sum"),
                "participation_rate": self.storage.get_aggregated_metric("federated.participation_rate"),
                "communication_cost": self.storage.get_aggregated_metric("federated.communication_cost", aggregator="sum"),
            }
        }


class MetricsExporter:
    """Export metrics to external monitoring systems."""

    def __init__(self, storage: MetricsStorage):
        self.storage = storage

    def export_to_prometheus(self, filename: str = "metrics.prom"):
        """Export metrics in Prometheus format."""
        lines = []

        for series_name, series in self.storage.series.items():
            # Get latest value
            if series.points:
                latest_point = series.points[-1]
                if isinstance(latest_point.value, (int, float)):
                    # Format as Prometheus metric
                    metric_name = series_name.replace(".", "_")
                    lines.append(f"# HELP {metric_name} {series_name}")
                    lines.append(f"# TYPE {metric_name} gauge")
                    lines.append(f"{metric_name} {latest_point.value}")

        with open(filename, 'w') as f:
            f.write("\\n".join(lines))

        logger.info(f"Exported metrics to {filename}")

    def export_to_json(self, filename: str = "metrics.json"):
        """Export metrics as JSON."""
        export_data = {}

        for series_name, series in self.storage.series.items():
            points_data = []
            for point in series.points:
                points_data.append({
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "tags": point.tags,
                    "metadata": point.metadata
                })

            export_data[series_name] = {
                "points": points_data,
                "aggregation_window": series.aggregation_window.total_seconds()
            }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported metrics to {filename}")

    def export_to_csv(self, filename: str = "metrics.csv"):
        """Export metrics as CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "name", "value", "tags", "metadata"])

            for series_name, series in self.storage.series.items():
                for point in series.points:
                    writer.writerow([
                        point.timestamp.isoformat(),
                        series_name,
                        point.value,
                        json.dumps(point.tags),
                        json.dumps(point.metadata)
                    ])

        logger.info(f"Exported metrics to {filename}")


def create_comprehensive_monitoring_system() -> Tuple[List[MetricsCollector], MetricsStorage, MetricsDashboard]:
    """
    Create a comprehensive monitoring system with all collectors.

    Returns:
        Tuple of (collectors, storage, dashboard)
    """
    # Create collectors
    collectors = [
        SystemMetricsCollector(),
        ApplicationMetricsCollector(),
        QuantumMetricsCollector(),
        PrivacyMetricsCollector(),
        FederatedMetricsCollector()
    ]

    # Create storage
    storage = MetricsStorage(storage_type="memory")

    # Create dashboard
    dashboard = MetricsDashboard(collectors, storage)

    # Connect collectors to storage
    for collector in collectors:
        # In practice, you'd set up proper metric forwarding
        # For this demo, collectors will handle their own logic
        pass

    return collectors, storage, dashboard


def main():
    """Demonstrate comprehensive metrics collection."""
    print("📊 Comprehensive Metrics Collection Demo")
    print("=" * 50)

    # Create monitoring system
    collectors, storage, dashboard = create_comprehensive_monitoring_system()

    # Start collectors
    print("\\n▶️  Starting metrics collectors...")
    for collector in collectors:
        collector.start_collection()

    # Start dashboard
    print("▶️  Starting metrics dashboard...")
    dashboard.start_dashboard(update_interval=10.0)

    # Simulate some activity
    print("\\n🔄 Simulating system activity...")

    # Simulate application requests
    app_collector = collectors[1]  # ApplicationMetricsCollector
    for i in range(50):
        response_time = np.random.normal(50, 10)  # ms
        success = np.random.random() > 0.05  # 95% success rate

        app_collector.record_request(response_time, success)

        if np.random.random() < 0.1:  # 10% of requests are active
            app_collector.increment_active_requests()
            time.sleep(0.1)
            app_collector.decrement_active_requests()

    # Simulate quantum operations
    quantum_collector = collectors[2]  # QuantumMetricsCollector
    for i in range(20):
        execution_time = np.random.normal(100, 20)  # ms
        fidelity = np.random.uniform(0.85, 0.99)
        success = np.random.random() > 0.1  # 90% success rate

        quantum_collector.record_circuit_execution(execution_time, fidelity, success)

    # Simulate privacy operations
    privacy_collector = collectors[3]  # PrivacyMetricsCollector
    for i in range(30):
        epsilon_used = np.random.uniform(0.1, 2.0)
        violation = np.random.random() < 0.02  # 2% violation rate

        privacy_collector.record_privacy_query(epsilon_used, violation)

        if np.random.random() < 0.3:  # 30% of queries trigger audit
            privacy_collector.record_audit_event()

    # Simulate federated learning
    fed_collector = collectors[4]  # FederatedMetricsCollector
    for round_num in range(5):
        round_time = np.random.normal(120, 30)  # seconds
        n_clients = np.random.randint(3, 8)
        comm_cost = np.random.normal(50, 20)  # MB

        fed_collector.record_round_completion(round_time, n_clients, comm_cost)

        # Record individual model updates
        for _ in range(n_clients):
            fed_collector.record_model_update()

    # Let dashboard run for a bit
    print("⏳ Running monitoring for 30 seconds...")
    time.sleep(30)

    # Stop everything
    print("\\n⏹️  Stopping monitoring system...")
    dashboard.stop_dashboard()

    for collector in collectors:
        collector.stop_collection()

    # Export metrics
    exporter = MetricsExporter(storage)
    print("\\n💾 Exporting metrics...")

    exporter.export_to_json("comprehensive_metrics.json")
    exporter.export_to_csv("comprehensive_metrics.csv")

    # Display final dashboard data
    print("\\n📋 Final Dashboard Summary")
    print("=" * 40)

    dashboard_data = dashboard.get_dashboard_data()

    print("\\n🔧 System Metrics:")
    sys_metrics = dashboard_data['system']
    print(f"   CPU Usage: {sys_metrics['cpu_percent']:.1f}%")
    print(f"   Memory Usage: {sys_metrics['memory_percent']:.1f}%")
    print(f"   Disk Usage: {sys_metrics['disk_percent']:.1f}%")

    print("\\n🚀 Application Metrics:")
    app_metrics = dashboard_data['application']
    print(f"   Total Requests: {app_metrics['total_requests']:.0f}")
    print(f"   Active Requests: {app_metrics['active_requests']:.0f}")
    print(f"   Error Rate: {app_metrics['error_rate']:.3f}")
    print(f"   Avg Response Time: {app_metrics['avg_response_time']:.1f}ms")

    print("\\n🧠 Quantum Metrics:")
    quantum_metrics = dashboard_data['quantum']
    print(f"   Circuits Executed: {quantum_metrics['circuits_executed']:.0f}")
    print(f"   Error Rate: {quantum_metrics['error_rate']:.3f}")
    print(f"   Avg Fidelity: {quantum_metrics['avg_fidelity']:.3f}")

    print("\\n🔒 Privacy Metrics:")
    privacy_metrics = dashboard_data['privacy']
    print(f"   Total Queries: {privacy_metrics['total_queries']:.0f}")
    print(f"   Violation Rate: {privacy_metrics['violation_rate']:.3f}")
    print(f"   Avg Epsilon Usage: {privacy_metrics['epsilon_usage']:.3f}")

    print("\\n🌐 Federated Learning Metrics:")
    fed_metrics = dashboard_data['federated']
    print(f"   Rounds Completed: {fed_metrics['rounds_completed']:.0f}")
    print(f"   Participation Rate: {fed_metrics['participation_rate']:.2f}")
    print(f"   Communication Cost: {fed_metrics['communication_cost']:.1f} MB")

    print("\\n✅ Comprehensive metrics collection demo completed!")
    print("📊 Metrics exported to 'comprehensive_metrics.json' and 'comprehensive_metrics.csv'")


if __name__ == "__main__":
    main()

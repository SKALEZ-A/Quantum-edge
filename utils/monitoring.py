"""
Quantum Edge AI Platform - Monitoring Utilities

Comprehensive monitoring, metrics collection, and alerting system
for edge AI deployments with real-time performance tracking.
"""

import psutil
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import deque
import statistics
import smtplib
from email.mime.text import MIMEText
import requests
import queue

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"          # Point-in-time value
    COUNTER = "counter"      # Monotonically increasing value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"      # Quantiles over sliding time window

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """Alert instance"""
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: Any
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_available: int
    disk_percent: float
    disk_used: int
    disk_free: int
    network_sent: int
    network_recv: int
    timestamp: datetime

class MetricsCollector:
    """Advanced metrics collection system"""

    def __init__(self, collection_interval: int = 60, max_history: int = 1000):
        self.collection_interval = collection_interval
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.custom_metrics = {}
        self.is_collecting = False
        self.collection_thread = None
        self.lock = threading.Lock()

    def start_collection(self):
        """Start metrics collection"""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                metrics = self._collect_system_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)

                # Collect custom metrics
                self._collect_custom_metrics()

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)

        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used
        memory_available = memory.available

        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used = disk.used
        disk_free = disk.free

        network = psutil.net_io_counters()
        network_sent = network.bytes_sent
        network_recv = network.bytes_recv

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            memory_available=memory_available,
            disk_percent=disk_percent,
            disk_used=disk_used,
            disk_free=disk_free,
            network_sent=network_sent,
            network_recv=network_recv,
            timestamp=datetime.utcnow()
        )

    def _collect_custom_metrics(self):
        """Collect custom metrics"""
        for metric_name, collector_func in self.custom_metrics.items():
            try:
                value = collector_func()
                metric = Metric(
                    name=metric_name,
                    value=value,
                    timestamp=datetime.utcnow()
                )
                with self.lock:
                    self.metrics_history.append(metric)
            except Exception as e:
                logger.error(f"Error collecting custom metric {metric_name}: {e}")

    def add_custom_metric(self, name: str, collector_func: Callable):
        """Add custom metric collector"""
        self.custom_metrics[name] = collector_func
        logger.info(f"Added custom metric: {name}")

    def record_metric(self, name: str, value: Union[int, float],
                     labels: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a custom metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metric_type=metric_type
        )

        with self.lock:
            self.metrics_history.append(metric)

    def get_recent_metrics(self, hours: int = 1) -> List[Union[SystemMetrics, Metric]]:
        """Get recent metrics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self.lock:
            recent = [m for m in self.metrics_history
                     if m.timestamp > cutoff_time]

        return recent

    def get_metric_statistics(self, metric_name: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for metrics"""
        recent_metrics = self.get_recent_metrics(hours)

        if metric_name:
            # Filter by metric name
            metrics = [m for m in recent_metrics
                      if isinstance(m, Metric) and m.name == metric_name]
        else:
            # Use system metrics
            metrics = [m for m in recent_metrics if isinstance(m, SystemMetrics)]

        if not metrics:
            return {}

        # Extract numeric values
        values = []
        for metric in metrics:
            if isinstance(metric, SystemMetrics):
                # Use CPU and memory as representative metrics
                values.extend([metric.cpu_percent, metric.memory_percent])
            elif isinstance(metric, Metric):
                values.append(metric.value)

        if not values:
            return {}

        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'percentiles': {
                '25': statistics.quantiles(values, n=4)[0] if len(values) >= 4 else None,
                '50': statistics.quantiles(values, n=4)[1] if len(values) >= 4 else None,
                '75': statistics.quantiles(values, n=4)[2] if len(values) >= 4 else None,
                '95': statistics.quantiles(values, n=4)[3] if len(values) >= 4 else None,
            }
        }

    def export_metrics(self, output_path: str, format: str = "json"):
        """Export metrics data"""
        with self.lock:
            metrics_data = list(self.metrics_history)

        if format == "json":
            # Convert to serializable format
            serializable_data = []
            for metric in metrics_data:
                if isinstance(metric, SystemMetrics):
                    data = {
                        'type': 'system',
                        'cpu_percent': metric.cpu_percent,
                        'memory_percent': metric.memory_percent,
                        'memory_used': metric.memory_used,
                        'memory_available': metric.memory_available,
                        'disk_percent': metric.disk_percent,
                        'disk_used': metric.disk_used,
                        'disk_free': metric.disk_free,
                        'network_sent': metric.network_sent,
                        'network_recv': metric.network_recv,
                        'timestamp': metric.timestamp.isoformat()
                    }
                elif isinstance(metric, Metric):
                    data = {
                        'type': 'custom',
                        'name': metric.name,
                        'value': metric.value,
                        'labels': metric.labels,
                        'metric_type': metric.metric_type.value,
                        'timestamp': metric.timestamp.isoformat()
                    }
                serializable_data.append(data)

            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(['timestamp', 'type', 'name', 'value', 'labels'])

                # Write data
                for metric in metrics_data:
                    if isinstance(metric, SystemMetrics):
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            'system',
                            'cpu_percent',
                            metric.cpu_percent,
                            json.dumps({})
                        ])
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            'system',
                            'memory_percent',
                            metric.memory_percent,
                            json.dumps({})
                        ])
                    elif isinstance(metric, Metric):
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            'custom',
                            metric.name,
                            metric.value,
                            json.dumps(metric.labels)
                        ])

        logger.info(f"Metrics exported to {output_path}")

class AlertManager:
    """Alert management system"""

    def __init__(self):
        self.rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_channels = []
        self.lock = threading.Lock()

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    def evaluate_rules(self, metrics_collector: MetricsCollector):
        """Evaluate all alert rules"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue

            # Check cooldown period
            if rule.last_triggered and \
               (datetime.utcnow() - rule.last_triggered).seconds < rule.cooldown_period:
                continue

            try:
                # Evaluate condition
                if self._evaluate_condition(rule.condition, metrics_collector):
                    self._trigger_alert(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")

    def _evaluate_condition(self, condition: str, metrics_collector: MetricsCollector) -> bool:
        """Evaluate alert condition"""
        # Create evaluation context
        context = {
            'metrics': metrics_collector,
            'get_recent_metrics': metrics_collector.get_recent_metrics,
            'get_metric_stats': metrics_collector.get_metric_statistics,
            'time': time,
            'datetime': datetime,
            'timedelta': timedelta
        }

        # Add common functions
        context.update({
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'avg': lambda x: sum(x)/len(x) if x else 0
        })

        try:
            result = eval(condition, {"__builtins__": {}}, context)
            return bool(result)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=rule.description,
            value=None,  # Could be populated with actual values
            timestamp=datetime.utcnow()
        )

        with self.lock:
            self.active_alerts[rule.name] = alert
            self.alert_history.append(alert)

        rule.last_triggered = datetime.utcnow()

        # Send notifications
        self._send_notifications(alert)

        logger.warning(f"Alert triggered: {rule.name} - {rule.description}")

    def resolve_alert(self, rule_name: str):
        """Resolve an alert"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

            del self.active_alerts[rule_name]

            logger.info(f"Alert resolved: {rule_name}")

    def add_notification_channel(self, channel_func: Callable[[Alert], None]):
        """Add notification channel"""
        self.notification_channels.append(channel_func)

    def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self.lock:
            return [alert for alert in self.alert_history
                   if alert.timestamp > cutoff_time]

class NotificationChannels:
    """Built-in notification channels"""

    @staticmethod
    def email_channel(smtp_server: str, smtp_port: int, username: str,
                     password: str, from_addr: str, to_addrs: List[str]):
        """Create email notification channel"""
        def send_email(alert: Alert):
            msg = MIMEText(f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Message: {alert.message}
Timestamp: {alert.timestamp.isoformat()}

This is an automated alert from Quantum Edge AI Platform.
            """)

            msg['Subject'] = f"Alert: {alert.rule_name} - {alert.severity.value.upper()}"
            msg['From'] = from_addr
            msg['To'] = ', '.join(to_addrs)

            try:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(username, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
                server.quit()
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")

        return send_email

    @staticmethod
    def webhook_channel(webhook_url: str, headers: Dict[str, str] = None):
        """Create webhook notification channel"""
        def send_webhook(alert: Alert):
            payload = {
                'alert_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'labels': alert.labels
            }

            try:
                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers=headers or {'Content-Type': 'application/json'},
                    timeout=10
                )
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")

        return send_webhook

    @staticmethod
    def log_channel(logger_name: str = None):
        """Create logging notification channel"""
        log = logging.getLogger(logger_name or __name__)

        def log_alert(alert: Alert):
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)

            log.log(log_level, f"ALERT: {alert.rule_name} - {alert.message}")

        return log_alert

class MonitoringDashboard:
    """Real-time monitoring dashboard"""

    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.dashboard_data = {}
        self.update_interval = 30  # seconds
        self.is_running = False
        self.dashboard_thread = None

    def start_dashboard(self):
        """Start dashboard updates"""
        if self.is_running:
            return

        self.is_running = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()
        logger.info("Monitoring dashboard started")

    def stop_dashboard(self):
        """Stop dashboard updates"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join()
        logger.info("Monitoring dashboard stopped")

    def _dashboard_loop(self):
        """Dashboard update loop"""
        while self.is_running:
            try:
                self._update_dashboard_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                time.sleep(self.update_interval)

    def _update_dashboard_data(self):
        """Update dashboard data"""
        current_time = datetime.utcnow()

        # Get recent metrics
        recent_metrics = self.metrics_collector.get_recent_metrics(hours=1)

        # Calculate summary statistics
        if recent_metrics:
            system_metrics = [m for m in recent_metrics if isinstance(m, SystemMetrics)]
            custom_metrics = [m for m in recent_metrics if isinstance(m, Metric)]

            self.dashboard_data = {
                'timestamp': current_time.isoformat(),
                'system_metrics': {
                    'latest': system_metrics[-1] if system_metrics else None,
                    'avg_cpu': statistics.mean([m.cpu_percent for m in system_metrics]) if system_metrics else 0,
                    'avg_memory': statistics.mean([m.memory_percent for m in system_metrics]) if system_metrics else 0,
                    'count': len(system_metrics)
                },
                'custom_metrics': {
                    'count': len(custom_metrics),
                    'latest_values': {m.name: m.value for m in custom_metrics[-10:]}  # Last 10
                },
                'alerts': {
                    'active_count': len(self.alert_manager.get_active_alerts()),
                    'recent_count': len(self.alert_manager.get_alert_history(hours=1))
                },
                'performance_indicators': self._calculate_performance_indicators(recent_metrics)
            }

    def _calculate_performance_indicators(self, metrics: List) -> Dict[str, Any]:
        """Calculate performance indicators"""
        system_metrics = [m for m in metrics if isinstance(m, SystemMetrics)]

        if not system_metrics:
            return {}

        cpu_values = [m.cpu_percent for m in system_metrics]
        memory_values = [m.memory_percent for m in system_metrics]

        return {
            'cpu_health': 'good' if statistics.mean(cpu_values) < 70 else 'warning' if statistics.mean(cpu_values) < 90 else 'critical',
            'memory_health': 'good' if statistics.mean(memory_values) < 80 else 'warning' if statistics.mean(memory_values) < 95 else 'critical',
            'system_load': 'low' if statistics.mean(cpu_values) < 50 else 'medium' if statistics.mean(cpu_values) < 80 else 'high',
            'trend_cpu': 'stable' if abs(cpu_values[-1] - statistics.mean(cpu_values)) < 10 else 'increasing' if cpu_values[-1] > statistics.mean(cpu_values) else 'decreasing',
            'trend_memory': 'stable' if abs(memory_values[-1] - statistics.mean(memory_values)) < 10 else 'increasing' if memory_values[-1] > statistics.mean(memory_values) else 'decreasing'
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()

    def export_dashboard_report(self, output_path: str):
        """Export dashboard report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'dashboard_data': self.get_dashboard_data(),
            'system_info': {
                'platform': psutil.platform(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Dashboard report exported to {output_path}")

# Convenience functions for quick setup
def create_basic_monitoring_system() -> Tuple[MetricsCollector, AlertManager, MonitoringDashboard]:
    """Create a basic monitoring system"""
    # Create components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    dashboard = MonitoringDashboard(metrics_collector, alert_manager)

    # Add basic alert rules
    alert_rules = [
        AlertRule(
            name="high_cpu_usage",
            condition="get_metric_stats()['mean'] > 90",
            severity=AlertSeverity.WARNING,
            description="CPU usage is above 90%"
        ),
        AlertRule(
            name="high_memory_usage",
            condition="len(get_recent_metrics(1)) > 0 and get_recent_metrics(1)[-1].memory_percent > 95",
            severity=AlertSeverity.CRITICAL,
            description="Memory usage is above 95%"
        ),
        AlertRule(
            name="low_disk_space",
            condition="len(get_recent_metrics(1)) > 0 and get_recent_metrics(1)[-1].disk_percent > 90",
            severity=AlertSeverity.ERROR,
            description="Disk space is below 10%"
        )
    ]

    for rule in alert_rules:
        alert_manager.add_rule(rule)

    # Add logging notification channel
    alert_manager.add_notification_channel(NotificationChannels.log_channel())

    return metrics_collector, alert_manager, dashboard

def start_full_monitoring():
    """Start full monitoring system"""
    collector, alert_manager, dashboard = create_basic_monitoring_system()

    # Start all components
    collector.start_collection()
    dashboard.start_dashboard()

    # Set up alert evaluation (run in background)
    def alert_loop():
        while True:
            alert_manager.evaluate_rules(collector)
            time.sleep(60)  # Check every minute

    alert_thread = threading.Thread(target=alert_loop, daemon=True)
    alert_thread.start()

    logger.info("Full monitoring system started")

    return collector, alert_manager, dashboard

"""
Quantum Edge AI Platform - Audit Logging Module

Comprehensive audit logging system for compliance, security monitoring,
and forensic analysis in edge AI systems.
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import os
from pathlib import Path
import gzip
import sqlite3
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY_INCIDENT = "security_incident"
    PRIVACY_VIOLATION = "privacy_violation"
    SYSTEM_CHANGE = "system_change"
    COMPLIANCE_CHECK = "compliance_check"
    MODEL_UPDATE = "model_update"
    INFERENCE_REQUEST = "inference_request"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditOutcome(Enum):
    """Audit event outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"
    WARNING = "warning"

@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    outcome: AuditOutcome
    timestamp: datetime
    user_id: str
    session_id: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # lat, lon
    compliance_flags: List[str] = field(default_factory=list)
    hash_chain: Optional[str] = None  # For tamper detection

@dataclass
class AuditLogEntry:
    """Audit log entry with integrity protection"""
    event: AuditEvent
    signature: str
    previous_hash: str
    sequence_number: int
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuditQuery:
    """Audit log query parameters"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: List[AuditEventType] = field(default_factory=list)
    user_ids: List[str] = field(default_factory=list)
    severities: List[AuditSeverity] = field(default_factory=list)
    outcomes: List[AuditOutcome] = field(default_factory=list)
    resource_ids: List[str] = field(default_factory=list)
    limit: int = 1000
    offset: int = 0

class TamperDetection:
    """Tamper detection using hash chains"""

    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.last_hash = self._initial_hash()

    def _initial_hash(self) -> str:
        """Generate initial hash for chain"""
        initial_data = f"audit_chain_start_{int(time.time())}"
        return hashlib.sha256(initial_data.encode()).hexdigest()

    def create_hash_chain(self, event_data: str, previous_hash: str) -> str:
        """Create next hash in chain"""
        combined = f"{previous_hash}:{event_data}"
        h = hmac.HMAC(self.secret_key, hashes.SHA256(), default_backend())
        h.update(combined.encode())
        return h.finalize().hex()

    def verify_chain_integrity(self, events: List[AuditLogEntry]) -> bool:
        """Verify integrity of hash chain"""
        current_hash = self._initial_hash()

        for entry in sorted(events, key=lambda x: x.sequence_number):
            expected_hash = self.create_hash_chain(
                self._event_to_string(entry.event),
                entry.previous_hash
            )

            if expected_hash != entry.hash_chain:
                return False

            current_hash = expected_hash

        return True

    def _event_to_string(self, event: AuditEvent) -> str:
        """Convert event to string for hashing"""
        event_dict = asdict(event)
        # Remove hash_chain to avoid circular dependency
        event_dict.pop('hash_chain', None)
        return json.dumps(event_dict, sort_keys=True, default=str)

class AuditStorage:
    """Audit log storage with integrity protection"""

    def __init__(self, storage_path: str, max_file_size: int = 104857600):  # 100MB
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.current_file = None
        self.sequence_number = 0
        self.tamper_detection = TamperDetection(secrets.token_bytes(32))

        # Initialize database for indexing
        self.db_path = self.storage_path / "audit_index.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for audit indexing"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    severity TEXT,
                    outcome TEXT,
                    timestamp TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    resource_id TEXT,
                    action TEXT,
                    sequence_number INTEGER,
                    file_path TEXT
                )
            ''')

            # Create indexes for efficient querying
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_resource_id ON audit_events(resource_id)')

    def store_event(self, event: AuditEvent) -> AuditLogEntry:
        """Store audit event with integrity protection"""
        # Get current sequence number
        self.sequence_number += 1

        # Create hash chain
        event_str = self.tamper_detection._event_to_string(event)
        hash_chain = self.tamper_detection.create_hash_chain(
            event_str,
            self.tamper_detection.last_hash
        )

        # Create log entry
        log_entry = AuditLogEntry(
            event=event,
            signature=self._sign_event(event_str),
            previous_hash=self.tamper_detection.last_hash,
            sequence_number=self.sequence_number
        )

        # Set hash chain on event
        event.hash_chain = hash_chain

        # Update last hash
        self.tamper_detection.last_hash = hash_chain

        # Store in file
        self._store_to_file(log_entry)

        # Index in database
        self._index_event(event, self.sequence_number)

        return log_entry

    def _store_to_file(self, log_entry: AuditLogEntry):
        """Store log entry to file"""
        # Check if we need a new file
        if self.current_file is None or self._get_file_size() >= self.max_file_size:
            self._rotate_file()

        # Append to current file
        with open(self.current_file, 'a', encoding='utf-8') as f:
            json.dump(asdict(log_entry), f, default=str)
            f.write('\n')

    def _rotate_file(self):
        """Rotate to new log file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.storage_path / f"audit_log_{timestamp}.json"

    def _get_file_size(self) -> int:
        """Get current file size"""
        if self.current_file and self.current_file.exists():
            return self.current_file.stat().st_size
        return 0

    def _index_event(self, event: AuditEvent, sequence_number: int):
        """Index event in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO audit_events
                (event_id, event_type, severity, outcome, timestamp, user_id,
                 session_id, resource_id, action, sequence_number, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type.value,
                event.severity.value,
                event.outcome.value,
                event.timestamp.isoformat(),
                event.user_id,
                event.session_id,
                event.resource_id,
                event.action,
                sequence_number,
                str(self.current_file)
            ))

    def _sign_event(self, event_str: str) -> str:
        """Sign event data for integrity"""
        h = hmac.HMAC(self.tamper_detection.secret_key, hashes.SHA256(), default_backend())
        h.update(event_str.encode())
        return base64.b64encode(h.finalize()).decode()

    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        with sqlite3.connect(self.db_path) as conn:
            # Build query
            conditions = []
            params = []

            if query.start_time:
                conditions.append("timestamp >= ?")
                params.append(query.start_time.isoformat())

            if query.end_time:
                conditions.append("timestamp <= ?")
                params.append(query.end_time.isoformat())

            if query.event_types:
                placeholders = ','.join('?' * len(query.event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in query.event_types])

            if query.user_ids:
                placeholders = ','.join('?' * len(query.user_ids))
                conditions.append(f"user_id IN ({placeholders})")
                params.extend(query.user_ids)

            if query.severities:
                placeholders = ','.join('?' * len(query.severities))
                conditions.append(f"severity IN ({placeholders})")
                params.extend([s.value for s in query.severies])

            if query.outcomes:
                placeholders = ','.join('?' * len(query.outcomes))
                conditions.append(f"outcome IN ({placeholders})")
                params.extend([o.value for o in query.outcomes])

            if query.resource_ids:
                placeholders = ','.join('?' * len(query.resource_ids))
                conditions.append(f"resource_id IN ({placeholders})")
                params.extend(query.resource_ids)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            sql = f'''
                SELECT event_id, file_path, sequence_number
                FROM audit_events
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            '''

            params.extend([query.limit, query.offset])

            # Execute query
            results = []
            for row in conn.execute(sql, params):
                event = self._load_event_from_file(row[1], row[0])
                if event:
                    results.append(event)

            return results

    def _load_event_from_file(self, file_path: str, event_id: str) -> Optional[AuditEvent]:
        """Load specific event from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    log_entry_data = json.loads(line)
                    if log_entry_data['event']['event_id'] == event_id:
                        return AuditEvent(**log_entry_data['event'])
        except Exception as e:
            logger.error(f"Error loading event {event_id} from {file_path}: {e}")

        return None

    def verify_integrity(self, start_sequence: int = 1, end_sequence: Optional[int] = None) -> bool:
        """Verify integrity of stored audit logs"""
        if end_sequence is None:
            end_sequence = self.sequence_number

        # Load all entries in range
        events = []
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute('''
                SELECT file_path, event_id, sequence_number
                FROM audit_events
                WHERE sequence_number BETWEEN ? AND ?
                ORDER BY sequence_number
            ''', (start_sequence, end_sequence)):
                event = self._load_event_from_file(row[0], row[1])
                if event:
                    # Recreate log entry structure
                    events.append(AuditLogEntry(
                        event=event,
                        signature="",
                        previous_hash="",
                        sequence_number=row[2]
                    ))

        return self.tamper_detection.verify_chain_integrity(events)

    def compress_old_logs(self, days_old: int = 30):
        """Compress old log files"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        for log_file in self.storage_path.glob("audit_log_*.json"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                self._compress_file(log_file)

    def _compress_file(self, file_path: Path):
        """Compress a log file"""
        compressed_path = file_path.with_suffix('.json.gz')

        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)

        # Remove original file
        file_path.unlink()

        logger.info(f"Compressed {file_path} to {compressed_path}")

class AuditLogger:
    """Main audit logging system"""

    def __init__(self, storage_path: str = "./audit_logs"):
        self.storage = AuditStorage(storage_path)
        self.event_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        self.handlers = {}
        self.alert_thresholds = {
            AuditSeverity.CRITICAL: 0,
            AuditSeverity.HIGH: 5,
            AuditSeverity.MEDIUM: 20,
            AuditSeverity.LOW: 100
        }

    def start(self):
        """Start audit logging system"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_events)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        logger.info("Audit logging system started")

    def stop(self):
        """Stop audit logging system"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()

        logger.info("Audit logging system stopped")

    def log_event(self, event_type: AuditEventType, severity: AuditSeverity,
                  outcome: AuditOutcome, user_id: str, **kwargs) -> str:
        """Log an audit event"""
        event_id = self._generate_event_id()

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            outcome=outcome,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            **kwargs
        )

        # Add to processing queue
        self.event_queue.put(event)

        # Check for alerts
        self._check_alerts(event)

        return event_id

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"audit_{int(time.time() * 1000000)}_{secrets.token_hex(8)}"

    def _process_events(self):
        """Process events from queue"""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1)

                # Store event
                log_entry = self.storage.store_event(event)

                # Call handlers
                if event.event_type in self.handlers:
                    for handler in self.handlers[event.event_type]:
                        try:
                            handler(event)
                        except Exception as e:
                            logger.error(f"Error in audit handler: {e}")

                self.event_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")

    def register_handler(self, event_type: AuditEventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)

    def _check_alerts(self, event: AuditEvent):
        """Check if event should trigger alerts"""
        threshold = self.alert_thresholds.get(event.severity, 1000)

        # Simplified alert logic - in production, this would check recent event counts
        if event.severity == AuditSeverity.CRITICAL:
            logger.critical(f"CRITICAL AUDIT EVENT: {event.event_type.value} by {event.user_id}")
        elif event.severity == AuditSeverity.HIGH and event.outcome == AuditOutcome.FAILURE:
            logger.warning(f"HIGH PRIORITY AUDIT EVENT: {event.event_type.value} failure by {event.user_id}")

    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        return self.storage.query_events(query)

    def generate_report(self, query: AuditQuery) -> Dict[str, Any]:
        """Generate audit report"""
        events = self.query_events(query)

        # Analyze events
        report = {
            'total_events': len(events),
            'time_range': {
                'start': query.start_time.isoformat() if query.start_time else None,
                'end': query.end_time.isoformat() if query.end_time else None
            },
            'event_breakdown': {},
            'severity_breakdown': {},
            'outcome_breakdown': {},
            'user_activity': {},
            'compliance_violations': []
        }

        for event in events:
            # Event type breakdown
            et = event.event_type.value
            report['event_breakdown'][et] = report['event_breakdown'].get(et, 0) + 1

            # Severity breakdown
            sev = event.severity.value
            report['severity_breakdown'][sev] = report['severity_breakdown'].get(sev, 0) + 1

            # Outcome breakdown
            out = event.outcome.value
            report['outcome_breakdown'][out] = report['outcome_breakdown'].get(out, 0) + 1

            # User activity
            uid = event.user_id
            if uid not in report['user_activity']:
                report['user_activity'][uid] = {'total': 0, 'by_type': {}}
            report['user_activity'][uid]['total'] += 1
            report['user_activity'][uid]['by_type'][et] = report['user_activity'][uid]['by_type'].get(et, 0) + 1

            # Compliance violations
            if event.compliance_flags:
                report['compliance_violations'].extend(event.compliance_flags)

        return report

    def verify_log_integrity(self, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> bool:
        """Verify integrity of audit logs"""
        # Find sequence numbers for time range
        with sqlite3.connect(self.storage.db_path) as conn:
            conditions = []
            params = []

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            result = conn.execute(f'''
                SELECT MIN(sequence_number), MAX(sequence_number)
                FROM audit_events
                WHERE {where_clause}
            ''', params).fetchone()

            if result and result[0] and result[1]:
                return self.storage.verify_integrity(result[0], result[1])

        return True  # No events in range

    def export_logs(self, query: AuditQuery, export_path: str, format: str = "json"):
        """Export audit logs"""
        events = self.query_events(query)

        if format == "json":
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(event) for event in events], f, indent=2, default=str)
        elif format == "csv":
            import csv
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=asdict(events[0]).keys())
                    writer.writeheader()
                    for event in events:
                        writer.writerow(asdict(event))

        logger.info(f"Exported {len(events)} audit events to {export_path}")

    def cleanup_old_logs(self, retention_days: int = 2555):  # 7 years
        """Clean up old audit logs"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        # Compress old logs
        self.storage.compress_old_logs(retention_days - 30)  # Compress after 30 days less than retention

        # In production, this would archive or delete very old logs
        logger.info(f"Cleaned up audit logs older than {retention_days} days")

# Convenience functions for common audit events
def log_authentication(logger: AuditLogger, user_id: str, success: bool, **kwargs):
    """Log authentication event"""
    outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE
    severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM

    return logger.log_event(
        AuditEventType.AUTHENTICATION,
        severity,
        outcome,
        user_id,
        action="login" if success else "failed_login",
        **kwargs
    )

def log_data_access(logger: AuditLogger, user_id: str, resource_id: str,
                   action: str, **kwargs):
    """Log data access event"""
    return logger.log_event(
        AuditEventType.DATA_ACCESS,
        AuditSeverity.LOW,
        AuditOutcome.SUCCESS,
        user_id,
        resource_id=resource_id,
        action=action,
        **kwargs
    )

def log_security_incident(logger: AuditLogger, user_id: str, incident_type: str,
                         severity: AuditSeverity, **kwargs):
    """Log security incident"""
    return logger.log_event(
        AuditEventType.SECURITY_INCIDENT,
        severity,
        AuditOutcome.WARNING,
        user_id,
        action=incident_type,
        **kwargs
    )

def log_model_inference(logger: AuditLogger, user_id: str, model_id: str,
                       input_size: int, **kwargs):
    """Log model inference event"""
    return logger.log_event(
        AuditEventType.INFERENCE_REQUEST,
        AuditSeverity.LOW,
        AuditOutcome.SUCCESS,
        user_id,
        resource_id=model_id,
        action="inference",
        details={'input_size': input_size},
        **kwargs
    )

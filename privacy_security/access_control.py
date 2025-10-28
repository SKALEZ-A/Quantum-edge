"""
Quantum Edge AI Platform - Access Control Module

Advanced access control frameworks including attribute-based access control,
policy-based access control, and context-aware authorization.
"""

import re
import json
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccessControlModel(Enum):
    """Access control models"""
    MAC = "mandatory_access_control"  # MAC
    DAC = "discretionary_access_control"  # DAC
    RBAC = "role_based_access_control"  # RBAC
    ABAC = "attribute_based_access_control"  # ABAC
    PBAC = "policy_based_access_control"  # PBAC

class AccessDecision(Enum):
    """Access decisions"""
    ALLOW = "allow"
    DENY = "deny"
    INDETERMINATE = "indeterminate"

class AccessOperation(Enum):
    """Access operations"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"
    ADMIN = "admin"

@dataclass
class AccessSubject:
    """Access control subject (user/principal)"""
    id: str
    type: str  # user, service, device, etc.
    attributes: Dict[str, Any] = field(default_factory=dict)
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)

@dataclass
class AccessResource:
    """Access control resource"""
    id: str
    type: str  # model, data, api, device, etc.
    attributes: Dict[str, Any] = field(default_factory=dict)
    owner: Optional[str] = None
    classification: str = "public"  # public, internal, confidential, restricted

@dataclass
class AccessContext:
    """Access context"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    location: Optional[str] = None
    ip_address: Optional[str] = None
    device_type: Optional[str] = None
    network_type: Optional[str] = None
    time_of_day: Optional[str] = None
    risk_score: float = 0.0
    session_attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessPolicy:
    """Access control policy"""
    id: str
    name: str
    description: str
    model: AccessControlModel
    effect: AccessDecision
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    obligations: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AccessRequest:
    """Access request"""
    subject: AccessSubject
    resource: AccessResource
    operation: AccessOperation
    context: AccessContext
    requested_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AccessResponse:
    """Access response"""
    decision: AccessDecision
    policies_applied: List[str] = field(default_factory=list)
    obligations: List[Dict[str, Any]] = field(default_factory=list)
    audit_data: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0

class AttributeBasedAccess:
    """Attribute-Based Access Control (ABAC)"""

    def __init__(self):
        self.policies: Dict[str, AccessPolicy] = {}
        self.attributes: Dict[str, Dict[str, Any]] = {}
        self.decision_cache: Dict[str, Tuple[AccessResponse, datetime]] = {}
        self.cache_timeout = timedelta(minutes=5)

    def add_policy(self, policy: AccessPolicy):
        """Add ABAC policy"""
        self.policies[policy.id] = policy
        logger.info(f"Added ABAC policy: {policy.name} ({policy.id})")

    def set_attribute(self, entity_id: str, attribute_name: str, value: Any):
        """Set entity attribute"""
        if entity_id not in self.attributes:
            self.attributes[entity_id] = {}
        self.attributes[entity_id][attribute_name] = value

    def evaluate_access(self, request: AccessRequest) -> AccessResponse:
        """Evaluate access using ABAC"""

        # Check cache
        cache_key = self._generate_cache_key(request)
        if cache_key in self.decision_cache:
            cached_response, cache_time = self.decision_cache[cache_key]
            if datetime.utcnow() - cache_time < self.cache_timeout:
                return cached_response

        start_time = datetime.utcnow()

        # Evaluate policies in priority order
        applicable_policies = []
        final_decision = AccessDecision.INDETERMINATE

        sorted_policies = sorted(self.policies.values(), key=lambda p: p.priority, reverse=True)

        for policy in sorted_policies:
            if not policy.enabled or policy.model != AccessControlModel.ABAC:
                continue

            if self._policy_matches(policy, request):
                applicable_policies.append(policy.id)

                # First-match-wins for ABAC
                if policy.effect == AccessDecision.ALLOW:
                    final_decision = AccessDecision.ALLOW
                    break
                elif policy.effect == AccessDecision.DENY:
                    final_decision = AccessDecision.DENY
                    break

        # If no policies matched, deny by default
        if final_decision == AccessDecision.INDETERMINATE:
            final_decision = AccessDecision.DENY

        # Collect obligations
        obligations = []
        for policy_id in applicable_policies:
            policy = self.policies[policy_id]
            obligations.extend(policy.obligations)

        response_time = (datetime.utcnow() - start_time).total_seconds()

        response = AccessResponse(
            decision=final_decision,
            policies_applied=applicable_policies,
            obligations=obligations,
            audit_data={
                'request_id': hash(f"{request.subject.id}{request.resource.id}{request.requested_at}"),
                'evaluation_time': response_time
            },
            response_time=response_time
        )

        # Cache response
        self.decision_cache[cache_key] = (response, datetime.utcnow())

        return response

    def _policy_matches(self, policy: AccessPolicy, request: AccessRequest) -> bool:
        """Check if policy matches the access request"""

        conditions = policy.conditions

        # Subject conditions
        subject_conditions = conditions.get('subject', {})
        if not self._evaluate_conditions(subject_conditions, request.subject, request.context):
            return False

        # Resource conditions
        resource_conditions = conditions.get('resource', {})
        if not self._evaluate_conditions(resource_conditions, request.resource, request.context):
            return False

        # Action conditions
        action_conditions = conditions.get('action', {})
        if action_conditions and request.operation.value not in action_conditions.get('allowed', []):
            return False

        # Environment conditions
        env_conditions = conditions.get('environment', {})
        if not self._evaluate_conditions(env_conditions, request.context, request.context):
            return False

        return True

    def _evaluate_conditions(self, conditions: Dict[str, Any], entity: Any, context: AccessContext) -> bool:
        """Evaluate policy conditions"""

        for condition_name, condition_spec in conditions.items():
            # Get attribute value
            attribute_value = self._get_attribute_value(entity, condition_name, context)

            if attribute_value is None:
                return False

            # Evaluate condition
            if not self._evaluate_condition(attribute_value, condition_spec):
                return False

        return True

    def _get_attribute_value(self, entity: Any, attribute_name: str, context: AccessContext) -> Any:
        """Get attribute value from entity or context"""

        # Check entity attributes
        if hasattr(entity, 'attributes') and attribute_name in entity.attributes:
            return entity.attributes[attribute_name]

        # Check entity direct attributes
        if hasattr(entity, attribute_name):
            return getattr(entity, attribute_name)

        # Check context
        if hasattr(context, attribute_name):
            return getattr(context, attribute_name)

        # Check global attributes
        if hasattr(entity, 'id') and entity.id in self.attributes:
            entity_attrs = self.attributes[entity.id]
            if attribute_name in entity_attrs:
                return entity_attrs[attribute_name]

        return None

    def _evaluate_condition(self, value: Any, condition_spec: Dict[str, Any]) -> bool:
        """Evaluate individual condition"""

        condition_type = condition_spec.get('type', 'equals')

        if condition_type == 'equals':
            return value == condition_spec.get('value')
        elif condition_type == 'not_equals':
            return value != condition_spec.get('value')
        elif condition_type == 'in':
            return value in condition_spec.get('values', [])
        elif condition_type == 'not_in':
            return value not in condition_spec.get('values', [])
        elif condition_type == 'regex':
            pattern = condition_spec.get('pattern', '')
            return bool(re.match(pattern, str(value)))
        elif condition_type == 'range':
            min_val = condition_spec.get('min')
            max_val = condition_spec.get('max')
            return min_val <= value <= max_val
        elif condition_type == 'greater_than':
            return value > condition_spec.get('value')
        elif condition_type == 'less_than':
            return value < condition_spec.get('value')

        return False

    def _generate_cache_key(self, request: AccessRequest) -> str:
        """Generate cache key for access request"""
        key_data = f"{request.subject.id}:{request.resource.id}:{request.operation.value}"
        return hash(key_data)

class PolicyBasedAccess:
    """Policy-Based Access Control (PBAC)"""

    def __init__(self):
        self.policies: Dict[str, AccessPolicy] = {}
        self.policy_engine = None  # Would integrate with policy engine like OPA

    def add_policy(self, policy: AccessPolicy):
        """Add PBAC policy"""
        self.policies[policy.id] = policy
        logger.info(f"Added PBAC policy: {policy.name} ({policy.id})")

    def evaluate_access(self, request: AccessRequest) -> AccessResponse:
        """Evaluate access using PBAC"""

        # In a full implementation, this would use a policy engine
        # For now, delegate to ABAC-style evaluation

        abac = AttributeBasedAccess()
        for policy in self.policies.values():
            abac.add_policy(policy)

        return abac.evaluate_access(request)

    def validate_policy(self, policy: AccessPolicy) -> List[str]:
        """Validate policy syntax and semantics"""
        errors = []

        # Check required fields
        if not policy.id:
            errors.append("Policy ID is required")

        if not policy.conditions:
            errors.append("Policy conditions are required")

        # Validate condition structure
        if not isinstance(policy.conditions, dict):
            errors.append("Policy conditions must be a dictionary")
        else:
            # Validate subject conditions
            subject_conditions = policy.conditions.get('subject', {})
            if subject_conditions:
                errors.extend(self._validate_conditions(subject_conditions, "subject"))

            # Validate resource conditions
            resource_conditions = policy.conditions.get('resource', {})
            if resource_conditions:
                errors.extend(self._validate_conditions(resource_conditions, "resource"))

            # Validate environment conditions
            env_conditions = policy.conditions.get('environment', {})
            if env_conditions:
                errors.extend(self._validate_conditions(env_conditions, "environment"))

        return errors

    def _validate_conditions(self, conditions: Dict[str, Any], context: str) -> List[str]:
        """Validate condition structure"""
        errors = []

        for condition_name, condition_spec in conditions.items():
            if not isinstance(condition_spec, dict):
                errors.append(f"{context} condition '{condition_name}' must be a dictionary")
                continue

            if 'type' not in condition_spec:
                errors.append(f"{context} condition '{condition_name}' missing 'type' field")

            condition_type = condition_spec.get('type')
            valid_types = ['equals', 'not_equals', 'in', 'not_in', 'regex', 'range',
                         'greater_than', 'less_than']

            if condition_type not in valid_types:
                errors.append(f"{context} condition '{condition_name}' has invalid type '{condition_type}'")

            # Validate type-specific requirements
            if condition_type in ['equals', 'not_equals', 'greater_than', 'less_than']:
                if 'value' not in condition_spec:
                    errors.append(f"{context} condition '{condition_name}' missing 'value' field")

            elif condition_type == 'in':
                if 'values' not in condition_spec:
                    errors.append(f"{context} condition '{condition_name}' missing 'values' field")
                elif not isinstance(condition_spec['values'], list):
                    errors.append(f"{context} condition '{condition_name}' 'values' must be a list")

            elif condition_type == 'regex':
                if 'pattern' not in condition_spec:
                    errors.append(f"{context} condition '{condition_name}' missing 'pattern' field")

            elif condition_type == 'range':
                if 'min' not in condition_spec or 'max' not in condition_spec:
                    errors.append(f"{context} condition '{condition_name}' missing 'min' or 'max' field")

        return errors

class ContextAwareAccess:
    """Context-Aware Access Control"""

    def __init__(self):
        self.context_providers: Dict[str, Callable] = {}
        self.risk_thresholds: Dict[str, float] = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.adaptive_policies: Dict[str, Dict[str, Any]] = {}

    def add_context_provider(self, context_type: str, provider: Callable):
        """Add context provider"""
        self.context_providers[context_type] = provider

    def evaluate_context_risk(self, context: AccessContext) -> Tuple[str, float]:
        """Evaluate context-based risk"""

        risk_score = 0.0
        risk_factors = []

        # Time-based risk
        if context.time_of_day:
            hour = int(context.time_of_day.split(':')[0])
            if hour < 6 or hour > 22:
                risk_score += 0.2
                risk_factors.append('unusual_time')

        # Location-based risk
        if context.location:
            # Check against known safe locations
            # In production, compare with user's historical locations
            risk_score += 0.1
            risk_factors.append('unknown_location')

        # Device-based risk
        if context.device_type:
            risky_devices = ['unknown', 'untrusted']
            if context.device_type in risky_devices:
                risk_score += 0.3
                risk_factors.append('untrusted_device')

        # Network-based risk
        if context.network_type:
            risky_networks = ['unknown', 'public_wifi']
            if context.network_type in risky_networks:
                risk_score += 0.4
                risk_factors.append('risky_network')

        # Session-based risk
        if context.session_attributes.get('consecutive_failures', 0) > 3:
            risk_score += 0.5
            risk_factors.append('suspicious_activity')

        # Determine risk level
        if risk_score >= self.risk_thresholds['high']:
            risk_level = 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return risk_level, risk_score

    def apply_adaptive_controls(self, user_id: str, risk_level: str,
                              base_permissions: List[str]) -> List[str]:
        """Apply adaptive access controls based on risk"""

        if risk_level == 'low':
            return base_permissions
        elif risk_level == 'medium':
            # Reduce permissions for medium risk
            restricted_permissions = []
            for perm in base_permissions:
                if not perm.endswith(':admin') and not perm.endswith(':delete'):
                    restricted_permissions.append(perm)
            return restricted_permissions
        else:  # high risk
            # Minimal permissions for high risk
            return [perm for perm in base_permissions if perm.endswith(':read')]

    def get_context_attributes(self, user_id: str) -> Dict[str, Any]:
        """Get context attributes for user"""
        attributes = {}

        for context_type, provider in self.context_providers.items():
            try:
                attributes[context_type] = provider(user_id)
            except Exception as e:
                logger.warning(f"Context provider {context_type} failed: {str(e)}")

        return attributes

    def enforce_step_up_auth(self, user_id: str, risk_level: str) -> bool:
        """Enforce step-up authentication for high-risk actions"""
        if risk_level in ['high', 'medium']:
            # Require additional authentication factors
            # In production, trigger MFA challenge
            return True

        return False

class AccessControlManager:
    """Unified Access Control Manager"""

    def __init__(self, model: AccessControlModel = AccessControlModel.ABAC):
        self.model = model
        self.abac = AttributeBasedAccess()
        self.pbac = PolicyBasedAccess()
        self.context_aware = ContextAwareAccess()
        self.audit_log: List[Dict[str, Any]] = []

    def add_policy(self, policy: AccessPolicy):
        """Add access control policy"""
        if policy.model == AccessControlModel.ABAC:
            self.abac.add_policy(policy)
        elif policy.model == AccessControlModel.PBAC:
            self.pbac.add_policy(policy)

    def check_access(self, subject: AccessSubject, resource: AccessResource,
                    operation: AccessOperation, context: Optional[AccessContext] = None) -> AccessResponse:
        """Check access for subject on resource"""

        request = AccessRequest(
            subject=subject,
            resource=resource,
            operation=operation,
            context=context or AccessContext()
        )

        # Evaluate context risk
        risk_level, risk_score = self.context_aware.evaluate_context_risk(request.context)
        request.context.risk_score = risk_score

        # Choose access control model
        if self.model == AccessControlModel.ABAC:
            response = self.abac.evaluate_access(request)
        elif self.model == AccessControlModel.PBAC:
            response = self.pbac.evaluate_access(request)
        else:
            response = AccessResponse(decision=AccessDecision.DENY)

        # Apply context-aware controls
        if response.decision == AccessDecision.ALLOW:
            base_permissions = self._get_subject_permissions(subject)
            adapted_permissions = self.context_aware.apply_adaptive_controls(
                subject.id, risk_level, base_permissions
            )

            # Check if operation is still allowed after adaptation
            operation_allowed = any(
                operation.value in perm or perm.endswith(':*') or perm.endswith(f':{operation.value}')
                for perm in adapted_permissions
            )

            if not operation_allowed:
                response.decision = AccessDecision.DENY
                response.audit_data['adaptation_reason'] = 'risk_based_restriction'

        # Log access decision
        self._audit_access_decision(request, response, risk_level)

        return response

    def _get_subject_permissions(self, subject: AccessSubject) -> List[str]:
        """Get permissions for subject"""
        permissions = list(subject.permissions)

        # Add role-based permissions
        for role in subject.roles:
            role_perms = self._get_role_permissions(role)
            permissions.extend(role_perms)

        return list(set(permissions))  # Remove duplicates

    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for role"""
        # In production, this would query a role database
        role_permissions = {
            'admin': ['*:*'],
            'data_scientist': ['model:*', 'data:*', 'training:*'],
            'user': ['inference:read', 'model:read'],
            'auditor': ['audit:read', 'logs:read']
        }

        return role_permissions.get(role, [])

    def _audit_access_decision(self, request: AccessRequest, response: AccessResponse, risk_level: str):
        """Audit access decision"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'subject_id': request.subject.id,
            'subject_type': request.subject.type,
            'resource_id': request.resource.id,
            'resource_type': request.resource.type,
            'operation': request.operation.value,
            'decision': response.decision.value,
            'risk_level': risk_level,
            'risk_score': request.context.risk_score,
            'policies_applied': response.policies_applied,
            'response_time': response.response_time,
            'context': {
                'ip_address': request.context.ip_address,
                'location': request.context.location,
                'device_type': request.context.device_type
            }
        }

        self.audit_log.append(audit_entry)

        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]

    def get_access_audit(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get access audit log"""
        if not filters:
            return self.audit_log[-100:]  # Last 100 entries

        filtered = []
        for entry in self.audit_log:
            match = True
            for key, value in filters.items():
                if entry.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(entry)

        return filtered[-100:]  # Return last 100 matching entries

    def set_risk_thresholds(self, thresholds: Dict[str, float]):
        """Set risk evaluation thresholds"""
        self.context_aware.risk_thresholds.update(thresholds)

    def add_context_provider(self, context_type: str, provider: Callable):
        """Add context provider"""
        self.context_aware.add_context_provider(context_type, provider)

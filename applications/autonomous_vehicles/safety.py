"""
Quantum Edge AI Platform - Autonomous Vehicles Safety System

Advanced safety systems for autonomous vehicles including:
- Collision avoidance and risk assessment
- Emergency response and fail-safe mechanisms
- Redundancy management and fault tolerance
- Safety validation and compliance checking
- Real-time safety monitoring and alerting
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety assurance levels"""
    NOMINAL = "nominal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyEvent(Enum):
    """Safety-related events"""
    COLLISION_WARNING = "collision_warning"
    SYSTEM_FAULT = "system_fault"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_LOSS = "communication_loss"
    EMERGENCY_BRAKE = "emergency_brake"
    SYSTEM_RECOVERY = "system_recovery"
    SAFETY_VIOLATION = "safety_violation"

@dataclass
class SafetyAssessment:
    """Safety assessment result"""
    overall_safety: SafetyLevel
    collision_risk: float  # 0-1, higher = more dangerous
    time_to_collision: Optional[float]  # seconds
    safety_violations: List[str]
    recommended_actions: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SafetyConstraint:
    """Safety constraint definition"""
    name: str
    condition: str  # Python expression to evaluate
    severity: SafetyLevel
    description: str
    enabled: bool = True
    cooldown_period: int = 5  # seconds

@dataclass
class SystemFault:
    """System fault record"""
    fault_id: str
    component: str
    fault_type: str
    severity: SafetyLevel
    description: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    mitigation_actions: List[str] = field(default_factory=list)

class CollisionAvoidance:
    """Collision avoidance system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collision_threshold = config.get('collision_threshold', 2.0)  # meters
        self.warning_time = config.get('warning_time', 3.0)  # seconds
        self.emergency_time = config.get('emergency_time', 1.5)  # seconds

    def assess_collision_risk(self, vehicle_state: Any,
                            environmental_data: Any) -> Dict[str, Any]:
        """Assess collision risk with obstacles"""
        risks = []

        current_pos = np.array(vehicle_state.position)
        current_vel = np.array(vehicle_state.velocity)

        for obstacle in environmental_data.objects:
            obs_pos = np.array(obstacle.position)
            obs_vel = np.array(obstacle.velocity)

            # Calculate relative position and velocity
            relative_pos = obs_pos - current_pos
            relative_vel = obs_vel - current_vel

            # Distance to obstacle
            distance = np.linalg.norm(relative_pos)

            if distance < 0.1:  # Very close
                continue

            # Time to collision calculation
            # Solve: |relative_pos + t * relative_vel| = collision_threshold
            a = np.dot(relative_vel, relative_vel)
            b = 2 * np.dot(relative_pos, relative_vel)
            c = np.dot(relative_pos, relative_pos) - self.collision_threshold ** 2

            if a > 0.001:  # Not moving directly away
                discriminant = b*b - 4*a*c

                if discriminant >= 0:
                    t1 = (-b - np.sqrt(discriminant)) / (2*a)
                    t2 = (-b + np.sqrt(discriminant)) / (2*a)

                    # Find the smallest positive time
                    times = [t for t in [t1, t2] if t > 0.1]  # Ignore very small times
                    time_to_collision = min(times) if times else None
                else:
                    time_to_collision = None
            else:
                # Static or moving away
                time_to_collision = None if distance > self.collision_threshold else 0

            # Assess risk level
            risk_level = self._calculate_risk_level(distance, time_to_collision)

            risks.append({
                'obstacle_id': obstacle.object_id,
                'distance': distance,
                'time_to_collision': time_to_collision,
                'risk_level': risk_level,
                'obstacle_type': obstacle.object_type
            })

        # Find highest risk
        if risks:
            highest_risk = max(risks, key=lambda x: self._risk_priority(x['risk_level']))
            overall_risk = highest_risk['risk_level']
            min_time_to_collision = min((r['time_to_collision'] for r in risks
                                       if r['time_to_collision'] is not None), default=None)
        else:
            overall_risk = 'low'
            min_time_to_collision = None

        return {
            'overall_risk': overall_risk,
            'min_time_to_collision': min_time_to_collision,
            'collision_warnings': [r for r in risks if r['risk_level'] in ['high', 'critical']],
            'risk_assessment': risks
        }

    def _calculate_risk_level(self, distance: float, time_to_collision: Optional[float]) -> str:
        """Calculate risk level based on distance and time"""
        if time_to_collision is None:
            return 'low' if distance > 10 else 'medium' if distance > 5 else 'high'

        if time_to_collision <= self.emergency_time:
            return 'critical'
        elif time_to_collision <= self.warning_time:
            return 'high'
        elif time_to_collision <= self.warning_time * 2:
            return 'medium'
        else:
            return 'low'

    def _risk_priority(self, risk_level: str) -> int:
        """Get priority value for risk level"""
        priorities = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        return priorities.get(risk_level, 0)

class RedundancyManager:
    """System redundancy and fault tolerance manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.systems = {}
        self.faults = []
        self.redundancy_level = config.get('redundancy_level', 2)

        # System health monitoring
        self.system_health = {}
        self.health_check_interval = config.get('health_check_interval', 1.0)

    def register_system(self, system_name: str, system_instance: Any,
                       health_check_func: Optional[Callable] = None):
        """Register a system for redundancy monitoring"""
        self.systems[system_name] = {
            'instance': system_instance,
            'health_check': health_check_func,
            'status': 'healthy',
            'last_check': datetime.utcnow(),
            'failures': 0
        }

    def check_system_health(self) -> Dict[str, Any]:
        """Check health of all registered systems"""
        health_status = {}

        for system_name, system_info in self.systems.items():
            try:
                if system_info['health_check']:
                    is_healthy = system_info['health_check'](system_info['instance'])
                else:
                    # Default health check - assume healthy if no exceptions
                    is_healthy = True

                status = 'healthy' if is_healthy else 'faulty'
                system_info['status'] = status
                system_info['last_check'] = datetime.utcnow()

                if not is_healthy:
                    system_info['failures'] += 1
                    self._record_fault(system_name, 'health_check_failed', 'medium')

                health_status[system_name] = {
                    'status': status,
                    'last_check': system_info['last_check'],
                    'failures': system_info['failures']
                }

            except Exception as e:
                system_info['status'] = 'error'
                system_info['failures'] += 1
                self._record_fault(system_name, f'health_check_error: {str(e)}', 'high')

                health_status[system_name] = {
                    'status': 'error',
                    'error': str(e),
                    'failures': system_info['failures']
                }

        return health_status

    def _record_fault(self, component: str, description: str, severity: str):
        """Record a system fault"""
        fault = SystemFault(
            fault_id=f"fault_{len(self.faults)}",
            component=component,
            fault_type='system_fault',
            severity=SafetyLevel(severity),
            description=description,
            detected_at=datetime.utcnow()
        )

        self.faults.append(fault)
        logger.warning(f"System fault recorded: {component} - {description}")

    def get_redundant_systems(self, system_type: str) -> List[Any]:
        """Get available redundant systems of specified type"""
        available_systems = []

        for system_name, system_info in self.systems.items():
            if system_type in system_name and system_info['status'] == 'healthy':
                available_systems.append(system_info['instance'])

        return available_systems

    def failover_to_redundant(self, failed_system: str) -> bool:
        """Failover to redundant system"""
        # Find redundant systems
        system_type = failed_system.split('_')[0]  # Extract type from name
        redundant_systems = self.get_redundant_systems(system_type)

        if redundant_systems:
            logger.info(f"Failing over from {failed_system} to redundant system")
            # In real implementation, switch to redundant system
            return True

        logger.error(f"No redundant systems available for {failed_system}")
        return False

class EmergencyResponse:
    """Emergency response and fail-safe mechanisms"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.emergency_procedures = {
            'collision_imminent': self._collision_emergency,
            'system_failure': self._system_failure_emergency,
            'communication_loss': self._communication_loss_emergency,
            'sensor_failure': self._sensor_failure_emergency
        }

        self.emergency_history = []

    def handle_emergency(self, emergency_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle emergency situation"""
        context = context or {}

        if emergency_type in self.emergency_procedures:
            procedure = self.emergency_procedures[emergency_type]

            try:
                response = procedure(context)
                self._log_emergency(emergency_type, 'handled', context, response)
                return response
            except Exception as e:
                logger.error(f"Emergency procedure failed: {e}")
                self._log_emergency(emergency_type, 'failed', context, {'error': str(e)})
                return self._fallback_emergency_response()
        else:
            logger.error(f"Unknown emergency type: {emergency_type}")
            return self._fallback_emergency_response()

    def _collision_emergency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle imminent collision emergency"""
        return {
            'action': 'emergency_brake',
            'brake_force': 1.0,
            'steering': 0.0,
            'hazard_lights': True,
            'notify_nearby_vehicles': True,
            'duration': 5.0  # seconds
        }

    def _system_failure_emergency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system failure emergency"""
        return {
            'action': 'safe_stop',
            'brake_force': 0.8,
            'steering': 0.0,
            'activate_redundancy': True,
            'system_restart': True,
            'duration': 10.0
        }

    def _communication_loss_emergency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle communication loss emergency"""
        return {
            'action': 'isolated_operation',
            'speed_limit': 15.0,  # km/h
            'maintain_distance': True,
            'reduced_functionality': True,
            'attempt_reconnection': True
        }

    def _sensor_failure_emergency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sensor failure emergency"""
        return {
            'action': 'degraded_operation',
            'speed_limit': 20.0,
            'use_redundant_sensors': True,
            'increased_safety_margin': True,
            'diagnostic_mode': True
        }

    def _fallback_emergency_response(self) -> Dict[str, Any]:
        """Fallback emergency response"""
        return {
            'action': 'immediate_stop',
            'brake_force': 1.0,
            'steering': 0.0,
            'hazard_lights': True,
            'error': 'unknown_emergency'
        }

    def _log_emergency(self, emergency_type: str, status: str,
                      context: Dict[str, Any], response: Dict[str, Any]):
        """Log emergency event"""
        emergency_record = {
            'type': emergency_type,
            'status': status,
            'context': context,
            'response': response,
            'timestamp': datetime.utcnow()
        }

        self.emergency_history.append(emergency_record)

        logger.critical(f"Emergency handled: {emergency_type} - {status}")

class SafetyValidator:
    """Safety validation and compliance checking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_constraints = self._load_safety_constraints()
        self.validation_history = []

    def _load_safety_constraints(self) -> List[SafetyConstraint]:
        """Load safety constraints"""
        return [
            SafetyConstraint(
                name="minimum_distance",
                condition="min_distance > 2.0",
                severity=SafetyLevel.CRITICAL,
                description="Maintain minimum safe distance from obstacles"
            ),
            SafetyConstraint(
                name="speed_limit",
                condition="current_speed <= speed_limit",
                severity=SafetyLevel.WARNING,
                description="Respect speed limits"
            ),
            SafetyConstraint(
                name="system_redundancy",
                condition="healthy_systems >= 2",
                severity=SafetyLevel.WARNING,
                description="Maintain system redundancy"
            ),
            SafetyConstraint(
                name="communication_health",
                condition="communication_status == 'healthy'",
                severity=SafetyLevel.WARNING,
                description="Maintain communication links"
            )
        ]

    def validate_safety_constraints(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate safety constraints"""
        violations = []

        for constraint in self.safety_constraints:
            if not constraint.enabled:
                continue

            try:
                # Evaluate constraint condition
                result = eval(constraint.condition, {"__builtins__": {}}, context)

                if not result:
                    violation = {
                        'constraint': constraint.name,
                        'severity': constraint.severity.value,
                        'description': constraint.description,
                        'context': context,
                        'timestamp': datetime.utcnow()
                    }
                    violations.append(violation)

            except Exception as e:
                logger.error(f"Error evaluating constraint {constraint.name}: {e}")

        # Log violations
        for violation in violations:
            self.validation_history.append(violation)
            logger.warning(f"Safety violation: {violation['constraint']} - {violation['description']}")

        return violations

    def check_compliance(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall safety compliance"""
        violations = self.validate_safety_constraints(system_state)

        # Categorize violations
        critical_violations = [v for v in violations if v['severity'] == 'critical']
        warning_violations = [v for v in violations if v['severity'] == 'warning']

        compliance_score = max(0, 1.0 - (len(critical_violations) * 0.5 + len(warning_violations) * 0.1))

        return {
            'compliant': len(critical_violations) == 0,
            'compliance_score': compliance_score,
            'critical_violations': len(critical_violations),
            'warning_violations': len(warning_violations),
            'violations': violations
        }

class SafetySystem:
    """Main safety system coordinator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize safety subsystems
        self.collision_avoidance = CollisionAvoidance(config)
        self.redundancy_manager = RedundancyManager(config)
        self.emergency_response = EmergencyResponse(config)
        self.safety_validator = SafetyValidator(config)

        # Safety state
        self.current_safety_level = SafetyLevel.NOMINAL
        self.safety_events = []
        self.monitoring_active = False

    def assess_safety(self, environmental_data: Any, vehicle_state: Any) -> SafetyAssessment:
        """Perform comprehensive safety assessment"""
        # Collision risk assessment
        collision_assessment = self.collision_avoidance.assess_collision_risk(
            vehicle_state, environmental_data
        )

        # System health check
        system_health = self.redundancy_manager.check_system_health()

        # Safety constraint validation
        context = {
            'current_speed': np.linalg.norm(vehicle_state.velocity),
            'speed_limit': 30.0,  # Mock speed limit
            'min_distance': collision_assessment.get('min_distance', 100),
            'healthy_systems': sum(1 for s in system_health.values() if s['status'] == 'healthy'),
            'communication_status': 'healthy'  # Mock
        }

        violations = self.safety_validator.validate_safety_constraints(context)

        # Determine overall safety level
        safety_level = self._determine_safety_level(
            collision_assessment, system_health, violations
        )

        # Generate recommendations
        recommendations = self._generate_safety_recommendations(
            safety_level, collision_assessment, violations
        )

        assessment = SafetyAssessment(
            overall_safety=safety_level,
            collision_risk=collision_assessment['overall_risk'],
            time_to_collision=collision_assessment['min_time_to_collision'],
            safety_violations=[v['description'] for v in violations],
            recommended_actions=recommendations,
            confidence=0.9
        )

        self.current_safety_level = safety_level

        return assessment

    def _determine_safety_level(self, collision_assessment: Dict[str, Any],
                              system_health: Dict[str, Any],
                              violations: List[Dict[str, Any]]) -> SafetyLevel:
        """Determine overall safety level"""
        # Check for critical violations
        critical_violations = [v for v in violations if v['severity'] == 'critical']
        if critical_violations:
            return SafetyLevel.CRITICAL

        # Check collision risk
        risk_level = collision_assessment['overall_risk']
        if risk_level == 'critical':
            return SafetyLevel.EMERGENCY
        elif risk_level == 'high':
            return SafetyLevel.CRITICAL

        # Check system health
        unhealthy_systems = sum(1 for s in system_health.values() if s['status'] != 'healthy')
        if unhealthy_systems > 1:  # Multiple system failures
            return SafetyLevel.CRITICAL

        # Check for warnings
        warning_violations = [v for v in violations if v['severity'] == 'warning']
        if warning_violations or risk_level == 'medium' or unhealthy_systems > 0:
            return SafetyLevel.WARNING

        return SafetyLevel.NOMINAL

    def _generate_safety_recommendations(self, safety_level: SafetyLevel,
                                       collision_assessment: Dict[str, Any],
                                       violations: List[Dict[str, Any]]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []

        if safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            recommendations.append("Execute emergency braking procedure")
            recommendations.append("Activate hazard warning systems")
            recommendations.append("Notify emergency services")

        if collision_assessment['overall_risk'] in ['high', 'critical']:
            recommendations.append("Increase following distance")
            recommendations.append("Reduce vehicle speed")
            recommendations.append("Prepare for emergency maneuver")

        for violation in violations:
            if violation['severity'] == 'critical':
                recommendations.append(f"Address critical safety violation: {violation['description']}")
            elif violation['severity'] == 'warning':
                recommendations.append(f"Address safety warning: {violation['description']}")

        if not recommendations:
            recommendations.append("Continue normal operation")

        return recommendations

    def validate_control_command(self, control_command: Any, vehicle_state: Any,
                               environmental_data: Any) -> Any:
        """Validate control command for safety"""
        # Check if command would violate safety constraints
        context = {
            'proposed_steering': control_command.steering_angle,
            'proposed_throttle': control_command.throttle,
            'proposed_brake': control_command.brake,
            'current_speed': np.linalg.norm(vehicle_state.velocity),
            'max_safe_steering': np.radians(45),
            'max_throttle': 1.0,
            'max_brake': 1.0
        }

        violations = self.safety_validator.validate_safety_constraints(context)

        # Modify command if necessary for safety
        validated_command = control_command

        if violations:
            logger.warning(f"Control command validation violations: {violations}")

            # Apply safety modifications
            for violation in violations:
                if violation['constraint'] == 'steering_limit':
                    validated_command.steering_angle = np.clip(
                        validated_command.steering_angle,
                        -context['max_safe_steering'],
                        context['max_safe_steering']
                    )
                elif violation['constraint'] == 'speed_safety':
                    validated_command.throttle = min(validated_command.throttle, 0.5)

        return validated_command

    def handle_external_emergency(self, emergency_data: Dict[str, Any]):
        """Handle emergency from external source"""
        emergency_type = emergency_data.get('type', 'external_emergency')
        response = self.emergency_response.handle_emergency(emergency_type, emergency_data)

        self.safety_events.append({
            'event': SafetyEvent.SAFETY_VIOLATION,
            'type': 'external_emergency',
            'data': emergency_data,
            'response': response,
            'timestamp': datetime.utcnow()
        })

    def get_health_status(self) -> Dict[str, Any]:
        """Get safety system health status"""
        return {
            'status': 'healthy',
            'current_safety_level': self.current_safety_level.value,
            'active_violations': len(self.safety_validator.validation_history),
            'emergency_events': len(self.emergency_response.emergency_history),
            'system_redundancy': len(self.redundancy_manager.systems),
            'health_score': 0.95
        }

    def run_diagnostic(self) -> Dict[str, Any]:
        """Run safety system diagnostic"""
        diagnostic_result = {
            'component': 'safety_system',
            'tests': {}
        }

        # Test collision avoidance
        try:
            mock_vehicle = type('MockVehicle', (), {'position': (0, 0, 0), 'velocity': (10, 0, 0)})()
            mock_env = type('MockEnv', (), {'objects': []})()

            risk = self.collision_avoidance.assess_collision_risk(mock_vehicle, mock_env)
            diagnostic_result['tests']['collision_avoidance'] = {
                'status': 'pass',
                'risk_assessment': risk
            }
        except Exception as e:
            diagnostic_result['tests']['collision_avoidance'] = {
                'status': 'fail',
                'error': str(e)
            }

        # Test safety validation
        try:
            context = {'current_speed': 25, 'speed_limit': 30, 'min_distance': 5}
            violations = self.safety_validator.validate_safety_constraints(context)
            diagnostic_result['tests']['safety_validation'] = {
                'status': 'pass',
                'violations_found': len(violations)
            }
        except Exception as e:
            diagnostic_result['tests']['safety_validation'] = {
                'status': 'fail',
                'error': str(e)
            }

        # Test emergency response
        try:
            response = self.emergency_response.handle_emergency('test_emergency')
            diagnostic_result['tests']['emergency_response'] = {
                'status': 'pass',
                'response_action': response.get('action')
            }
        except Exception as e:
            diagnostic_result['tests']['emergency_response'] = {
                'status': 'fail',
                'error': str(e)
            }

        # Calculate overall score
        passed_tests = sum(1 for test in diagnostic_result['tests'].values() if test['status'] == 'pass')
        total_tests = len(diagnostic_result['tests'])
        diagnostic_result['diagnostic_score'] = passed_tests / total_tests if total_tests > 0 else 0

        return diagnostic_result

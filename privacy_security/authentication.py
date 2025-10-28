"""
Quantum Edge AI Platform - Authentication Module

Advanced authentication frameworks including multi-factor authentication,
biometric authentication, and zero-trust security models.
"""

import os
import json
import hashlib
import hmac
import base64
import time
import secrets
import re
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading

# Third-party imports (would be installed in production)
try:
    import jwt
    import bcrypt
    import pyotp
    import qrcode
    import cv2
    import face_recognition
    import speech_recognition as sr
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.x509 import Certificate
except ImportError:
    # Fallback for development without dependencies
    jwt = bcrypt = pyotp = qrcode = cv2 = face_recognition = sr = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    HARDWARE_TOKEN = "hardware_token"
    SOCIAL = "social"

class AuthFactor(Enum):
    """Authentication factors"""
    KNOWLEDGE = "knowledge"  # Something you know
    POSSESSION = "possession"  # Something you have
    INHERENCE = "inherence"  # Something you are

class BiometricType(Enum):
    """Biometric authentication types"""
    FACE = "face"
    FINGERPRINT = "fingerprint"
    VOICE = "voice"
    IRIS = "iris"
    BEHAVIORAL = "behavioral"

@dataclass
class UserCredentials:
    """User authentication credentials"""
    user_id: str
    username: str
    password_hash: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    auth_methods: List[AuthMethod] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_methods: List[AuthMethod] = field(default_factory=list)
    biometric_data: Dict[str, Any] = field(default_factory=dict)
    certificate_thumbprint: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuthSession:
    """Authentication session"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    auth_factors: List[AuthFactor] = field(default_factory=list)
    device_fingerprint: Optional[str] = None
    risk_score: float = 0.0
    active: bool = True

@dataclass
class AuthChallenge:
    """Authentication challenge"""
    challenge_id: str
    user_id: str
    method: AuthMethod
    challenge_data: Dict[str, Any]
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3
    completed: bool = False

class MultiFactorAuth:
    """Multi-factor authentication system"""

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.users: Dict[str, UserCredentials] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.challenges: Dict[str, AuthChallenge] = {}
        self.totp_secrets: Dict[str, str] = {}

        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=8)

        # Risk assessment
        self.risk_threshold = 0.7

    def register_user(self, username: str, password: str, email: Optional[str] = None,
                     phone: Optional[str] = None) -> str:
        """Register new user"""
        user_id = secrets.token_hex(16)

        # Hash password
        if bcrypt:
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            # Simple hash for development
            password_hash = hashlib.sha256(password.encode()).hexdigest()

        user = UserCredentials(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            email=email,
            phone=phone,
            auth_methods=[AuthMethod.PASSWORD]
        )

        self.users[user_id] = user
        logger.info(f"Registered user: {username} ({user_id})")
        return user_id

    def authenticate(self, username: str, password: str, factors: List[AuthFactor] = None,
                    device_info: Dict[str, Any] = None) -> Optional[AuthSession]:
        """Authenticate user with multiple factors"""

        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            logger.warning(f"Authentication failed: user {username} not found")
            return None

        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            logger.warning(f"Authentication failed: account {username} is locked")
            return None

        # Primary authentication (password)
        if not self._verify_password(password, user.password_hash):
            user.failed_attempts += 1

            if user.failed_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                logger.warning(f"Account {username} locked due to failed attempts")

            logger.warning(f"Authentication failed: invalid password for {username}")
            return None

        # Reset failed attempts on successful primary auth
        user.failed_attempts = 0

        # Multi-factor authentication
        if user.mfa_enabled and factors:
            if not self._verify_additional_factors(user, factors):
                logger.warning(f"MFA failed for {username}")
                return None

        # Create session
        session = self._create_session(user, device_info)
        user.last_login = datetime.utcnow()

        logger.info(f"Authentication successful for {username}")
        return session

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password"""
        if bcrypt:
            return bcrypt.checkpw(password.encode(), password_hash.encode())
        else:
            # Simple verification for development
            return hashlib.sha256(password.encode()).hexdigest() == password_hash

    def _verify_additional_factors(self, user: UserCredentials, factors: List[AuthFactor]) -> bool:
        """Verify additional authentication factors"""
        for factor in factors:
            if factor == AuthFactor.POSSESSION:
                # Verify TOTP, SMS, etc.
                if AuthMethod.TOTP in user.mfa_methods:
                    # TOTP verification would happen in challenge-response
                    pass
            elif factor == AuthFactor.INHERENCE:
                # Biometric verification
                if AuthMethod.BIOMETRIC in user.mfa_methods:
                    # Biometric verification would happen separately
                    pass

        return True  # Simplified

    def _create_session(self, user: UserCredentials, device_info: Dict[str, Any] = None) -> AuthSession:
        """Create authentication session"""
        session_id = secrets.token_hex(32)

        session = AuthSession(
            session_id=session_id,
            user_id=user.user_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.session_timeout,
            ip_address=device_info.get('ip_address', 'unknown') if device_info else 'unknown',
            user_agent=device_info.get('user_agent', 'unknown') if device_info else 'unknown',
            device_fingerprint=self._generate_device_fingerprint(device_info) if device_info else None,
            auth_factors=[AuthFactor.KNOWLEDGE]  # Password verified
        )

        self.sessions[session_id] = session
        return session

    def _generate_device_fingerprint(self, device_info: Dict[str, Any]) -> str:
        """Generate device fingerprint"""
        fingerprint_data = [
            device_info.get('user_agent', ''),
            device_info.get('ip_address', ''),
            device_info.get('screen_resolution', ''),
            device_info.get('timezone', ''),
            device_info.get('language', '')
        ]

        fingerprint = hashlib.sha256('|'.join(fingerprint_data).encode()).hexdigest()
        return fingerprint

    def verify_session(self, session_id: str) -> Optional[AuthSession]:
        """Verify authentication session"""
        session = self.sessions.get(session_id)

        if not session or not session.active:
            return None

        if datetime.utcnow() > session.expires_at:
            session.active = False
            return None

        return session

    def logout(self, session_id: str):
        """Logout user session"""
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            logger.info(f"Session {session_id} logged out")

    def enable_mfa(self, user_id: str, methods: List[AuthMethod]) -> bool:
        """Enable MFA for user"""
        if user_id not in self.users:
            return False

        user = self.users[user_id]
        user.mfa_enabled = True
        user.mfa_methods = methods

        # Initialize MFA methods
        for method in methods:
            if method == AuthMethod.TOTP:
                self._setup_totp(user_id)
            elif method == AuthMethod.SMS:
                self._setup_sms(user_id)

        logger.info(f"MFA enabled for user {user_id}")
        return True

    def _setup_totp(self, user_id: str) -> str:
        """Setup TOTP for user"""
        if not pyotp:
            logger.warning("pyotp not available, TOTP setup skipped")
            return ""

        secret = pyotp.random_base32()
        self.totp_secrets[user_id] = secret

        # Generate QR code provisioning URI
        uri = pyotp.totp.TOTP(secret).provisioning_uri(name=f"user_{user_id}", issuer_name="Quantum Edge AI")

        logger.info(f"TOTP setup for user {user_id}")
        return uri

    def _setup_sms(self, user_id: str):
        """Setup SMS verification for user"""
        # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
        logger.info(f"SMS setup for user {user_id}")

    def verify_totp(self, user_id: str, code: str) -> bool:
        """Verify TOTP code"""
        if not pyotp or user_id not in self.totp_secrets:
            return False

        totp = pyotp.TOTP(self.totp_secrets[user_id])
        return totp.verify(code)

    def send_sms_challenge(self, user_id: str) -> str:
        """Send SMS challenge"""
        if user_id not in self.users:
            raise ValueError("User not found")

        user = self.users[user_id]
        if not user.phone:
            raise ValueError("User has no phone number")

        # Generate challenge code
        code = secrets.token_hex(3).upper()

        # In production, send SMS via service
        logger.info(f"SMS challenge sent to {user.phone} for user {user_id}: {code}")

        # Store challenge
        challenge = AuthChallenge(
            challenge_id=secrets.token_hex(16),
            user_id=user_id,
            method=AuthMethod.SMS,
            challenge_data={'code': code, 'phone': user.phone},
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )

        self.challenges[challenge.challenge_id] = challenge
        return challenge.challenge_id

    def verify_challenge(self, challenge_id: str, response: str) -> bool:
        """Verify authentication challenge"""
        challenge = self.challenges.get(challenge_id)

        if not challenge or challenge.completed:
            return False

        if datetime.utcnow() > challenge.expires_at:
            return False

        challenge.attempts += 1

        if challenge.attempts > challenge.max_attempts:
            return False

        # Verify response
        if challenge.method == AuthMethod.SMS:
            expected_code = challenge.challenge_data.get('code')
            if response == expected_code:
                challenge.completed = True
                return True

        elif challenge.method == AuthMethod.TOTP:
            return self.verify_totp(challenge.user_id, response)

        return False

    def assess_risk(self, session: AuthSession, context: Dict[str, Any]) -> float:
        """Assess authentication risk"""
        risk_score = 0.0

        # Check IP address change
        if context.get('ip_address') != session.ip_address:
            risk_score += 0.3

        # Check user agent change
        if context.get('user_agent') != session.user_agent:
            risk_score += 0.2

        # Check time-based patterns
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:  # Unusual hours
            risk_score += 0.2

        # Check geographic anomalies
        # In production, compare with user's typical locations

        session.risk_score = min(risk_score, 1.0)
        return session.risk_score

class BiometricAuth:
    """Biometric authentication system"""

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Any] = {}

        # Initialize biometric libraries
        self.face_recognition_available = face_recognition is not None
        self.cv2_available = cv2 is not None
        self.speech_available = sr is not None

    def enroll_face(self, user_id: str, face_image_path: str) -> bool:
        """Enroll face biometric"""
        if not self.face_recognition_available:
            logger.warning("Face recognition not available")
            return False

        try:
            # Load and process face image
            image = face_recognition.load_image_file(face_image_path)
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                logger.error(f"No face detected in image for user {user_id}")
                return False

            # Store face encoding
            self.templates[user_id] = {
                'type': BiometricType.FACE,
                'face_encoding': face_encodings[0].tolist(),
                'enrolled_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Face enrolled for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Face enrollment failed for user {user_id}: {str(e)}")
            return False

    def verify_face(self, user_id: str, face_image_path: str) -> Tuple[bool, float]:
        """Verify face biometric"""
        if not self.face_recognition_available or user_id not in self.templates:
            return False, 0.0

        try:
            # Load stored template
            template = self.templates[user_id]
            stored_encoding = template['face_encoding']

            # Load and process verification image
            image = face_recognition.load_image_file(face_image_path)
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                return False, 0.0

            # Compare faces
            results = face_recognition.compare_faces([stored_encoding], face_encodings[0])
            face_distance = face_recognition.face_distance([stored_encoding], face_encodings[0])[0]

            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0, 1.0 - face_distance)

            return results[0], confidence

        except Exception as e:
            logger.error(f"Face verification failed for user {user_id}: {str(e)}")
            return False, 0.0

    def enroll_voice(self, user_id: str, audio_file_path: str) -> bool:
        """Enroll voice biometric"""
        if not self.speech_available:
            logger.warning("Speech recognition not available")
            return False

        try:
            # Load and process audio file
            recognizer = sr.Recognizer()

            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)

            # Extract voice features (simplified)
            voice_features = {
                'sample_rate': audio.sample_rate,
                'duration': len(audio.frame_data) / audio.sample_rate,
                'energy': sum(abs(sample) for sample in audio.frame_data) / len(audio.frame_data)
            }

            # Store voice template
            self.templates[f"{user_id}_voice"] = {
                'type': BiometricType.VOICE,
                'features': voice_features,
                'enrolled_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Voice enrolled for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Voice enrollment failed for user {user_id}: {str(e)}")
            return False

    def verify_voice(self, user_id: str, audio_file_path: str) -> Tuple[bool, float]:
        """Verify voice biometric"""
        if not self.speech_available or f"{user_id}_voice" not in self.templates:
            return False, 0.0

        try:
            # Load stored template
            template = self.templates[f"{user_id}_voice"]
            stored_features = template['features']

            # Process verification audio
            recognizer = sr.Recognizer()

            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)

            # Extract features from verification audio
            current_features = {
                'sample_rate': audio.sample_rate,
                'duration': len(audio.frame_data) / audio.sample_rate,
                'energy': sum(abs(sample) for sample in audio.frame_data) / len(audio.frame_data)
            }

            # Compare features (simplified similarity check)
            similarity = self._calculate_voice_similarity(stored_features, current_features)
            confidence = min(similarity / 100.0, 1.0)  # Normalize to 0-1

            return confidence > 0.7, confidence  # Threshold-based verification

        except Exception as e:
            logger.error(f"Voice verification failed for user {user_id}: {str(e)}")
            return False, 0.0

    def _calculate_voice_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate voice similarity score"""
        # Simplified similarity calculation
        similarity = 0.0
        weights = {'sample_rate': 0.3, 'duration': 0.3, 'energy': 0.4}

        for feature, weight in weights.items():
            if feature in features1 and feature in features2:
                val1, val2 = features1[feature], features2[feature]
                if val1 != 0:
                    diff = abs(val1 - val2) / abs(val1)
                    similarity += weight * (1.0 - diff)

        return similarity * 100.0

    def behavioral_auth(self, user_id: str, behavior_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Behavioral biometric authentication"""
        # Analyze typing patterns, mouse movements, etc.
        # This is a simplified implementation

        if user_id not in self.templates:
            # First time - enroll behavioral pattern
            self.templates[user_id] = {
                'type': BiometricType.BEHAVIORAL,
                'patterns': behavior_data,
                'enrolled_at': datetime.utcnow().isoformat()
            }
            return True, 1.0

        # Compare with stored patterns
        stored_patterns = self.templates[user_id]['patterns']

        # Simple pattern matching
        similarity = self._calculate_behavioral_similarity(stored_patterns, behavior_data)
        confidence = similarity / 100.0

        return confidence > 0.6, confidence

    def _calculate_behavioral_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate behavioral pattern similarity"""
        similarity = 0.0
        matched_features = 0

        for key in set(pattern1.keys()) | set(pattern2.keys()):
            if key in pattern1 and key in pattern2:
                val1, val2 = pattern1[key], pattern2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 != 0:
                        diff = abs(val1 - val2) / abs(val1)
                        similarity += (1.0 - diff)
                        matched_features += 1

        if matched_features > 0:
            return (similarity / matched_features) * 100.0
        return 0.0

class ZeroTrustAuth:
    """Zero Trust authentication and authorization"""

    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.context_providers: Dict[str, Callable] = {}
        self.decision_cache: Dict[str, Tuple[bool, datetime]] = {}
        self.cache_timeout = timedelta(minutes=5)

    def add_policy(self, policy_id: str, policy: Dict[str, Any]):
        """Add zero trust policy"""
        self.policies[policy_id] = policy
        logger.info(f"Added zero trust policy: {policy_id}")

    def add_context_provider(self, context_type: str, provider: Callable):
        """Add context provider"""
        self.context_providers[context_type] = provider

    def evaluate_access(self, user_id: str, resource: str, action: str,
                       context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate access using zero trust model"""

        # Build comprehensive context
        full_context = self._build_context(user_id, context)

        # Check cache
        cache_key = f"{user_id}:{resource}:{action}:{hash(str(sorted(full_context.items())))}"
        if cache_key in self.decision_cache:
            decision, timestamp = self.decision_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_timeout:
                return decision, {'cached': True, 'cache_time': timestamp}

        # Evaluate against policies
        allowed = False
        applied_policies = []

        for policy_id, policy in self.policies.items():
            if self._matches_policy(policy, user_id, resource, action, full_context):
                allowed = policy.get('effect', 'deny') == 'allow'
                applied_policies.append(policy_id)

                # Stop at first matching policy (in order of priority)
                break

        # Cache decision
        self.decision_cache[cache_key] = (allowed, datetime.utcnow())

        return allowed, {
            'policies_applied': applied_policies,
            'context_used': list(full_context.keys()),
            'risk_score': full_context.get('risk_score', 0.0)
        }

    def _build_context(self, user_id: str, provided_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build comprehensive context for decision"""
        context = provided_context or {}

        # Add user context
        context['user_id'] = user_id
        context['timestamp'] = datetime.utcnow()

        # Add contextual information from providers
        for context_type, provider in self.context_providers.items():
            try:
                context[context_type] = provider(user_id)
            except Exception as e:
                logger.warning(f"Context provider {context_type} failed: {str(e)}")

        # Calculate risk score
        context['risk_score'] = self._calculate_risk_score(context)

        return context

    def _calculate_risk_score(self, context: Dict[str, Any]) -> float:
        """Calculate risk score for access decision"""
        risk_score = 0.0

        # Location-based risk
        if context.get('location') != context.get('usual_location'):
            risk_score += 0.3

        # Time-based risk
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 0.2

        # Device risk
        if not context.get('device_trusted', False):
            risk_score += 0.4

        # Behavioral risk
        if context.get('behavioral_anomaly', False):
            risk_score += 0.5

        # Network risk
        if context.get('network_anomaly', False):
            risk_score += 0.3

        return min(risk_score, 1.0)

    def _matches_policy(self, policy: Dict[str, Any], user_id: str, resource: str,
                       action: str, context: Dict[str, Any]) -> bool:
        """Check if request matches policy conditions"""

        # Check subject (user)
        subject_conditions = policy.get('subject', {})
        if not self._matches_conditions(subject_conditions, {'user_id': user_id}, context):
            return False

        # Check resource
        resource_conditions = policy.get('resource', {})
        if not self._matches_conditions(resource_conditions, {'resource': resource}, context):
            return False

        # Check action
        action_conditions = policy.get('action', {})
        if not self._matches_conditions(action_conditions, {'action': action}, context):
            return False

        # Check environment/context
        env_conditions = policy.get('environment', {})
        if not self._matches_conditions(env_conditions, {}, context):
            return False

        return True

    def _matches_conditions(self, conditions: Dict[str, Any], request_data: Dict[str, Any],
                           context: Dict[str, Any]) -> bool:
        """Check if conditions match"""

        for condition_key, condition_value in conditions.items():
            # Get actual value from request or context
            actual_value = None
            if condition_key in request_data:
                actual_value = request_data[condition_key]
            elif condition_key in context:
                actual_value = context[condition_key]

            if actual_value is None:
                return False

            # Check condition
            if isinstance(condition_value, dict):
                # Complex condition
                if 'equals' in condition_value and actual_value != condition_value['equals']:
                    return False
                if 'in' in condition_value and actual_value not in condition_value['in']:
                    return False
                if 'regex' in condition_value and not re.match(condition_value['regex'], str(actual_value)):
                    return False
                if 'range' in condition_value:
                    min_val, max_val = condition_value['range']
                    if not (min_val <= actual_value <= max_val):
                        return False
            else:
                # Simple equality
                if actual_value != condition_value:
                    return False

        return True

    def continuous_verification(self, session_id: str) -> bool:
        """Continuous verification for zero trust"""
        # Implement continuous verification logic
        # Check user behavior, device health, network security, etc.

        # Placeholder implementation
        return True

    def adaptive_access(self, user_id: str, resource: str, action: str) -> Dict[str, Any]:
        """Adaptive access control based on context"""
        # Analyze user behavior patterns and adjust access levels

        # Placeholder implementation
        return {
            'access_level': 'normal',
            'additional_factors_required': [],
            'monitoring_level': 'standard'
        }

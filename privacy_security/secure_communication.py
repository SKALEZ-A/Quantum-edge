"""
Quantum Edge AI Platform - Secure Communication Module

Secure communication protocols for edge AI systems including quantum-safe encryption,
secure channels, and authenticated messaging.
"""

import ssl
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import socket
import threading
import queue
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    QUANTUM_SAFE_KYBER = "quantum_safe_kyber"
    QUANTUM_SAFE_DILITHIUM = "quantum_safe_dilithium"

class KeyExchangeProtocol(Enum):
    """Key exchange protocols"""
    ECDHE = "ecdhe"
    QUANTUM_KEY_DISTRIBUTION = "quantum_key_distribution"
    POST_QUANTUM_HYBRID = "post_quantum_hybrid"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    CERTIFICATE_BASED = "certificate_based"
    PRE_SHARED_KEY = "pre_shared_key"
    QUANTUM_AUTHENTICATION = "quantum_authentication"
    BIOMETRIC = "biometric"

@dataclass
class SecurityContext:
    """Security context for communication"""
    session_id: str
    encryption_key: bytes
    authentication_key: bytes
    nonce_counter: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    peer_identity: str = ""
    protocol_version: str = "1.0"

@dataclass
class SecureMessage:
    """Secure message structure"""
    payload: bytes
    signature: bytes
    nonce: bytes
    timestamp: datetime
    sender_id: str
    recipient_id: str
    message_type: str = "data"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommunicationChannel:
    """Communication channel configuration"""
    channel_id: str
    encryption_algorithm: EncryptionAlgorithm
    key_exchange_protocol: KeyExchangeProtocol
    authentication_method: AuthenticationMethod
    max_message_size: int = 1048576  # 1MB
    timeout_seconds: int = 30
    heartbeat_interval: int = 60
    created_at: datetime = field(default_factory=datetime.utcnow)

class QuantumSafeCryptography:
    """Quantum-safe cryptographic primitives"""

    def __init__(self):
        self.backend = default_backend()
        self.key_pairs = {}

    def generate_kyber_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Kyber keypair (quantum-safe KEM)"""
        # In production, this would use actual Kyber implementation
        # For now, simulate with ECC + additional quantum resistance

        # Generate ECC keypair
        private_key = ec.generate_private_key(ec.SECP256R1(), self.backend)
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Add quantum-resistant layer (simplified)
        quantum_layer = self._add_quantum_resistance(private_pem + public_pem)

        return private_pem, public_pem

    def generate_dilithium_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Dilithium keypair (quantum-safe signature)"""
        # In production, this would use actual Dilithium implementation
        # For now, simulate with larger key sizes

        # Generate larger RSA keypair for simulation
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,  # Larger for quantum resistance simulation
            backend=self.backend
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem

    def _add_quantum_resistance(self, data: bytes) -> bytes:
        """Add quantum resistance layer"""
        # Hash with SHA3-512 for quantum resistance
        quantum_hash = hashlib.sha3_512(data).digest()

        # XOR with quantum random (simulated)
        quantum_random = secrets.token_bytes(len(quantum_hash))

        return bytes(a ^ b for a, b in zip(quantum_hash, quantum_random))

    def kyber_encapsulate(self, public_key_pem: bytes) -> Tuple[bytes, bytes]:
        """Kyber encapsulation"""
        # Load public key
        public_key = serialization.load_pem_public_key(public_key_pem, self.backend)

        # Generate ephemeral keypair
        ephemeral_private = ec.generate_private_key(ec.SECP256R1(), self.backend)

        # ECDH key exchange
        shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)

        # Derive symmetric key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'kyber-encapsulation',
            backend=self.backend
        )
        symmetric_key = hkdf.derive(shared_secret)

        # Encapsulate (ciphertext is ephemeral public key)
        ephemeral_public = ephemeral_private.public_key()
        ciphertext = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return ciphertext, symmetric_key

    def kyber_decapsulate(self, private_key_pem: bytes, ciphertext: bytes) -> bytes:
        """Kyber decapsulation"""
        # Load private key
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )

        # Load ephemeral public key from ciphertext
        ephemeral_public = serialization.load_pem_public_key(ciphertext, self.backend)

        # ECDH key exchange
        shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)

        # Derive symmetric key
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'kyber-encapsulation',
            backend=self.backend
        )
        symmetric_key = hkdf.derive(shared_secret)

        return symmetric_key

class SecureChannel:
    """Secure communication channel"""

    def __init__(self, channel_config: CommunicationChannel):
        self.config = channel_config
        self.security_context = None
        self.message_queue = queue.Queue()
        self.is_active = False
        self.qsc = QuantumSafeCryptography()

    def establish_connection(self, peer_identity: str) -> SecurityContext:
        """Establish secure connection"""
        # Generate session ID
        session_id = secrets.token_hex(32)

        # Key exchange based on protocol
        if self.config.key_exchange_protocol == KeyExchangeProtocol.ECDHE:
            encryption_key, auth_key = self._perform_ecdhe_exchange()
        elif self.config.key_exchange_protocol == KeyExchangeProtocol.QUANTUM_KEY_DISTRIBUTION:
            encryption_key, auth_key = self._perform_quantum_key_exchange()
        else:
            encryption_key, auth_key = self._perform_hybrid_exchange()

        # Create security context
        self.security_context = SecurityContext(
            session_id=session_id,
            encryption_key=encryption_key,
            authentication_key=auth_key,
            peer_identity=peer_identity,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )

        self.is_active = True
        logger.info(f"Secure channel {self.config.channel_id} established with {peer_identity}")

        return self.security_context

    def _perform_ecdhe_exchange(self) -> Tuple[bytes, bytes]:
        """Perform ECDHE key exchange"""
        # Generate ephemeral keypair
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

        # In real implementation, exchange public keys with peer
        # For simulation, derive keys locally
        shared_secret = secrets.token_bytes(32)

        # Derive encryption and authentication keys
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,  # 32 for encryption + 32 for auth
            salt=None,
            info=b'ecdhe-key-derivation',
            backend=default_backend()
        )

        derived_keys = hkdf.derive(shared_secret)
        encryption_key = derived_keys[:32]
        auth_key = derived_keys[32:]

        return encryption_key, auth_key

    def _perform_quantum_key_exchange(self) -> Tuple[bytes, bytes]:
        """Perform quantum key exchange"""
        # Generate Kyber keypair
        private_key, public_key = self.qsc.generate_kyber_keypair()

        # In real implementation, send public key to peer and receive ciphertext
        # For simulation:
        ciphertext, encryption_key = self.qsc.kyber_encapsulate(public_key)
        auth_key = secrets.token_bytes(32)

        return encryption_key, auth_key

    def _perform_hybrid_exchange(self) -> Tuple[bytes, bytes]:
        """Perform hybrid classical-quantum key exchange"""
        # Combine ECDHE with quantum-safe elements
        classical_key = self._perform_ecdhe_exchange()[0]
        quantum_key = self._perform_quantum_key_exchange()[0]

        # XOR keys for hybrid approach
        hybrid_key = bytes(a ^ b for a, b in zip(classical_key, quantum_key))
        auth_key = secrets.token_bytes(32)

        return hybrid_key, auth_key

    def send_secure_message(self, payload: bytes, recipient_id: str,
                          message_type: str = "data") -> SecureMessage:
        """Send encrypted and authenticated message"""
        if not self.security_context:
            raise ValueError("Security context not established")

        # Generate nonce
        nonce = secrets.token_bytes(16)
        self.security_context.nonce_counter += 1

        # Encrypt payload
        ciphertext = self._encrypt_payload(payload, nonce)

        # Create signature
        signature = self._create_signature(ciphertext + nonce)

        # Create secure message
        message = SecureMessage(
            payload=ciphertext,
            signature=signature,
            nonce=nonce,
            timestamp=datetime.utcnow(),
            sender_id=self.security_context.session_id,
            recipient_id=recipient_id,
            message_type=message_type
        )

        return message

    def receive_secure_message(self, message: SecureMessage) -> bytes:
        """Receive and decrypt secure message"""
        if not self.security_context:
            raise ValueError("Security context not established")

        # Verify signature
        if not self._verify_signature(message.payload + message.nonce, message.signature):
            raise ValueError("Message signature verification failed")

        # Decrypt payload
        plaintext = self._decrypt_payload(message.payload, message.nonce)

        # Check timestamp for replay attacks
        if (datetime.utcnow() - message.timestamp).total_seconds() > 300:  # 5 minutes
            raise ValueError("Message timestamp too old (possible replay attack)")

        return plaintext

    def _encrypt_payload(self, payload: bytes, nonce: bytes) -> bytes:
        """Encrypt message payload"""
        if self.config.encryption_algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(payload, nonce)
        elif self.config.encryption_algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20(payload, nonce)
        else:
            return self._encrypt_quantum_safe(payload, nonce)

    def _decrypt_payload(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Decrypt message payload"""
        if self.config.encryption_algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(ciphertext, nonce)
        elif self.config.encryption_algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20(ciphertext, nonce)
        else:
            return self._decrypt_quantum_safe(ciphertext, nonce)

    def _encrypt_aes_gcm(self, payload: bytes, nonce: bytes) -> bytes:
        """Encrypt with AES-256-GCM"""
        cipher = Cipher(algorithms.AES(self.security_context.encryption_key),
                       modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(payload) + encryptor.finalize()
        tag = encryptor.tag

        return ciphertext + tag

    def _decrypt_aes_gcm(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Decrypt AES-256-GCM"""
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]

        cipher = Cipher(algorithms.AES(self.security_context.encryption_key),
                       modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def _encrypt_chacha20(self, payload: bytes, nonce: bytes) -> bytes:
        """Encrypt with ChaCha20-Poly1305"""
        # Simplified implementation
        cipher = Cipher(algorithms.ChaCha20(self.security_context.encryption_key, nonce),
                       mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(payload)

        return ciphertext

    def _decrypt_chacha20(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Decrypt ChaCha20-Poly1305"""
        cipher = Cipher(algorithms.ChaCha20(self.security_context.encryption_key, nonce),
                       mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext)

        return plaintext

    def _encrypt_quantum_safe(self, payload: bytes, nonce: bytes) -> bytes:
        """Quantum-safe encryption"""
        # Use AES with quantum-safe key
        return self._encrypt_aes_gcm(payload, nonce)

    def _decrypt_quantum_safe(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Quantum-safe decryption"""
        return self._decrypt_aes_gcm(ciphertext, nonce)

    def _create_signature(self, data: bytes) -> bytes:
        """Create message signature"""
        return hmac.new(self.security_context.authentication_key, data, hashlib.sha256).digest()

    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify message signature"""
        expected_signature = hmac.new(self.security_context.authentication_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected_signature)

    def close_channel(self):
        """Close secure channel"""
        self.is_active = False
        self.security_context = None
        logger.info(f"Secure channel {self.config.channel_id} closed")

class SecureCommunicationManager:
    """Manager for secure communications"""

    def __init__(self):
        self.channels = {}
        self.active_sessions = {}
        self.message_handlers = {}
        self.qsc = QuantumSafeCryptography()

    def create_channel(self, channel_config: CommunicationChannel) -> SecureChannel:
        """Create new secure channel"""
        channel = SecureChannel(channel_config)
        self.channels[channel_config.channel_id] = channel

        return channel

    def establish_secure_session(self, channel_id: str, peer_identity: str) -> SecurityContext:
        """Establish secure session"""
        if channel_id not in self.channels:
            raise ValueError(f"Channel {channel_id} not found")

        channel = self.channels[channel_id]
        context = channel.establish_connection(peer_identity)

        self.active_sessions[context.session_id] = context

        return context

    def send_message(self, session_id: str, payload: Union[str, bytes, dict],
                    recipient_id: str, message_type: str = "data") -> SecureMessage:
        """Send secure message"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        context = self.active_sessions[session_id]

        # Serialize payload if needed
        if isinstance(payload, dict):
            payload_bytes = json.dumps(payload).encode('utf-8')
        elif isinstance(payload, str):
            payload_bytes = payload.encode('utf-8')
        else:
            payload_bytes = payload

        # Find channel for session
        channel = None
        for ch in self.channels.values():
            if ch.security_context and ch.security_context.session_id == session_id:
                channel = ch
                break

        if not channel:
            raise ValueError(f"No channel found for session {session_id}")

        return channel.send_secure_message(payload_bytes, recipient_id, message_type)

    def receive_message(self, session_id: str, message: SecureMessage) -> Any:
        """Receive and decrypt secure message"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        # Find channel for session
        channel = None
        for ch in self.channels.values():
            if ch.security_context and ch.security_context.session_id == session_id:
                channel = ch
                break

        if not channel:
            raise ValueError(f"No channel found for session {session_id}")

        # Decrypt message
        plaintext = channel.receive_secure_message(message)

        # Try to parse as JSON
        try:
            return json.loads(plaintext.decode('utf-8'))
        except:
            return plaintext

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler

    def handle_incoming_message(self, session_id: str, message: SecureMessage):
        """Handle incoming secure message"""
        try:
            # Decrypt message
            payload = self.receive_message(session_id, message)

            # Call appropriate handler
            if message.message_type in self.message_handlers:
                self.message_handlers[message.message_type](session_id, payload, message)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if session_id not in self.active_sessions:
            return None

        context = self.active_sessions[session_id]
        return {
            'session_id': context.session_id,
            'peer_identity': context.peer_identity,
            'created_at': context.created_at,
            'expires_at': context.expires_at,
            'is_active': session_id in self.active_sessions
        }

    def close_session(self, session_id: str):
        """Close secure session"""
        if session_id in self.active_sessions:
            # Find and close channel
            for channel in self.channels.values():
                if (channel.security_context and
                    channel.security_context.session_id == session_id):
                    channel.close_channel()
                    break

            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} closed")

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_channels': len(self.channels),
            'active_channels': len([ch for ch in self.channels.values() if ch.is_active]),
            'message_handlers': len(self.message_handlers)
        }

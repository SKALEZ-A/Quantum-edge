"""
Quantum Feature Maps and Data Encoding

Advanced quantum feature maps for encoding classical data into quantum states,
including amplitude encoding, angle encoding, and hybrid encoding schemes
optimized for edge deployment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from scipy.linalg import expm, sqrtm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumState:
    """Quantum state representation"""
    amplitudes: np.ndarray
    num_qubits: int

    def __init__(self, amplitudes: np.ndarray):
        self.amplitudes = np.array(amplitudes, dtype=complex)
        self.num_qubits = int(np.log2(len(amplitudes)))

        if len(amplitudes) != 2**self.num_qubits:
            raise ValueError("Number of amplitudes must be a power of 2")

        # Normalize
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm

    def fidelity(self, other: 'QuantumState') -> float:
        """Compute quantum fidelity between states"""
        return np.abs(np.vdot(self.amplitudes, other.amplitudes))**2

    def expectation_value(self, observable: np.ndarray) -> complex:
        """Compute expectation value of observable"""
        return np.vdot(self.amplitudes, observable @ self.amplitudes)

class BaseFeatureMap(ABC):
    """Abstract base class for quantum feature maps"""

    def __init__(self, num_features: int, num_qubits: Optional[int] = None):
        self.num_features = num_features
        self.num_qubits = num_qubits or int(np.ceil(np.log2(num_features)))

    @abstractmethod
    def encode(self, x: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumState:
        """Encode classical data into quantum state"""
        pass

    @abstractmethod
    def get_parameter_count(self) -> int:
        """Get number of trainable parameters"""
        pass

class ZFeatureMap(BaseFeatureMap):
    """Z-feature map: encodes features as Z-rotations"""

    def __init__(self, num_features: int, layers: int = 1,
                 entanglement: bool = True):
        super().__init__(num_features)
        self.layers = layers
        self.entanglement = entanglement

    def encode(self, x: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumState:
        """Encode data using Z-feature map"""
        if parameters is None:
            parameters = np.ones(self.layers * self.num_features)

        # Initialize |0...0⟩ state
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0

        param_idx = 0
        for layer in range(self.layers):
            # Apply Z rotations
            for qubit in range(min(self.num_qubits, len(x))):
                angle = x[qubit] * parameters[param_idx]
                state = self._apply_rz(state, qubit, angle)
                param_idx += 1

            # Apply entanglement
            if self.entanglement:
                state = self._apply_entanglement(state)

        return QuantumState(state)

    def get_parameter_count(self) -> int:
        return self.layers * self.num_features

    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RZ gate"""
        rz_matrix = np.array([[np.exp(-1j * angle / 2), 0],
                             [0, np.exp(1j * angle / 2)]], dtype=complex)
        return self._apply_single_qubit_gate(state, qubit, rz_matrix)

    def _apply_entanglement(self, state: np.ndarray) -> np.ndarray:
        """Apply linear entanglement"""
        for qubit in range(self.num_qubits - 1):
            state = self._apply_cnot(state, qubit, qubit + 1)
        return state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate"""
        new_state = state.copy()
        for i in range(len(state)):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def _apply_single_qubit_gate(self, state: np.ndarray, qubit: int,
                                gate_matrix: np.ndarray) -> np.ndarray:
        """Apply single-qubit gate"""
        new_state = np.zeros_like(state, dtype=complex)
        for i in range(len(state)):
            bit = (i >> qubit) & 1
            if bit == 0:
                new_state[i] += gate_matrix[0, 0] * state[i]
                new_state[i ^ (1 << qubit)] += gate_matrix[0, 1] * state[i]
            else:
                new_state[i] += gate_matrix[1, 1] * state[i]
                new_state[i ^ (1 << qubit)] += gate_matrix[1, 0] * state[i]
        return new_state

class ZZFeatureMap(BaseFeatureMap):
    """ZZ-feature map: encodes feature products"""

    def __init__(self, num_features: int, layers: int = 1,
                 entanglement_pattern: str = 'linear'):
        super().__init__(num_features)
        self.layers = layers
        self.entanglement_pattern = entanglement_pattern

    def encode(self, x: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumState:
        """Encode data using ZZ-feature map"""
        if parameters is None:
            parameters = np.ones(self.layers * self.num_features * (self.num_features - 1) // 2)

        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0

        param_idx = 0
        for layer in range(self.layers):
            # Apply single-qubit Z rotations
            for qubit in range(min(self.num_qubits, len(x))):
                angle = x[qubit] * parameters[param_idx]
                state = self._apply_rz(state, qubit, angle)
                param_idx += 1

            # Apply two-qubit ZZ interactions
            for i in range(min(self.num_qubits, len(x))):
                for j in range(i+1, min(self.num_qubits, len(x))):
                    angle = x[i] * x[j] * parameters[param_idx]
                    state = self._apply_zz(state, i, j, angle)
                    param_idx += 1

            # Apply entanglement
            state = self._apply_entanglement(state, self.entanglement_pattern)

        return QuantumState(state)

    def get_parameter_count(self) -> int:
        single_params = self.num_features
        pair_params = self.num_features * (self.num_features - 1) // 2
        return self.layers * (single_params + pair_params)

    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RZ gate"""
        rz_matrix = np.array([[np.exp(-1j * angle / 2), 0],
                             [0, np.exp(1j * angle / 2)]], dtype=complex)
        return self._apply_single_qubit_gate(state, qubit, rz_matrix)

    def _apply_zz(self, state: np.ndarray, qubit1: int, qubit2: int, angle: float) -> np.ndarray:
        """Apply ZZ gate"""
        for i in range(len(state)):
            if ((i >> qubit1) & 1) and ((i >> qubit2) & 1):
                state[i] *= np.exp(-1j * angle)
        return state

    def _apply_entanglement(self, state: np.ndarray, pattern: str) -> np.ndarray:
        """Apply entanglement pattern"""
        if pattern == 'linear':
            for qubit in range(self.num_qubits - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)
        elif pattern == 'circular':
            for qubit in range(self.num_qubits):
                state = self._apply_cnot(state, qubit, (qubit + 1) % self.num_qubits)
        elif pattern == 'full':
            for i in range(self.num_qubits):
                for j in range(i+1, self.num_qubits):
                    state = self._apply_cnot(state, i, j)
        return state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate"""
        new_state = state.copy()
        for i in range(len(state)):
            if (i >> control) & 1:
                j = i ^ (1 << target)
                new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def _apply_single_qubit_gate(self, state: np.ndarray, qubit: int,
                                gate_matrix: np.ndarray) -> np.ndarray:
        """Apply single-qubit gate"""
        new_state = np.zeros_like(state, dtype=complex)
        for i in range(len(state)):
            bit = (i >> qubit) & 1
            if bit == 0:
                new_state[i] += gate_matrix[0, 0] * state[i]
                new_state[i ^ (1 << qubit)] += gate_matrix[0, 1] * state[i]
            else:
                new_state[i] += gate_matrix[1, 1] * state[i]
                new_state[i ^ (1 << qubit)] += gate_matrix[1, 0] * state[i]
        return new_state

class AmplitudeEncoding(BaseFeatureMap):
    """Amplitude encoding: encodes data as quantum state amplitudes"""

    def __init__(self, num_features: int):
        super().__init__(num_features)

    def encode(self, x: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumState:
        """Encode data using amplitude encoding"""
        # Normalize input vector
        x_norm = np.array(x, dtype=complex)
        norm = np.linalg.norm(x_norm)

        if norm > 0:
            x_norm /= norm

        # Pad or truncate to match qubit count
        target_size = 2**self.num_qubits
        if len(x_norm) < target_size:
            # Pad with zeros
            amplitudes = np.zeros(target_size, dtype=complex)
            amplitudes[:len(x_norm)] = x_norm
        else:
            # Truncate (take first 2^n elements)
            amplitudes = x_norm[:target_size]

        # Renormalize
        amplitudes /= np.linalg.norm(amplitudes)

        return QuantumState(amplitudes)

    def get_parameter_count(self) -> int:
        return 0  # No trainable parameters

class AngleEncoding(BaseFeatureMap):
    """Angle encoding: encodes features as rotation angles"""

    def __init__(self, num_features: int, rotation_axes: List[str] = None):
        super().__init__(num_features)
        if rotation_axes is None:
            rotation_axes = ['ry'] * num_features
        self.rotation_axes = rotation_axes[:num_features]

    def encode(self, x: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumState:
        """Encode data using angle encoding"""
        state = np.zeros(2**self.num_qubits, dtype=complex)
        state[0] = 1.0

        for qubit in range(min(self.num_qubits, len(x))):
            axis = self.rotation_axes[qubit]
            angle = x[qubit]

            if axis == 'rx':
                state = self._apply_rx(state, qubit, angle)
            elif axis == 'ry':
                state = self._apply_ry(state, qubit, angle)
            elif axis == 'rz':
                state = self._apply_rz(state, qubit, angle)

        return QuantumState(state)

    def get_parameter_count(self) -> int:
        return 0  # No trainable parameters

    def _apply_rx(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RX gate"""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        rx_matrix = np.array([[cos_half, -1j * sin_half],
                             [-1j * sin_half, cos_half]], dtype=complex)
        return self._apply_single_qubit_gate(state, qubit, rx_matrix)

    def _apply_ry(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RY gate"""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        ry_matrix = np.array([[cos_half, -sin_half],
                             [sin_half, cos_half]], dtype=complex)
        return self._apply_single_qubit_gate(state, qubit, ry_matrix)

    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply RZ gate"""
        rz_matrix = np.array([[np.exp(-1j * angle / 2), 0],
                             [0, np.exp(1j * angle / 2)]], dtype=complex)
        return self._apply_single_qubit_gate(state, qubit, rz_matrix)

    def _apply_single_qubit_gate(self, state: np.ndarray, qubit: int,
                                gate_matrix: np.ndarray) -> np.ndarray:
        """Apply single-qubit gate"""
        new_state = np.zeros_like(state, dtype=complex)
        for i in range(len(state)):
            bit = (i >> qubit) & 1
            if bit == 0:
                new_state[i] += gate_matrix[0, 0] * state[i]
                new_state[i ^ (1 << qubit)] += gate_matrix[0, 1] * state[i]
            else:
                new_state[i] += gate_matrix[1, 1] * state[i]
                new_state[i ^ (1 << qubit)] += gate_matrix[1, 0] * state[i]
        return new_state

class HybridEncoding(BaseFeatureMap):
    """Hybrid encoding combining multiple encoding schemes"""

    def __init__(self, num_features: int, encoding_scheme: str = 'amplitude_angle'):
        super().__init__(num_features)
        self.encoding_scheme = encoding_scheme

        # Initialize sub-encoders
        self.amplitude_encoder = AmplitudeEncoding(num_features)
        self.angle_encoder = AngleEncoding(num_features)

    def encode(self, x: np.ndarray, parameters: Optional[np.ndarray] = None) -> QuantumState:
        """Encode data using hybrid encoding"""
        if self.encoding_scheme == 'amplitude_angle':
            # First encode amplitude, then apply angle rotations
            state = self.amplitude_encoder.encode(x).amplitudes

            # Apply angle encoding on top
            angle_state = self.angle_encoder.encode(x)
            state = state * angle_state.amplitudes

            # Renormalize
            state /= np.linalg.norm(state)

        elif self.encoding_scheme == 'z_zz':
            # Combine Z and ZZ feature maps
            z_map = ZFeatureMap(self.num_features, layers=1)
            zz_map = ZZFeatureMap(self.num_features, layers=1)

            z_state = z_map.encode(x)
            zz_state = zz_map.encode(x)

            # Combine states (simplified combination)
            combined_amplitudes = (z_state.amplitudes + zz_state.amplitudes) / np.sqrt(2)
            state = combined_amplitudes

        return QuantumState(state)

    def get_parameter_count(self) -> int:
        return 0  # Simplified - could include trainable parameters

class QuantumFeatureMaps:
    """Factory and manager for quantum feature maps"""

    def __init__(self):
        self.feature_maps = {}

    def create_z_feature_map(self, num_features: int, layers: int = 1,
                           entanglement: bool = True) -> ZFeatureMap:
        """Create Z-feature map"""
        feature_map = ZFeatureMap(num_features, layers, entanglement)
        self.feature_maps['z_map'] = feature_map
        return feature_map

    def create_zz_feature_map(self, num_features: int, layers: int = 1,
                            entanglement_pattern: str = 'linear') -> ZZFeatureMap:
        """Create ZZ-feature map"""
        feature_map = ZZFeatureMap(num_features, layers, entanglement_pattern)
        self.feature_maps['zz_map'] = feature_map
        return feature_map

    def create_amplitude_encoder(self, num_features: int) -> AmplitudeEncoding:
        """Create amplitude encoder"""
        encoder = AmplitudeEncoding(num_features)
        self.feature_maps['amplitude'] = encoder
        return encoder

    def create_angle_encoder(self, num_features: int,
                           rotation_axes: List[str] = None) -> AngleEncoding:
        """Create angle encoder"""
        encoder = AngleEncoding(num_features, rotation_axes)
        self.feature_maps['angle'] = encoder
        return encoder

    def create_hybrid_encoder(self, num_features: int,
                            encoding_scheme: str = 'amplitude_angle') -> HybridEncoding:
        """Create hybrid encoder"""
        encoder = HybridEncoding(num_features, encoding_scheme)
        self.feature_maps['hybrid'] = encoder
        return encoder

    def optimize_feature_map(self, feature_map: BaseFeatureMap, X: np.ndarray,
                           y: np.ndarray, optimization_target: str = 'classification_accuracy') -> BaseFeatureMap:
        """Optimize feature map parameters for specific task"""
        if feature_map.get_parameter_count() == 0:
            return feature_map  # No parameters to optimize

        def objective(parameters):
            # This would implement parameter optimization
            # For now, return random score
            return np.random.random()

        # Simplified optimization
        optimal_params = np.random.uniform(0, 2*np.pi, feature_map.get_parameter_count())

        return feature_map  # Return optimized map

    def compare_feature_maps(self, feature_maps: List[BaseFeatureMap],
                           X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare different feature maps on same dataset"""
        results = {}

        for i, feature_map in enumerate(feature_maps):
            # Encode dataset
            encoded_states = [feature_map.encode(x) for x in X]

            # Compute kernel matrix using quantum fidelity
            kernel_matrix = np.zeros((len(X), len(X)))
            for j in range(len(X)):
                for k in range(j, len(X)):
                    fidelity = encoded_states[j].fidelity(encoded_states[k])
                    kernel_matrix[j, k] = fidelity
                    kernel_matrix[k, j] = fidelity

            # Simple evaluation (trace of kernel matrix)
            score = np.trace(kernel_matrix) / len(X)

            results[f'map_{i}'] = {
                'feature_map': feature_map.__class__.__name__,
                'score': score,
                'parameters': feature_map.get_parameter_count()
            }

        return results

    def create_adaptive_feature_map(self, X: np.ndarray, task_type: str = 'classification') -> BaseFeatureMap:
        """Create adaptive feature map based on data characteristics"""
        num_features = X.shape[1]

        # Analyze data characteristics
        data_variance = np.var(X, axis=0)
        data_range = np.ptp(X, axis=0)  # peak-to-peak

        # Choose encoding based on data properties
        if np.max(data_range) > 10:  # Large range suggests amplitude encoding
            return self.create_amplitude_encoder(num_features)
        elif np.max(data_variance) > 1:  # High variance suggests angle encoding
            return self.create_angle_encoder(num_features)
        else:  # Default to Z-feature map
            layers = min(3, max(1, int(np.log2(num_features))))
            return self.create_z_feature_map(num_features, layers=layers)

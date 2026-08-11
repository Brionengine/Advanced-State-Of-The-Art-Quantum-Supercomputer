"""
Base abstract classes for quantum backend abstraction layer
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class BackendType(Enum):
    """Quantum backend types"""
    CIRQ = "cirq"
    QISKIT = "qiskit"
    TFQ = "tensorflow_quantum"
    SIMULATOR = "simulator"
    HARDWARE = "hardware"


class ExecutionMode(Enum):
    """Quantum execution modes"""
    SIMULATION = "simulation"
    REAL_QUANTUM = "real_quantum"
    HYBRID = "hybrid"


@dataclass
class QuantumResult:
    """Standardized quantum execution result"""
    counts: Dict[str, int]
    statevector: Optional[np.ndarray] = None
    probabilities: Optional[Dict[str, float]] = None
    execution_time: float = 0.0
    backend_name: str = ""
    num_qubits: int = 0
    shots: int = 0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Calculate probabilities if not provided
        if self.probabilities is None and self.counts:
            total_shots = sum(self.counts.values())
            self.probabilities = {
                state: count / total_shots
                for state, count in self.counts.items()
            }


@dataclass
class QuantumCircuit:
    """Universal quantum circuit representation"""
    num_qubits: int
    num_classical_bits: int = 0
    gates: List[Dict[str, Any]] = None
    measurements: List[int] = None
    name: str = "quantum_circuit"
    parameters: Dict[str, float] = None

    def __post_init__(self):
        if self.gates is None:
            self.gates = []
        if self.measurements is None:
            self.measurements = []
        if self.parameters is None:
            self.parameters = {}
        if self.num_classical_bits == 0:
            self.num_classical_bits = self.num_qubits


class QuantumBackend(ABC):
    """
    Abstract base class for quantum computing backends

    This provides a unified interface across Cirq, Qiskit, and TFQ,
    enabling seamless switching between quantum hardware providers.
    """

    def __init__(
        self,
        backend_name: str,
        backend_type: BackendType,
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        **kwargs
    ):
        self.backend_name = backend_name
        self.backend_type = backend_type
        self.execution_mode = execution_mode
        self.config = kwargs
        self._native_backend = None
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the quantum backend

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    def create_circuit(
        self,
        num_qubits: int,
        num_classical_bits: Optional[int] = None
    ) -> Any:
        """
        Create a native quantum circuit for this backend

        Args:
            num_qubits: Number of qubits
            num_classical_bits: Number of classical measurement bits

        Returns:
            Native circuit object (Cirq.Circuit, QuantumCircuit, etc.)
        """
        pass

    @abstractmethod
    def execute(
        self,
        circuit: Any,
        shots: int = 1024,
        **kwargs
    ) -> QuantumResult:
        """
        Execute a quantum circuit

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            **kwargs: Additional backend-specific parameters

        Returns:
            QuantumResult object with standardized results
        """
        pass

    @abstractmethod
    def transpile(
        self,
        circuit: Any,
        optimization_level: int = 1,
        **kwargs
    ) -> Any:
        """
        Transpile circuit for target backend

        Args:
            circuit: Input quantum circuit
            optimization_level: Optimization level (0-3)
            **kwargs: Backend-specific options

        Returns:
            Transpiled circuit
        """
        pass

    @abstractmethod
    def get_backend_properties(self) -> Dict[str, Any]:
        """
        Get properties of the quantum backend

        Returns:
            Dictionary with backend capabilities and properties
        """
        pass

    def convert_to_universal(self, native_circuit: Any) -> QuantumCircuit:
        """
        Convert native circuit to universal QuantumCircuit representation

        Args:
            native_circuit: Backend-specific circuit object

        Returns:
            QuantumCircuit object
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclass must implement convert_to_universal")

    def convert_from_universal(self, circuit: QuantumCircuit) -> Any:
        """
        Convert universal QuantumCircuit to native circuit format

        Args:
            circuit: QuantumCircuit object

        Returns:
            Native circuit object for this backend
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclass must implement convert_from_universal")

    def validate_circuit(self, circuit: Any) -> bool:
        """
        Validate that circuit can run on this backend

        Args:
            circuit: Circuit to validate

        Returns:
            bool: True if valid
        """
        props = self.get_backend_properties()
        # Basic validation - subclasses can override
        return True

    def estimate_resources(self, circuit: Any) -> Dict[str, Any]:
        """
        Estimate resources required to run circuit

        Args:
            circuit: Quantum circuit

        Returns:
            Dictionary with resource estimates
        """
        return {
            'num_qubits': 0,
            'depth': 0,
            'gate_count': 0,
            'estimated_time': 0.0,
            'estimated_error_rate': 0.0
        }

    @property
    def is_available(self) -> bool:
        """Check if backend is available and ready"""
        return self._is_initialized

    @property
    def num_qubits(self) -> int:
        """Get number of qubits available on backend"""
        props = self.get_backend_properties()
        return props.get('num_qubits', 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.backend_name}', "
            f"type={self.backend_type.value}, "
            f"mode={self.execution_mode.value}, "
            f"available={self.is_available})"
        )

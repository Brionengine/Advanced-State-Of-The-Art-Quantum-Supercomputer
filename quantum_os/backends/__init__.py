"""
Quantum OS Backend Abstraction Layer

Provides unified interface for multiple quantum computing backends:
- Google Cirq (Willow simulator and hardware)
- IBM Qiskit (Brisbane, Torino processors)
- TensorFlow Quantum (hybrid quantum-classical ML)
"""

from .base import (
    BackendType,
    ExecutionMode,
    QuantumBackend,
    QuantumCircuit,
    QuantumResult,
)
from .cirq_backend import CirqBackend
from .qiskit_backend import QiskitBackend
from .tfq_backend import TFQBackend

__all__ = [
    'QuantumBackend',
    'QuantumCircuit',
    'QuantumResult',
    # Re-exported from .base: the kernel imports these from the package root,
    # so omitting them here broke `from ..backends import ExecutionMode`.
    'BackendType',
    'ExecutionMode',
    'CirqBackend',
    'QiskitBackend',
    'TFQBackend',
]

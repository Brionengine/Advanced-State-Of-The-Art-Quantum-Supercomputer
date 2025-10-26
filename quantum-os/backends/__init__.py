"""
Quantum OS Backend Abstraction Layer

Provides unified interface for multiple quantum computing backends:
- Google Cirq (Willow simulator and hardware)
- IBM Qiskit (Brisbane, Torino processors)
- TensorFlow Quantum (hybrid quantum-classical ML)
"""

from .base import QuantumBackend, QuantumCircuit, QuantumResult
from .cirq_backend import CirqBackend
from .qiskit_backend import QiskitBackend
from .tfq_backend import TFQBackend

__all__ = [
    'QuantumBackend',
    'QuantumCircuit',
    'QuantumResult',
    'CirqBackend',
    'QiskitBackend',
    'TFQBackend',
]

"""
Echo Circuit Builder
===================

Build quantum circuits using echo-based operations.
Transpiles standard gates to topological implementations.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.quantum_echoes import EchoCircuit

logger = logging.getLogger(__name__)


class EchoCircuitBuilder:
    """
    Builder for quantum circuits using Quantum Echoes framework.

    Provides high-level interface for constructing circuits that will
    be executed using topological qubits and echo operations.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize circuit builder.

        Args:
            num_qubits: Number of qubits in circuit
        """
        self.num_qubits = num_qubits
        self.circuit = EchoCircuit(num_qubits=num_qubits)

        logger.info(f"Initialized circuit builder with {num_qubits} qubits")

    def h(self, qubit: int) -> 'EchoCircuitBuilder':
        """Apply Hadamard gate."""
        self.circuit.add_gate('H', [qubit])
        return self

    def x(self, qubit: int) -> 'EchoCircuitBuilder':
        """Apply X (NOT) gate."""
        self.circuit.add_gate('X', [qubit])
        return self

    def y(self, qubit: int) -> 'EchoCircuitBuilder':
        """Apply Y gate."""
        self.circuit.add_gate('Y', [qubit])
        return self

    def z(self, qubit: int) -> 'EchoCircuitBuilder':
        """Apply Z gate."""
        self.circuit.add_gate('Z', [qubit])
        return self

    def s(self, qubit: int) -> 'EchoCircuitBuilder':
        """Apply S gate."""
        self.circuit.add_gate('S', [qubit])
        return self

    def t(self, qubit: int) -> 'EchoCircuitBuilder':
        """Apply T gate."""
        self.circuit.add_gate('T', [qubit])
        return self

    def cnot(self, control: int, target: int) -> 'EchoCircuitBuilder':
        """Apply CNOT gate."""
        self.circuit.add_gate('CNOT', [control, target])
        return self

    def cz(self, control: int, target: int) -> 'EchoCircuitBuilder':
        """Apply CZ gate."""
        self.circuit.add_gate('CZ', [control, target])
        return self

    def swap(self, qubit1: int, qubit2: int) -> 'EchoCircuitBuilder':
        """Apply SWAP gate."""
        self.circuit.add_gate('SWAP', [qubit1, qubit2])
        return self

    def measure(self, qubit: int) -> 'EchoCircuitBuilder':
        """Add measurement."""
        self.circuit.add_measurement(qubit)
        return self

    def measure_all(self) -> 'EchoCircuitBuilder':
        """Measure all qubits."""
        for i in range(self.num_qubits):
            self.circuit.add_measurement(i)
        return self

    def build(self) -> EchoCircuit:
        """Return constructed circuit."""
        logger.info(f"Built circuit with depth {self.circuit.depth()}")
        return self.circuit

    # Composite gates
    def bell_pair(self, qubit1: int, qubit2: int) -> 'EchoCircuitBuilder':
        """Create Bell pair between two qubits."""
        self.h(qubit1)
        self.cnot(qubit1, qubit2)
        return self

    def ghz_state(self, qubits: List[int]) -> 'EchoCircuitBuilder':
        """Create GHZ state on specified qubits."""
        if len(qubits) < 2:
            raise ValueError("Need at least 2 qubits for GHZ state")

        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cnot(qubits[0], qubits[i])

        return self

    def qft(self, qubits: List[int]) -> 'EchoCircuitBuilder':
        """Apply Quantum Fourier Transform."""
        n = len(qubits)

        for j in range(n):
            self.h(qubits[j])

            for k in range(j + 1, n):
                # Controlled phase rotation
                angle = np.pi / (2 ** (k - j))
                # For now, approximate with T gates
                self.t(qubits[k])

        # Reverse qubit order (using swaps)
        for i in range(n // 2):
            self.swap(qubits[i], qubits[n - 1 - i])

        return self


def build_circuit(num_qubits: int) -> EchoCircuitBuilder:
    """
    Convenience function to create circuit builder.

    Args:
        num_qubits: Number of qubits

    Returns:
        EchoCircuitBuilder instance
    """
    return EchoCircuitBuilder(num_qubits)


# Example circuits
def create_bell_circuit() -> EchoCircuit:
    """Create Bell state circuit."""
    builder = build_circuit(2)
    builder.h(0).cnot(0, 1).measure_all()
    return builder.build()


def create_ghz_circuit(num_qubits: int = 3) -> EchoCircuit:
    """Create GHZ state circuit."""
    builder = build_circuit(num_qubits)
    builder.ghz_state(list(range(num_qubits))).measure_all()
    return builder.build()


def create_qft_circuit(num_qubits: int = 3) -> EchoCircuit:
    """Create QFT circuit."""
    builder = build_circuit(num_qubits)
    builder.qft(list(range(num_qubits))).measure_all()
    return builder.build()

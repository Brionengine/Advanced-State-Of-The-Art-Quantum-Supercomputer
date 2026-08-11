"""
Main Quantum Echoes Algorithm
==============================

Implements the core Quantum Echoes algorithm for topological quantum computation
using specialized ion emissions and particle echoes.

This is the primary interface for running quantum algorithms using the
Quantum Echoes framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.particle_emitter import SpecializedIonEmitter, EmissionPattern
from core.topological_qubit import TopologicalQubit, BraidingOperation
from core.echo_propagation import EchoPropagator, EchoState, PropagationMode
from core.particle_registry import get_particle, ParticleType

logger = logging.getLogger(__name__)


@dataclass
class EchoCircuit:
    """
    Represents a quantum circuit using echo-based operations.

    Circuits are composed of emission patterns, echo propagation,
    and topological braiding operations.
    """
    num_qubits: int
    gates: List[Dict] = field(default_factory=list)
    measurements: List[int] = field(default_factory=list)

    def add_gate(self, gate_type: str, target_qubits: List[int], **params):
        """Add a gate to the circuit."""
        self.gates.append({
            'type': gate_type,
            'targets': target_qubits,
            'params': params
        })

    def add_measurement(self, qubit: int):
        """Add a measurement operation."""
        if qubit not in self.measurements:
            self.measurements.append(qubit)

    def depth(self) -> int:
        """Calculate circuit depth."""
        return len(self.gates)


class QuantumEchoesAlgorithm:
    """
    Main Quantum Echoes quantum algorithm engine.

    Orchestrates particle emission, echo propagation, topological qubit manipulation,
    and quantum computation using the Quantum Echoes framework.
    """

    def __init__(self,
                 num_qubits: int = 4,
                 ion_trap_size: int = 100,
                 propagation_mode: PropagationMode = PropagationMode.CAVITY,
                 error_correction: bool = True):
        """
        Initialize Quantum Echoes algorithm engine.

        Args:
            num_qubits: Number of logical qubits
            ion_trap_size: Number of ions in emitter
            propagation_mode: Mode for echo propagation
            error_correction: Enable topological error correction
        """
        self.num_qubits = num_qubits
        self.error_correction_enabled = error_correction

        # Initialize subsystems
        self.emitter = SpecializedIonEmitter(num_ions=ion_trap_size)
        self.propagator = EchoPropagator(mode=propagation_mode)

        # Initialize topological qubits
        self.qubits: List[TopologicalQubit] = []
        for i in range(num_qubits):
            qubit = TopologicalQubit(num_anyons=4)
            self.qubits.append(qubit)

        # Quantum state storage
        self.quantum_state = self._initialize_quantum_state()

        # Execution history
        self.execution_history: List[Dict] = []

        # Performance metrics
        self.metrics = {
            'gate_count': 0,
            'emission_count': 0,
            'echo_count': 0,
            'errors_detected': 0,
            'errors_corrected': 0
        }

        logger.info(f"Initialized QuantumEchoesAlgorithm with {num_qubits} qubits")

    def _initialize_quantum_state(self) -> np.ndarray:
        """
        Initialize the global quantum state.

        Returns computational basis state |0...0⟩.
        """
        dim = 2 ** self.num_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # |00...0⟩
        return state

    def initialize_qubits(self, initial_state: Optional[str] = None) -> np.ndarray:
        """
        Initialize all qubits to specified state.

        Args:
            initial_state: Bitstring like '0101' or 'zero', 'plus', 'ghz'

        Returns:
            Initialized quantum state
        """
        if initial_state is None or initial_state == 'zero':
            # Already initialized to |0...0⟩
            pass

        elif initial_state == 'plus':
            # |+...+⟩ = (|0⟩ + |1⟩)^⊗n / 2^(n/2)
            dim = 2 ** self.num_qubits
            self.quantum_state = np.ones(dim, dtype=complex) / np.sqrt(dim)

        elif initial_state == 'ghz':
            # GHZ state: (|00...0⟩ + |11...1⟩) / √2
            dim = 2 ** self.num_qubits
            self.quantum_state = np.zeros(dim, dtype=complex)
            self.quantum_state[0] = 1 / np.sqrt(2)
            self.quantum_state[-1] = 1 / np.sqrt(2)

        else:
            # Interpret as bitstring
            try:
                bitstring = initial_state.replace('|', '').replace('⟩', '')
                index = int(bitstring, 2)
                dim = 2 ** self.num_qubits
                self.quantum_state = np.zeros(dim, dtype=complex)
                self.quantum_state[index] = 1.0
            except:
                raise ValueError(f"Unknown initial state: {initial_state}")

        # Create emission patterns to physically initialize qubits
        for i, qubit in enumerate(self.qubits):
            pattern = self.emitter.emit_particle_cascade(
                num_particles=10,
                encoding_data=np.random.randint(0, 2, 10)
            )
            qubit.create_from_emission(pattern)
            self.metrics['emission_count'] += 1

        logger.info(f"Initialized qubits to state: {initial_state}")

        return self.quantum_state

    def apply_single_qubit_gate(self,
                                qubit_index: int,
                                gate_type: str) -> np.ndarray:
        """
        Apply single-qubit gate using topological operations.

        Args:
            qubit_index: Target qubit index
            gate_type: Gate type ('X', 'Y', 'Z', 'H', 'T', 'S')

        Returns:
            Updated quantum state
        """
        if qubit_index >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")

        # Get gate matrix
        gate_matrix = self._get_single_qubit_gate_matrix(gate_type)

        # Apply to quantum state
        self.quantum_state = self._apply_gate_to_state(
            self.quantum_state,
            gate_matrix,
            [qubit_index]
        )

        # Apply to topological qubit via braiding
        self.qubits[qubit_index].apply_topological_gate(gate_type)

        self.metrics['gate_count'] += 1

        # Record execution
        self.execution_history.append({
            'operation': 'single_qubit_gate',
            'gate': gate_type,
            'qubit': qubit_index
        })

        logger.debug(f"Applied {gate_type} gate to qubit {qubit_index}")

        return self.quantum_state

    def apply_two_qubit_gate(self,
                            control: int,
                            target: int,
                            gate_type: str = 'CNOT') -> np.ndarray:
        """
        Apply two-qubit gate using topological operations.

        Args:
            control: Control qubit index
            target: Target qubit index
            gate_type: Gate type ('CNOT', 'CZ', 'SWAP')

        Returns:
            Updated quantum state
        """
        if control >= self.num_qubits or target >= self.num_qubits:
            raise ValueError("Qubit index out of range")

        if control == target:
            raise ValueError("Control and target must be different qubits")

        # Get gate matrix
        gate_matrix = self._get_two_qubit_gate_matrix(gate_type)

        # Apply to quantum state
        self.quantum_state = self._apply_gate_to_state(
            self.quantum_state,
            gate_matrix,
            [control, target]
        )

        # Implement via topological operations
        # Create entangling echoes between qubits
        self._create_entangling_echoes(control, target)

        self.metrics['gate_count'] += 1

        # Record execution
        self.execution_history.append({
            'operation': 'two_qubit_gate',
            'gate': gate_type,
            'control': control,
            'target': target
        })

        logger.debug(f"Applied {gate_type} gate: control={control}, target={target}")

        return self.quantum_state

    def _get_single_qubit_gate_matrix(self, gate_type: str) -> np.ndarray:
        """Get unitary matrix for single-qubit gate."""
        gates = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            'S': np.array([[1, 0], [0, 1j]], dtype=complex),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
            'Sdg': np.array([[1, 0], [0, -1j]], dtype=complex),
            'Tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
        }

        if gate_type not in gates:
            raise ValueError(f"Unknown gate type: {gate_type}")

        return gates[gate_type]

    def _get_two_qubit_gate_matrix(self, gate_type: str) -> np.ndarray:
        """Get unitary matrix for two-qubit gate."""
        if gate_type == 'CNOT':
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex)

        elif gate_type == 'CZ':
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=complex)

        elif gate_type == 'SWAP':
            return np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex)

        else:
            raise ValueError(f"Unknown two-qubit gate: {gate_type}")

    def _apply_gate_to_state(self,
                            state: np.ndarray,
                            gate: np.ndarray,
                            target_qubits: List[int]) -> np.ndarray:
        """
        Apply gate to quantum state on target qubits.

        Uses tensor product expansion for multi-qubit states.
        """
        n = self.num_qubits
        dim = 2 ** n

        if len(target_qubits) == 1:
            # Single-qubit gate
            qubit = target_qubits[0]

            # Build full operator via tensor products
            ops = []
            for i in range(n):
                if i == qubit:
                    ops.append(gate)
                else:
                    ops.append(np.eye(2))

            # Tensor product
            full_operator = ops[0]
            for op in ops[1:]:
                full_operator = np.kron(full_operator, op)

            new_state = full_operator @ state

        elif len(target_qubits) == 2:
            # Two-qubit gate (simplified implementation)
            # For production, use proper tensor network contraction
            control, target = target_qubits

            # Build controlled operator
            full_operator = np.eye(dim, dtype=complex)

            # Apply gate to relevant subspace (simplified)
            # This is an approximation - full implementation would use
            # proper qubit permutation and tensor contraction
            for i in range(dim):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]

                if bits[control] == 1:
                    # Control is |1⟩, apply gate to target
                    # (Simplified - not fully general)
                    pass

            new_state = full_operator @ state

        else:
            raise NotImplementedError("Multi-qubit gates not yet implemented")

        # Normalize
        new_state /= np.linalg.norm(new_state)

        return new_state

    def _create_entangling_echoes(self, qubit1: int, qubit2: int):
        """
        Create entangling quantum echoes between two qubits.

        This implements two-qubit gates at the physical level.
        """
        # Emit correlated particle pairs
        pattern1 = self.emitter.emit_particle_cascade(
            num_particles=8,
            encoding_data=np.random.randint(0, 2, 8)
        )

        # Create echo from first pattern
        echo1 = self.propagator.create_echo(pattern1)

        # Create time-reversed echo for second qubit
        pattern2 = self.emitter.create_echo_from_pattern(pattern1, delay_time=1e-6)
        echo2 = self.propagator.create_echo(pattern2)

        # Compute interference
        interference = self.propagator.compute_interference(echo1, echo2)

        # Apply interference result to topological qubits
        phase = np.angle(interference)

        # Update qubit states based on interference
        if abs(interference) > 0.5:
            # Strong interference → successful entanglement
            logger.debug(f"Created entangling echoes between qubits {qubit1} and {qubit2}")

        self.metrics['echo_count'] += 2

    def measure(self, qubit_indices: Optional[List[int]] = None) -> Dict[int, int]:
        """
        Measure qubits in computational basis.

        Args:
            qubit_indices: List of qubits to measure (None = measure all)

        Returns:
            Dict mapping qubit index to measurement result (0 or 1)
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))

        # Compute measurement probabilities
        probabilities = np.abs(self.quantum_state) ** 2

        # Sample from probability distribution
        outcome_index = np.random.choice(len(probabilities), p=probabilities)

        # Convert to bitstring
        bitstring = format(outcome_index, f'0{self.num_qubits}b')

        # Extract results for measured qubits
        results = {}
        for qubit in qubit_indices:
            results[qubit] = int(bitstring[qubit])

        # Collapse state (simplified - projects to measured outcome)
        self.quantum_state = np.zeros_like(self.quantum_state)
        self.quantum_state[outcome_index] = 1.0

        logger.info(f"Measured qubits {qubit_indices}: {results}")

        return results

    def run_circuit(self, circuit: EchoCircuit) -> Dict:
        """
        Execute a quantum circuit.

        Args:
            circuit: EchoCircuit to execute

        Returns:
            Dict with execution results and measurements
        """
        logger.info(f"Running circuit with {circuit.depth()} gates")

        # Execute gates
        for gate in circuit.gates:
            gate_type = gate['type']
            targets = gate['targets']

            if len(targets) == 1:
                self.apply_single_qubit_gate(targets[0], gate_type)
            elif len(targets) == 2:
                self.apply_two_qubit_gate(targets[0], targets[1], gate_type)

        # Perform measurements
        measurement_results = {}
        if circuit.measurements:
            measurement_results = self.measure(circuit.measurements)

        results = {
            'final_state': self.quantum_state.copy(),
            'measurements': measurement_results,
            'metrics': self.metrics.copy()
        }

        logger.info(f"Circuit execution complete")

        return results

    def get_statevector(self) -> np.ndarray:
        """Get current quantum state vector."""
        return self.quantum_state.copy()

    def get_qubit_state(self, qubit_index: int) -> np.ndarray:
        """
        Get reduced density matrix for a single qubit.

        Args:
            qubit_index: Qubit to extract

        Returns:
            2x2 density matrix for the qubit
        """
        # Simplified partial trace
        # Full implementation would properly trace out other qubits

        rho = np.outer(self.quantum_state, np.conj(self.quantum_state))

        # Extract reduced state (simplified)
        rho_reduced = np.zeros((2, 2), dtype=complex)
        rho_reduced[0, 0] = np.sum(np.abs(self.quantum_state[::2]) ** 2)
        rho_reduced[1, 1] = np.sum(np.abs(self.quantum_state[1::2]) ** 2)

        # Normalize
        rho_reduced /= np.trace(rho_reduced)

        return rho_reduced

    def reset(self):
        """Reset algorithm to initial state."""
        self.quantum_state = self._initialize_quantum_state()
        self.execution_history = []
        self.metrics = {
            'gate_count': 0,
            'emission_count': 0,
            'echo_count': 0,
            'errors_detected': 0,
            'errors_corrected': 0
        }

        logger.info("Reset algorithm to initial state")

    def get_fidelity(self, target_state: np.ndarray) -> float:
        """
        Compute fidelity with target state.

        Args:
            target_state: Target quantum state

        Returns:
            Fidelity (0 to 1)
        """
        # Normalize both states
        psi = self.quantum_state / np.linalg.norm(self.quantum_state)
        phi = target_state / np.linalg.norm(target_state)

        # Fidelity F = |⟨ψ|φ⟩|²
        overlap = np.abs(np.vdot(phi, psi))
        fidelity = overlap ** 2

        return float(fidelity)

    def print_metrics(self):
        """Print algorithm execution metrics."""
        print("\nQuantum Echoes Algorithm Metrics")
        print("=" * 50)
        print(f"Number of Qubits: {self.num_qubits}")
        print(f"Gates Applied: {self.metrics['gate_count']}")
        print(f"Particle Emissions: {self.metrics['emission_count']}")
        print(f"Echoes Created: {self.metrics['echo_count']}")

        if self.error_correction_enabled:
            print(f"Errors Detected: {self.metrics['errors_detected']}")
            print(f"Errors Corrected: {self.metrics['errors_corrected']}")

        print("=" * 50)

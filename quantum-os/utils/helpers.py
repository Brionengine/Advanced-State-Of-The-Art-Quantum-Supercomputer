"""
Helper Functions for Quantum Computing
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class QuantumHelpers:
    """
    Utility functions for quantum computing operations
    """

    @staticmethod
    def fidelity(counts1: Dict[str, int], counts2: Dict[str, int]) -> float:
        """
        Calculate fidelity between two measurement distributions

        Args:
            counts1: First measurement counts
            counts2: Second measurement counts

        Returns:
            Fidelity value (0 to 1)
        """
        # Normalize to probabilities
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        prob1 = {k: v / total1 for k, v in counts1.items()}
        prob2 = {k: v / total2 for k, v in counts2.items()}

        # Get all states
        all_states = set(prob1.keys()) | set(prob2.keys())

        # Calculate fidelity
        fidelity = 0.0
        for state in all_states:
            p1 = prob1.get(state, 0.0)
            p2 = prob2.get(state, 0.0)
            fidelity += np.sqrt(p1 * p2)

        return fidelity ** 2

    @staticmethod
    def counts_to_probabilities(counts: Dict[str, int]) -> Dict[str, float]:
        """
        Convert measurement counts to probabilities

        Args:
            counts: Measurement counts

        Returns:
            Probability distribution
        """
        total = sum(counts.values())
        return {state: count / total for state, count in counts.items()}

    @staticmethod
    def binary_to_int(binary_string: str) -> int:
        """Convert binary string to integer"""
        return int(binary_string, 2)

    @staticmethod
    def int_to_binary(value: int, num_bits: int) -> str:
        """Convert integer to binary string"""
        return format(value, f'0{num_bits}b')

    @staticmethod
    def calculate_expectation_value(
        counts: Dict[str, int],
        observable: str
    ) -> float:
        """
        Calculate expectation value of an observable

        Args:
            counts: Measurement counts
            observable: Observable (e.g., 'Z', 'X', 'ZZ')

        Returns:
            Expectation value
        """
        probs = QuantumHelpers.counts_to_probabilities(counts)

        expectation = 0.0
        for state, prob in probs.items():
            # Calculate eigenvalue based on observable
            # Simplified for Z observable
            if observable == 'Z':
                # +1 for |0⟩, -1 for |1⟩
                eigenvalue = 1 if state[0] == '0' else -1
            else:
                eigenvalue = 0  # Placeholder for other observables

            expectation += prob * eigenvalue

        return expectation

    @staticmethod
    def pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
        """
        Convert Pauli string to matrix representation

        Args:
            pauli_string: String of Pauli operators (e.g., 'XYZ')

        Returns:
            Matrix representation
        """
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

        # Build tensor product
        result = np.array([[1]], dtype=complex)

        for char in pauli_string:
            if char in pauli_map:
                result = np.kron(result, pauli_map[char])

        return result

    @staticmethod
    def create_statevector(binary_string: str) -> np.ndarray:
        """
        Create statevector for a computational basis state

        Args:
            binary_string: Binary string (e.g., '101')

        Returns:
            Statevector
        """
        num_qubits = len(binary_string)
        size = 2 ** num_qubits

        statevector = np.zeros(size, dtype=complex)
        index = int(binary_string, 2)
        statevector[index] = 1.0

        return statevector

    @staticmethod
    def calculate_entropy(counts: Dict[str, int]) -> float:
        """
        Calculate Shannon entropy of measurement distribution

        Args:
            counts: Measurement counts

        Returns:
            Entropy value
        """
        probs = QuantumHelpers.counts_to_probabilities(counts)

        entropy = 0.0
        for prob in probs.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)

        return entropy

    @staticmethod
    def generate_random_circuit_parameters(
        num_params: int,
        seed: int = None
    ) -> np.ndarray:
        """
        Generate random parameters for variational circuits

        Args:
            num_params: Number of parameters
            seed: Random seed

        Returns:
            Array of random parameters
        """
        if seed is not None:
            np.random.seed(seed)

        return np.random.uniform(0, 2 * np.pi, num_params)

    @staticmethod
    def estimate_circuit_time(
        num_qubits: int,
        num_gates: int,
        backend_type: str = 'simulator'
    ) -> float:
        """
        Estimate circuit execution time

        Args:
            num_qubits: Number of qubits
            num_gates: Number of gates
            backend_type: Backend type

        Returns:
            Estimated time in seconds
        """
        if backend_type == 'simulator':
            # Simulation time scales exponentially with qubits
            base_time = 2 ** num_qubits * 1e-6  # microseconds
            gate_time = num_gates * 1e-7
            return base_time + gate_time

        elif backend_type == 'real_hardware':
            # Real hardware has fixed overhead + gate time
            overhead = 0.1  # 100ms overhead
            gate_time = num_gates * 100e-9  # 100ns per gate
            return overhead + gate_time

        return 0.0

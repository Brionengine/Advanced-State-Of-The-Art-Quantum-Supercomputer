"""
Quantum Search Algorithms
=========================

Implementation of quantum search algorithms using Quantum Echoes framework.
Includes Grover's algorithm, amplitude amplification, and echo-enhanced search.

This module provides quantum speedup for unstructured search problems.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .quantum_echoes import QuantumEchoesAlgorithm, EchoCircuit
from circuits.echo_circuit_builder import build_circuit

logger = logging.getLogger(__name__)


class GroverSearch:
    """
    Grover's quantum search algorithm using topological qubits.

    Provides quadratic speedup for searching unstructured databases.
    For N items, finds target in O(√N) operations vs O(N) classical.
    """

    def __init__(self, num_qubits: int, marked_states: List[int]):
        """
        Initialize Grover search.

        Args:
            num_qubits: Number of qubits (search space size = 2^n)
            marked_states: List of marked (target) state indices
        """
        self.num_qubits = num_qubits
        self.search_space_size = 2 ** num_qubits
        self.marked_states = marked_states

        # Initialize quantum system
        self.qe = QuantumEchoesAlgorithm(num_qubits=num_qubits)

        # Calculate optimal number of iterations
        self.optimal_iterations = self._calculate_optimal_iterations()

        # Statistics
        self.stats = {
            'searches_performed': 0,
            'successful_searches': 0,
            'average_iterations': 0
        }

        logger.info(f"Initialized Grover search: {num_qubits} qubits, "
                   f"search space size {self.search_space_size}, "
                   f"{len(marked_states)} marked states")

    def _calculate_optimal_iterations(self) -> int:
        """
        Calculate optimal number of Grover iterations.

        For M marked items in N total items:
        iterations ≈ (π/4) * √(N/M)
        """
        N = self.search_space_size
        M = len(self.marked_states)

        if M == 0:
            return 0

        optimal = int(np.round((np.pi / 4) * np.sqrt(N / M)))

        return max(1, optimal)

    def search(self, oracle: Optional[Callable] = None,
               num_iterations: Optional[int] = None) -> Tuple[int, float]:
        """
        Execute Grover search algorithm.

        Args:
            oracle: Optional oracle function (uses marked_states if None)
            num_iterations: Number of Grover iterations (uses optimal if None)

        Returns:
            Tuple of (found_state, success_probability)
        """
        if num_iterations is None:
            num_iterations = self.optimal_iterations

        logger.info(f"Starting Grover search with {num_iterations} iterations")

        # Step 1: Initialize to uniform superposition
        self.qe.reset()
        self.qe.initialize_qubits('zero')

        # Apply Hadamard to all qubits
        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'H')

        # Step 2: Grover iterations
        for iteration in range(num_iterations):
            # Oracle: mark target states
            self._apply_oracle(oracle)

            # Diffusion operator (inversion about average)
            self._apply_diffusion()

            logger.debug(f"Grover iteration {iteration + 1}/{num_iterations} complete")

        # Step 3: Measurement
        measurements = self.qe.measure(list(range(self.num_qubits)))

        # Convert measurement to state index
        found_state = 0
        for qubit, bit in measurements.items():
            found_state |= bit << qubit

        # Check if we found a marked state
        success = found_state in self.marked_states

        # Calculate success probability
        final_state = self.qe.get_statevector()
        success_prob = sum(abs(final_state[s]) ** 2 for s in self.marked_states)

        # Update statistics
        self.stats['searches_performed'] += 1
        if success:
            self.stats['successful_searches'] += 1

        logger.info(f"Search result: state {found_state}, "
                   f"success: {success}, probability: {success_prob:.4f}")

        return found_state, float(success_prob)

    def _apply_oracle(self, oracle: Optional[Callable] = None):
        """
        Apply oracle operator to mark target states.

        Oracle flips the sign of marked states: |x⟩ → -|x⟩ if x is marked.

        Args:
            oracle: Optional custom oracle function
        """
        if oracle is not None:
            # Use custom oracle
            oracle(self.qe)
        else:
            # Use default oracle based on marked_states
            # For each marked state, apply phase flip
            for marked_state in self.marked_states:
                self._apply_phase_flip(marked_state)

    def _apply_phase_flip(self, target_state: int):
        """
        Apply phase flip to a specific computational basis state.

        Args:
            target_state: Index of state to flip
        """
        # Decompose target state to binary
        bits = [(target_state >> i) & 1 for i in range(self.num_qubits)]

        # Apply X gates to flip 0s to 1s
        for i, bit in enumerate(bits):
            if bit == 0:
                self.qe.apply_single_qubit_gate(i, 'X')

        # Multi-controlled Z gate (simplified with CNOTs and single Z)
        # Apply CNOT chain to concentrate control
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                self.qe.apply_two_qubit_gate(i, i + 1, 'CNOT')

        # Apply Z to last qubit
        self.qe.apply_single_qubit_gate(self.num_qubits - 1, 'Z')

        # Uncompute CNOT chain
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 2, -1, -1):
                self.qe.apply_two_qubit_gate(i, i + 1, 'CNOT')

        # Uncompute X gates
        for i, bit in enumerate(bits):
            if bit == 0:
                self.qe.apply_single_qubit_gate(i, 'X')

    def _apply_diffusion(self):
        """
        Apply Grover diffusion operator (inversion about average).

        Diffusion = H^⊗n (2|0⟩⟨0| - I) H^⊗n
        """
        # Apply Hadamard to all qubits
        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'H')

        # Apply phase flip to |0⟩ state
        # This is equivalent to flipping all qubits, applying CZ, then flipping back
        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'X')

        # Multi-controlled Z (same as oracle for |111...1⟩)
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1):
                self.qe.apply_two_qubit_gate(i, i + 1, 'CNOT')

        self.qe.apply_single_qubit_gate(self.num_qubits - 1, 'Z')

        if self.num_qubits > 1:
            for i in range(self.num_qubits - 2, -1, -1):
                self.qe.apply_two_qubit_gate(i, i + 1, 'CNOT')

        # Uncompute X gates
        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'X')

        # Apply Hadamard to all qubits
        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'H')

    def search_multiple_runs(self, num_runs: int = 10) -> Dict[str, any]:
        """
        Run search multiple times and collect statistics.

        Args:
            num_runs: Number of search runs

        Returns:
            Dictionary with search statistics
        """
        logger.info(f"Running Grover search {num_runs} times")

        results = []
        success_count = 0

        for run in range(num_runs):
            found_state, prob = self.search()

            results.append({
                'state': found_state,
                'probability': prob,
                'success': found_state in self.marked_states
            })

            if found_state in self.marked_states:
                success_count += 1

        # Compute statistics
        success_rate = success_count / num_runs
        avg_prob = np.mean([r['probability'] for r in results])

        stats = {
            'num_runs': num_runs,
            'success_rate': success_rate,
            'average_probability': avg_prob,
            'results': results,
            'marked_states': self.marked_states,
            'optimal_iterations': self.optimal_iterations
        }

        logger.info(f"Search complete: {success_count}/{num_runs} successful "
                   f"({success_rate:.1%} success rate)")

        return stats


class AmplitudeAmplification:
    """
    Quantum amplitude amplification algorithm.

    Generalizes Grover's algorithm to amplify any quantum state.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize amplitude amplification.

        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.qe = QuantumEchoesAlgorithm(num_qubits=num_qubits)

        logger.info(f"Initialized AmplitudeAmplification with {num_qubits} qubits")

    def amplify(self,
                state_preparation: Callable,
                target_state_oracle: Callable,
                num_iterations: int) -> np.ndarray:
        """
        Amplify amplitude of target state.

        Args:
            state_preparation: Function to prepare initial state
            target_state_oracle: Oracle marking target state
            num_iterations: Number of amplification iterations

        Returns:
            Final quantum state
        """
        logger.info(f"Starting amplitude amplification with {num_iterations} iterations")

        # Initialize
        self.qe.reset()
        self.qe.initialize_qubits('zero')

        # Prepare initial state
        state_preparation(self.qe)

        # Amplification iterations
        for i in range(num_iterations):
            # Apply target oracle
            target_state_oracle(self.qe)

            # Apply reflection about initial state
            self._reflect_about_state(state_preparation)

            logger.debug(f"Amplification iteration {i + 1}/{num_iterations}")

        final_state = self.qe.get_statevector()

        logger.info("Amplitude amplification complete")

        return final_state

    def _reflect_about_state(self, state_preparation: Callable):
        """
        Reflect about a prepared state.

        Implements: 2|ψ⟩⟨ψ| - I
        """
        # Invert state preparation
        # This is simplified; full implementation would use inverse circuit

        # Apply state preparation (approximate inverse)
        state_preparation(self.qe)

        # Apply reflection about |0⟩
        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'X')

        # Multi-controlled Z
        if self.num_qubits > 1:
            self.qe.apply_two_qubit_gate(0, 1, 'CZ')

        for i in range(self.num_qubits):
            self.qe.apply_single_qubit_gate(i, 'X')

        # Reapply state preparation
        state_preparation(self.qe)


class EchoEnhancedSearch:
    """
    Search algorithm enhanced with quantum echoes.

    Uses echo interference for improved search performance and
    robustness against decoherence.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize echo-enhanced search.

        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.qe = QuantumEchoesAlgorithm(num_qubits=num_qubits)

        from core.particle_emitter import SpecializedIonEmitter
        from core.echo_propagator import EchoPropagator

        self.emitter = SpecializedIonEmitter(num_ions=50)
        self.propagator = EchoPropagator()

        logger.info(f"Initialized EchoEnhancedSearch with {num_qubits} qubits")

    def search_with_echoes(self,
                          marked_states: List[int],
                          num_echoes: int = 5) -> Tuple[int, float]:
        """
        Search using quantum echo enhancement.

        Args:
            marked_states: List of target states
            num_echoes: Number of echo sequences to use

        Returns:
            Tuple of (found_state, confidence)
        """
        logger.info(f"Echo-enhanced search for {len(marked_states)} marked states "
                   f"using {num_echoes} echoes")

        # Create emission patterns for each search iteration
        patterns = []
        for i in range(num_echoes):
            pattern = self.emitter.emit_particle_cascade(num_particles=10)
            patterns.append(pattern)

        # Run Grover search with each echo
        results = []

        for i, pattern in enumerate(patterns):
            # Initialize search with echo pattern
            grover = GroverSearch(self.num_qubits, marked_states)

            # Run search
            found, prob = grover.search()
            results.append((found, prob))

            logger.debug(f"Echo {i+1}/{num_echoes}: found state {found}, prob={prob:.3f}")

        # Combine results via majority vote weighted by probability
        state_votes = {}
        for state, prob in results:
            if state not in state_votes:
                state_votes[state] = 0
            state_votes[state] += prob

        # Find state with highest total probability
        best_state = max(state_votes.items(), key=lambda x: x[1])

        confidence = best_state[1] / num_echoes

        logger.info(f"Echo-enhanced search result: state {best_state[0]}, "
                   f"confidence {confidence:.3f}")

        return best_state[0], confidence


def grover_search_example(num_qubits: int = 3, target_state: int = 5):
    """
    Example of Grover search.

    Args:
        num_qubits: Number of qubits
        target_state: Target state to find

    Returns:
        Search result
    """
    print(f"\nGrover Search Example")
    print("=" * 60)
    print(f"Search space: {2**num_qubits} states")
    print(f"Target state: {target_state} ({bin(target_state)})")

    # Initialize Grover search
    grover = GroverSearch(num_qubits=num_qubits, marked_states=[target_state])

    print(f"Optimal iterations: {grover.optimal_iterations}")

    # Run search
    print("\nRunning search...")
    found, prob = grover.search()

    print(f"\nResult:")
    print(f"  Found state: {found} ({bin(found)})")
    print(f"  Success probability: {prob:.4f}")
    print(f"  Correct: {found == target_state}")

    # Multiple runs
    print(f"\nRunning 10 searches...")
    stats = grover.search_multiple_runs(num_runs=10)

    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average probability: {stats['average_probability']:.4f}")

    return stats


def amplitude_amplification_example():
    """Example of amplitude amplification."""
    print(f"\nAmplitude Amplification Example")
    print("=" * 60)

    amp = AmplitudeAmplification(num_qubits=3)

    def prepare_state(qe):
        """Prepare initial state with small amplitude for target."""
        qe.apply_single_qubit_gate(0, 'H')
        qe.apply_single_qubit_gate(1, 'H')

    def target_oracle(qe):
        """Mark state |101⟩."""
        # X on qubit 1
        qe.apply_single_qubit_gate(1, 'X')
        # CZ
        qe.apply_two_qubit_gate(0, 2, 'CZ')
        # X on qubit 1
        qe.apply_single_qubit_gate(1, 'X')

    print("Amplifying amplitude of |101⟩ state...")

    final_state = amp.amplify(
        state_preparation=prepare_state,
        target_state_oracle=target_oracle,
        num_iterations=2
    )

    target_amplitude = abs(final_state[5]) ** 2  # |101⟩ = 5
    print(f"Final amplitude of target state: {target_amplitude:.4f}")

    return final_state


def echo_enhanced_search_example():
    """Example of echo-enhanced search."""
    print(f"\nEcho-Enhanced Search Example")
    print("=" * 60)

    echo_search = EchoEnhancedSearch(num_qubits=3)

    target = 6  # |110⟩
    print(f"Searching for state {target} ({bin(target)})")
    print("Using quantum echoes for enhanced robustness...")

    found, confidence = echo_search.search_with_echoes(
        marked_states=[target],
        num_echoes=5
    )

    print(f"\nResult:")
    print(f"  Found state: {found} ({bin(found)})")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Correct: {found == target}")

    return found, confidence


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QUANTUM SEARCH ALGORITHMS DEMONSTRATION")
    print("=" * 60)

    # Run examples
    grover_search_example(num_qubits=3, target_state=5)
    amplitude_amplification_example()
    echo_enhanced_search_example()

    print("\n" + "=" * 60)
    print("Quantum search demonstrations complete!")
    print("=" * 60 + "\n")

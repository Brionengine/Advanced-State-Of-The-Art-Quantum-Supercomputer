"""
Quantum Circuit and Result Visualization
"""

from typing import Dict, Any, Optional
import numpy as np


class QuantumVisualizer:
    """
    Visualization utilities for quantum computing results
    """

    @staticmethod
    def print_counts(counts: Dict[str, int], top_n: int = 10):
        """
        Print measurement counts in a formatted way

        Args:
            counts: Measurement counts
            top_n: Number of top results to show
        """
        total = sum(counts.values())

        print(f"\nMeasurement Results ({total} shots):")
        print("=" * 50)

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        for i, (state, count) in enumerate(sorted_counts[:top_n]):
            probability = count / total * 100
            bar_length = int(probability / 2)
            bar = "█" * bar_length

            print(f"|{state}⟩: {count:5d} ({probability:5.2f}%) {bar}")

        if len(sorted_counts) > top_n:
            print(f"... and {len(sorted_counts) - top_n} more states")

    @staticmethod
    def print_statevector(statevector: np.ndarray, threshold: float = 0.01):
        """
        Print statevector in ket notation

        Args:
            statevector: Quantum statevector
            threshold: Amplitude threshold for display
        """
        num_qubits = int(np.log2(len(statevector)))

        print(f"\nStatevector ({num_qubits} qubits):")
        print("=" * 50)

        for i, amplitude in enumerate(statevector):
            if np.abs(amplitude) > threshold:
                state = format(i, f'0{num_qubits}b')
                real = amplitude.real
                imag = amplitude.imag

                if abs(imag) < 1e-10:
                    print(f"|{state}⟩: {real:.4f}")
                else:
                    sign = '+' if imag >= 0 else '-'
                    print(f"|{state}⟩: {real:.4f} {sign} {abs(imag):.4f}i")

    @staticmethod
    def print_circuit_info(circuit_analysis: Dict[str, Any]):
        """
        Print circuit information

        Args:
            circuit_analysis: Circuit analysis dictionary
        """
        print("\nCircuit Information:")
        print("=" * 50)
        print(f"Number of qubits: {circuit_analysis.get('num_qubits', 0)}")
        print(f"Circuit depth: {circuit_analysis.get('depth', 0)}")
        print(f"Total gates: {circuit_analysis.get('gate_count', 0)}")
        print(f"Two-qubit gates: {circuit_analysis.get('two_qubit_gates', 0)}")

    @staticmethod
    def print_backend_status(backend_props: Dict[str, Any]):
        """
        Print backend status information

        Args:
            backend_props: Backend properties dictionary
        """
        print("\nBackend Status:")
        print("=" * 50)
        print(f"Name: {backend_props.get('backend_name', 'Unknown')}")
        print(f"Type: {backend_props.get('backend_type', 'Unknown')}")
        print(f"Mode: {backend_props.get('execution_mode', 'Unknown')}")
        print(f"Available: {backend_props.get('is_available', False)}")

        if 'num_qubits' in backend_props:
            print(f"Qubits: {backend_props['num_qubits']}")

        if 'avg_gate_error' in backend_props:
            print(f"Avg gate error: {backend_props['avg_gate_error']:.4f}")

        if 'avg_t1' in backend_props:
            print(f"Avg T1 time: {backend_props['avg_t1']*1e6:.2f} μs")

    @staticmethod
    def create_ascii_histogram(counts: Dict[str, int], max_width: int = 50):
        """
        Create ASCII histogram of measurement results

        Args:
            counts: Measurement counts
            max_width: Maximum bar width

        Returns:
            ASCII histogram string
        """
        if not counts:
            return "No data to display"

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_counts[0][1]

        lines = []
        lines.append("\nMeasurement Histogram:")
        lines.append("=" * (max_width + 15))

        for state, count in sorted_counts:
            bar_length = int((count / max_count) * max_width)
            bar = "█" * bar_length
            lines.append(f"|{state}⟩ {bar} {count}")

        return "\n".join(lines)

    @staticmethod
    def compare_results(
        counts1: Dict[str, int],
        counts2: Dict[str, int],
        label1: str = "Result 1",
        label2: str = "Result 2"
    ):
        """
        Compare two sets of measurement results

        Args:
            counts1: First measurement counts
            counts2: Second measurement counts
            label1: Label for first result
            label2: Label for second result
        """
        print(f"\nComparing Results:")
        print("=" * 70)

        # Get all states
        all_states = sorted(set(counts1.keys()) | set(counts2.keys()))

        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        print(f"{'State':<10} {label1:<20} {label2:<20} {'Diff':<10}")
        print("-" * 70)

        for state in all_states:
            count1 = counts1.get(state, 0)
            count2 = counts2.get(state, 0)

            prob1 = count1 / total1 * 100 if total1 > 0 else 0
            prob2 = count2 / total2 * 100 if total2 > 0 else 0

            diff = prob1 - prob2

            print(f"|{state}⟩ {prob1:6.2f}% {prob2:6.2f}% {diff:+7.2f}%")

#!/usr/bin/env python3
"""
Bell State Creation Example
============================

Simple example demonstrating Bell state creation using Quantum Echoes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.quantum_echoes import QuantumEchoesAlgorithm
from circuits.echo_circuit_builder import build_circuit
import numpy as np


def main():
    print("\n" + "=" * 60)
    print("BELL STATE CREATION EXAMPLE")
    print("=" * 60 + "\n")

    # Create 2-qubit system
    print("1. Initializing 2-qubit system...")
    qe = QuantumEchoesAlgorithm(num_qubits=2, error_correction=True)

    # Initialize to |00⟩
    print("2. Initializing qubits to |00⟩...")
    qe.initialize_qubits('zero')

    # Build Bell state circuit: H(0) → CNOT(0,1)
    print("3. Building Bell state circuit...")
    circuit = build_circuit(2)
    circuit.h(0)  # Hadamard on qubit 0
    circuit.cnot(0, 1)  # CNOT with control=0, target=1
    circuit.measure_all()

    print("   Circuit:")
    print("   |0⟩ ─H─●─M")
    print("   |0⟩ ───X─M")

    # Execute circuit
    print("\n4. Executing circuit...")
    results = qe.run_circuit(circuit.build())

    # Display results
    print("\n5. Results:")
    print(f"   Measurements: {results['measurements']}")
    print(f"   Gates executed: {results['metrics']['gate_count']}")
    print(f"   Particle emissions: {results['metrics']['emission_count']}")

    # Check Bell state fidelity
    ideal_bell = np.zeros(4, dtype=complex)
    ideal_bell[0] = 1/np.sqrt(2)  # |00⟩
    ideal_bell[3] = 1/np.sqrt(2)  # |11⟩

    fidelity = qe.get_fidelity(ideal_bell)
    print(f"\n   State fidelity with |Φ+⟩: {fidelity:.4f}")

    # Multiple measurements to see statistics
    print("\n6. Running 10 measurements to see Bell state statistics...")
    outcomes = {'00': 0, '11': 0, 'other': 0}

    for i in range(10):
        qe.initialize_qubits('zero')
        results = qe.run_circuit(circuit.build())
        meas = results['measurements']

        outcome = f"{meas[0]}{meas[1]}"
        if outcome in outcomes:
            outcomes[outcome] += 1
        else:
            outcomes['other'] += 1

    print(f"\n   Outcome statistics:")
    print(f"   |00⟩: {outcomes['00']}/10")
    print(f"   |11⟩: {outcomes['11']}/10")
    print(f"   Other: {outcomes['other']}/10")

    print("\n" + "=" * 60)
    print("Bell state example complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

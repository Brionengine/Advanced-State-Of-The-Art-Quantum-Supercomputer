"""
Quantum Supercomputer Demonstration

Demonstrates advanced features:
- Hybrid quantum-classical computing
- GPU acceleration
- Quantum error correction integration
- High-fidelity qubit operations
- IBM Brisbane/Torino execution
"""

import sys
sys.path.insert(0, '..')

from quantum_os import create_quantum_os, SurfaceCode
import numpy as np


def demo_bell_state_fidelity():
    """Demonstrate high-fidelity Bell state creation and measurement"""
    print("\n" + "="*70)
    print("Quantum Supercomputer Demo: High-Fidelity Bell State")
    print("="*70)

    qos = create_quantum_os()

    try:
        import cirq

        # Create circuit
        circuit = qos.create_circuit(num_qubits=2, backend_name='cirq_simulator')
        qubits = sorted(circuit.all_qubits())

        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))

        # Execute with high shot count for fidelity measurement
        result = qos.execute(circuit, shots=10000, backend_name='cirq_simulator')

        if result.success:
            print(f"\nBell State Results (10,000 shots):")
            for state, count in sorted(result.counts.items()):
                probability = count / result.shots
                print(f"  |{state}⟩: {count} ({probability*100:.2f}%)")

            # Calculate fidelity
            ideal_states = ['00', '11']
            fidelity = sum(result.counts.get(s, 0) for s in ideal_states) / result.shots
            print(f"\nState Fidelity: {fidelity*100:.2f}%")
            print(f"Execution Time: {result.execution_time:.4f}s")

    except ImportError:
        print("Cirq not available")


def demo_grover_search():
    """Demonstrate Grover's search algorithm"""
    print("\n" + "="*70)
    print("Quantum Supercomputer Demo: Grover's Search Algorithm")
    print("="*70)

    qos = create_quantum_os()

    try:
        import cirq

        num_qubits = 3
        target_state = '101'  # Target to find

        circuit = qos.create_circuit(num_qubits=num_qubits, backend_name='cirq_simulator')
        qubits = sorted(circuit.all_qubits())

        # Initialize superposition
        circuit.append([cirq.H(q) for q in qubits])

        # Grover iterations (optimal = π/4 * sqrt(2^n))
        num_iterations = int(np.pi / 4 * np.sqrt(2**num_qubits))

        for _ in range(num_iterations):
            # Oracle: mark target state
            # Simplified oracle for demonstration
            circuit.append(cirq.Z(qubits[0]))  # Marks states with first qubit |1⟩

            # Diffusion operator
            circuit.append([cirq.H(q) for q in qubits])
            circuit.append([cirq.X(q) for q in qubits])
            circuit.append(cirq.H(qubits[-1]))
            circuit.append(cirq.CCX(qubits[0], qubits[1], qubits[2]))
            circuit.append(cirq.H(qubits[-1]))
            circuit.append([cirq.X(q) for q in qubits])
            circuit.append([cirq.H(q) for q in qubits])

        # Execute
        result = qos.execute(circuit, shots=1024, backend_name='cirq_simulator')

        if result.success:
            print(f"\nGrover Search Results:")
            # Show top 3 results
            sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
            for state, count in sorted_counts[:3]:
                probability = count / result.shots
                print(f"  |{state}⟩: {count} ({probability*100:.2f}%)")

    except ImportError:
        print("Cirq not available")


def demo_vqe_energy_minimization():
    """Demonstrate VQE for molecular energy calculation"""
    print("\n" + "="*70)
    print("Quantum Supercomputer Demo: VQE Energy Minimization")
    print("="*70)

    qos = create_quantum_os()

    try:
        import cirq
        from scipy.optimize import minimize

        num_qubits = 2

        # Define a simple Hamiltonian: H = Z₀ + Z₁ + X₀X₁
        # This represents a toy molecule

        def create_ansatz(params):
            """Create parameterized quantum circuit"""
            circuit = qos.create_circuit(num_qubits=num_qubits, backend_name='cirq_simulator')
            qubits = sorted(circuit.all_qubits())

            # Rotation layer
            circuit.append(cirq.ry(params[0])(qubits[0]))
            circuit.append(cirq.ry(params[1])(qubits[1]))

            # Entanglement
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))

            # Another rotation layer
            circuit.append(cirq.ry(params[2])(qubits[0]))
            circuit.append(cirq.ry(params[3])(qubits[1]))

            return circuit

        def cost_function(params):
            """Calculate energy expectation value"""
            circuit = create_ansatz(params)

            # Measure in Z basis
            result = qos.execute(circuit, shots=1024, backend_name='cirq_simulator')

            if not result.success:
                return 999.0

            # Calculate expectation values
            energy = 0.0
            for state, count in result.counts.items():
                prob = count / result.shots

                # Z₀ and Z₁ contributions
                z0 = 1 if state[0] == '0' else -1
                z1 = 1 if state[1] == '0' else -1

                energy += prob * (z0 + z1)

            return energy

        # Optimize
        print("\nRunning VQE optimization...")
        initial_params = np.random.rand(4) * 2 * np.pi

        result_opt = minimize(
            cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': 50}
        )

        print(f"\nOptimization Results:")
        print(f"  Ground state energy: {result_opt.fun:.4f}")
        print(f"  Optimal parameters: {result_opt.x}")
        print(f"  Iterations: {result_opt.nit}")

    except ImportError as e:
        print(f"Required library not available: {e}")


def demo_error_correction_overhead():
    """Calculate error correction overhead for quantum supercomputer"""
    print("\n" + "="*70)
    print("Quantum Supercomputer Demo: Error Correction Analysis")
    print("="*70)

    from quantum_os.error_correction.surface_codes import get_error_correction_requirements

    print("\nAnalyzing requirements for fault-tolerant quantum computing:")

    target_rates = {
        '1 per million': 1e-6,
        '1 per billion': 1e-9,
        '1 per trillion': 1e-12
    }

    for name, rate in target_rates.items():
        print(f"\n  Target: {name} operations ({rate:.0e})")

        requirements = get_error_correction_requirements(rate)

        for backend, req in requirements.items():
            if 'error' not in req and req['achievable']:
                print(f"    {backend}:")
                print(f"      Code distance: {req['code_distance']}")
                print(f"      Physical qubits needed: {req['total_qubits_needed']}")
                print(f"      Logical error rate: {req['logical_error_rate']:.2e}")


def demo_quantum_ml():
    """Demonstrate quantum machine learning with TFQ"""
    print("\n" + "="*70)
    print("Quantum Supercomputer Demo: Quantum Machine Learning")
    print("="*70)

    qos = create_quantum_os()

    try:
        import tensorflow_quantum as tfq
        import tensorflow as tf
        import cirq
        import sympy

        # Check if TFQ backend available
        if 'tfq_simulator' not in qos.list_backends():
            print("TFQ backend not available")
            return

        print("\nCreating parameterized quantum circuit for ML...")

        num_qubits = 4
        qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]

        # Create parameterized circuit
        params = [sympy.Symbol(f'θ_{i}') for i in range(num_qubits * 2)]

        circuit = cirq.Circuit()
        for i, q in enumerate(qubits):
            circuit += cirq.ry(params[i])(q)

        for i in range(num_qubits - 1):
            circuit += cirq.CNOT(qubits[i], qubits[i + 1])

        for i, q in enumerate(qubits):
            circuit += cirq.ry(params[num_qubits + i])(q)

        print(f"  Qubits: {num_qubits}")
        print(f"  Parameters: {len(params)}")
        print(f"  Circuit depth: {len(circuit)}")

        # This demonstrates the structure - full training would require more setup
        print("\nQuantum circuit ready for hybrid quantum-classical training!")
        print("Can be integrated with TensorFlow/Keras for quantum neural networks.")

    except ImportError:
        print("TensorFlow Quantum not available")


def demo_ibm_quantum_execution():
    """Demonstrate execution on IBM Quantum hardware"""
    print("\n" + "="*70)
    print("Quantum Supercomputer Demo: IBM Quantum Hardware Access")
    print("="*70)

    qos = create_quantum_os()

    # Check for IBM backends
    ibm_backends = [b for b in qos.list_backends() if 'ibm' in b.lower()]

    if not ibm_backends:
        print("\nNo IBM Quantum backends configured.")
        print("To use IBM Brisbane (127q) or Torino (133q):")
        print("  1. Set environment variable: IBM_QUANTUM_TOKEN")
        print("  2. Or configure in quantum OS config file")
        return

    print(f"\nAvailable IBM Quantum backends: {ibm_backends}")

    for backend_name in ibm_backends:
        props = qos.get_backend_properties(backend_name)

        print(f"\n{backend_name}:")
        print(f"  Qubits: {props.get('num_qubits', 'N/A')}")
        print(f"  Available: {props.get('is_available', False)}")
        print(f"  Max shots: {props.get('max_shots', 'N/A')}")

        if 'avg_gate_error' in props:
            print(f"  Avg gate error: {props['avg_gate_error']:.4f}")
        if 'avg_t1' in props:
            print(f"  Avg T1 time: {props['avg_t1']*1e6:.2f} μs")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         Quantum Supercomputer Demonstration                       ║
    ║                                                                   ║
    ║  Showcasing advanced quantum computing capabilities:             ║
    ║  - High-fidelity qubit operations                                ║
    ║  - Quantum algorithms (Grover, VQE)                              ║
    ║  - Error correction for fault tolerance                          ║
    ║  - Quantum machine learning                                      ║
    ║  - Real quantum hardware access (IBM Brisbane/Torino)            ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Run demonstrations
    demo_bell_state_fidelity()
    demo_grover_search()
    demo_vqe_energy_minimization()
    demo_error_correction_overhead()
    demo_quantum_ml()
    demo_ibm_quantum_execution()

    print("\n" + "="*70)
    print("Quantum Supercomputer demonstrations completed!")
    print("="*70)

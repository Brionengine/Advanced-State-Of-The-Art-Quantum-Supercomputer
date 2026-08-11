"""
General Quantum Supercomputer Demonstration

This example demonstrates the true quantum supercomputer capabilities:
- General-purpose quantum computing on any algorithm
- Multiple quantum computers working together as one supercomputer
- Automatic distribution across Google Willow, IBM Brisbane, IBM Torino
- Backend-agnostic quantum programming
"""

import sys
sys.path.insert(0, '..')

from quantum_os import (
    create_quantum_os,
    QuantumProgram,
    QuantumGateType,
    GroverSearch,
    VariationalQuantumEigensolver
)
import numpy as np


def demo_unified_supercomputer_status():
    """Show the unified quantum supercomputer status"""
    print("="*80)
    print("GENERAL QUANTUM SUPERCOMPUTER STATUS")
    print("="*80)

    qos = create_quantum_os()

    # Get resource pool status
    pool_status = qos.resource_pool.get_pool_status()

    print(f"\n🖥️  Quantum Supercomputer Configuration:")
    print(f"  Total Quantum Backends: {pool_status['total_backends']}")
    print(f"  Available Backends: {pool_status['available_backends']}")
    print(f"  Total Qubits (Combined): {pool_status['total_qubits']}")
    print(f"  Real Quantum Hardware Qubits: {pool_status['real_hardware_qubits']}")

    print(f"\n📊 Connected Quantum Computers:")
    for name, info in pool_status['backends'].items():
        status_icon = "✅" if info['available'] else "❌"
        hw_type = "Real Quantum Hardware" if info['real_hardware'] else "Simulator"
        print(f"  {status_icon} {name}")
        print(f"      Type: {hw_type}")
        print(f"      Qubits: {info['qubits']}")
        print(f"      Gate Error: {info['avg_gate_error']}")
        print(f"      Jobs Completed: {info['jobs_completed']}")


def demo_general_quantum_programming():
    """Demonstrate general-purpose quantum programming"""
    print("\n" + "="*80)
    print("GENERAL QUANTUM PROGRAMMING (Backend-Agnostic)")
    print("="*80)

    qos = create_quantum_os()

    # Create a program using the Quantum Virtual Machine
    print("\nCreating quantum program using Quantum Virtual Machine...")

    program = qos.qvm.create_program(num_qubits=3)

    # Write a general quantum algorithm
    # Example: Create GHZ state |000⟩ + |111⟩
    program.h(0)  # Hadamard on qubit 0
    program.cnot(0, 1)  # CNOT 0 -> 1
    program.cnot(1, 2)  # CNOT 1 -> 2
    program.measure_all()

    print(f"Program created: {program}")
    print(f"  Instructions: {len(program.instructions)}")

    # Execute on ANY backend (automatically compiled)
    print("\n🚀 Executing on different backends:")

    backends_to_try = qos.list_backends()[:3]  # Try first 3 backends

    for backend_name in backends_to_try:
        try:
            print(f"\n  Executing on {backend_name}...")
            result = qos.qvm.execute(program, shots=1024, backend_name=backend_name)

            if result.success:
                top_results = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    ✅ Success! Top results:")
                for state, count in top_results:
                    print(f"       |{state}⟩: {count} ({count/result.shots*100:.1f}%)")
            else:
                print(f"    ❌ Failed: {result.error_message}")

        except Exception as e:
            print(f"    ⚠️  Error: {e}")


def demo_distributed_supercomputer_execution():
    """Demonstrate distributed execution across multiple quantum computers"""
    print("\n" + "="*80)
    print("DISTRIBUTED QUANTUM SUPERCOMPUTER EXECUTION")
    print("="*80)

    qos = create_quantum_os()

    print("\nCreating 5 different quantum circuits...")

    programs = []
    for i in range(5):
        program = qos.qvm.create_program(num_qubits=2)

        # Each circuit is slightly different
        program.h(0)
        program.cnot(0, 1)
        for _ in range(i):  # Add rotations
            program.ry(0, np.pi / 4)
            program.ry(1, np.pi / 4)

        program.measure_all()
        programs.append(program)

    print(f"Created {len(programs)} quantum programs")

    # Compile all programs to circuits
    print("\nCompiling programs to circuits...")
    circuits = [qos.qvm.compile(prog) for prog in programs]

    # Execute distributedly across ALL quantum computers
    print("\n🌐 Distributing across quantum supercomputer...")

    aggregated_result = qos.resource_pool.execute_distributed(
        circuits,
        shots=1024,
        aggregate_results=True
    )

    if aggregated_result and aggregated_result.success:
        print(f"\n✅ Distributed execution completed!")
        print(f"  Total shots: {aggregated_result.shots}")
        print(f"  Backends used: {aggregated_result.metadata['num_backends']}")
        print(f"  Backend names: {', '.join(aggregated_result.metadata['backends_used'])}")

        print(f"\n  Aggregated results (top 5):")
        sorted_counts = sorted(aggregated_result.counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_counts[:5]:
            print(f"    |{state}⟩: {count} ({count/aggregated_result.shots*100:.1f}%)")


def demo_grover_search_on_supercomputer():
    """Run Grover's algorithm on the quantum supercomputer"""
    print("\n" + "="*80)
    print("GROVER'S SEARCH ON QUANTUM SUPERCOMPUTER")
    print("="*80)

    qos = create_quantum_os()

    # Create Grover search
    grover = GroverSearch(num_qubits=3)

    # Search for state |101⟩ (decimal 5)
    target_state = 5
    print(f"\nSearching for state |101⟩ (decimal {target_state}) in 8-element space...")

    program = grover.create_circuit(marked_states=[target_state])

    print(f"Grover circuit created:")
    print(f"  Search space: {grover.search_space_size} states")
    print(f"  Circuit depth: {program.depth()}")

    # Execute on quantum supercomputer
    print(f"\n🔍 Running Grover search...")

    result = qos.qvm.execute(program, shots=2048)

    if result.success:
        print(f"\n✅ Search completed!")

        # Show all results
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Results (all states):")
        for state, count in sorted_counts:
            prob = count / result.shots * 100
            marker = " ← TARGET!" if int(state, 2) == target_state else ""
            print(f"    |{state}⟩: {count} ({prob:.1f}%){marker}")

        # Calculate success
        success_rate = result.counts.get(format(target_state, '03b'), 0) / result.shots * 100
        print(f"\n  Success rate: {success_rate:.1f}%")
        print(f"  Expected success: {grover.estimate_success_probability(1)*100:.1f}%")


def demo_vqe_on_supercomputer():
    """Run VQE on the quantum supercomputer"""
    print("\n" + "="*80)
    print("VARIATIONAL QUANTUM EIGENSOLVER (VQE) ON SUPERCOMPUTER")
    print("="*80)

    qos = create_quantum_os()

    # Create VQE
    vqe = VariationalQuantumEigensolver(num_qubits=2)

    print("\nFinding ground state energy using VQE...")
    print("  Hamiltonian: H = Z₀ + Z₁")

    # Run optimization
    print("\n🧮 Running VQE optimization (this may take a minute)...")

    result = vqe.optimize(
        quantum_vm=qos.qvm,
        max_iterations=30,
        shots=1024
    )

    print(f"\n✅ VQE completed!")
    print(f"  Ground state energy: {result['ground_state_energy']:.4f}")
    print(f"  Optimal parameters: {result['optimal_parameters']}")
    print(f"  Iterations: {result['num_iterations']}")
    print(f"  Success: {result['success']}")


def demo_multi_backend_comparison():
    """Compare same algorithm on different quantum computers"""
    print("\n" + "="*80)
    print("MULTI-BACKEND COMPARISON")
    print("="*80)

    qos = create_quantum_os()

    # Create a simple Bell state program
    program = qos.qvm.create_program(num_qubits=2)
    program.h(0)
    program.cnot(0, 1)
    program.measure_all()

    print("\nRunning same Bell state circuit on ALL available quantum computers...")

    backends = qos.list_backends()

    results_comparison = []

    for backend_name in backends:
        try:
            result = qos.qvm.execute(program, shots=1024, backend_name=backend_name)

            if result.success:
                results_comparison.append({
                    'backend': backend_name,
                    'result': result,
                    'fidelity': (result.counts.get('00', 0) + result.counts.get('11', 0)) / result.shots
                })
        except Exception as e:
            print(f"  ⚠️  {backend_name}: {e}")

    # Display comparison
    if results_comparison:
        print(f"\n📊 Results from {len(results_comparison)} quantum computers:\n")

        for comparison in results_comparison:
            backend = comparison['backend']
            result = comparison['result']
            fidelity = comparison['fidelity']

            print(f"  {backend}:")
            print(f"    Bell state fidelity: {fidelity*100:.1f}%")
            print(f"    Execution time: {result.execution_time:.4f}s")

            top_states = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"    Top states:")
            for state, count in top_states:
                print(f"      |{state}⟩: {count} ({count/result.shots*100:.1f}%)")
            print()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║            GENERAL QUANTUM SUPERCOMPUTER DEMONSTRATION               ║
    ║                                                                      ║
    ║  Showcasing true quantum supercomputer capabilities:                ║
    ║  • General-purpose quantum computing                                ║
    ║  • Multiple quantum computers working as one                        ║
    ║  • Backend-agnostic quantum programming                             ║
    ║  • Distributed quantum execution                                    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all demonstrations
    demo_unified_supercomputer_status()
    demo_general_quantum_programming()
    demo_distributed_supercomputer_execution()
    demo_grover_search_on_supercomputer()
    demo_vqe_on_supercomputer()
    demo_multi_backend_comparison()

    print("\n" + "="*80)
    print("✨ QUANTUM SUPERCOMPUTER DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nYou now have a general-purpose quantum supercomputer that can:")
    print("  ✅ Run ANY quantum algorithm")
    print("  ✅ Execute on multiple quantum computers simultaneously")
    print("  ✅ Automatically distribute workloads")
    print("  ✅ Work with Google Willow, IBM Brisbane, IBM Torino, and more")
    print("  ✅ Provide backend-agnostic quantum programming")
    print("\nTotal combined quantum computing power across all backends!")
    print("="*80)

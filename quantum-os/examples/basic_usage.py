"""
Basic Usage Examples for Quantum OS

Demonstrates core functionality:
- Creating quantum circuits
- Executing on different backends
- Using error correction
- Resource management
"""

import sys
sys.path.insert(0, '..')

from quantum_os import create_quantum_os, SurfaceCode
import numpy as np


def example_1_basic_circuit():
    """Example 1: Create and execute a simple quantum circuit"""
    print("\n" + "="*60)
    print("Example 1: Basic Quantum Circuit")
    print("="*60)

    # Create Quantum OS
    qos = create_quantum_os()

    # Check available backends
    print(f"Available backends: {qos.list_backends()}")

    # Create a circuit using Cirq backend
    try:
        import cirq
        circuit = qos.create_circuit(num_qubits=3, backend_name='cirq_simulator')

        # Get qubits
        qubits = sorted(circuit.all_qubits())

        # Create Bell state on first two qubits
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))

        # Add superposition on third qubit
        circuit.append(cirq.H(qubits[2]))

        # Execute
        result = qos.execute(circuit, shots=1024, backend_name='cirq_simulator')

        print(f"\nExecution successful: {result.success}")
        print(f"Measurement counts: {result.counts}")
        print(f"Execution time: {result.execution_time:.4f}s")

    except ImportError:
        print("Cirq not installed, skipping this example")


def example_2_multiple_backends():
    """Example 2: Execute same circuit on multiple backends"""
    print("\n" + "="*60)
    print("Example 2: Multiple Backend Execution")
    print("="*60)

    qos = create_quantum_os()

    try:
        # Create circuit using Qiskit
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        # Execute on different backends
        backends = ['aer_simulator', 'tfq_simulator']

        for backend_name in backends:
            if backend_name in qos.list_backends():
                print(f"\nExecuting on {backend_name}...")

                result = qos.execute(circuit, shots=1024, backend_name=backend_name)

                if result.success:
                    print(f"  Results: {result.counts}")
                    print(f"  Time: {result.execution_time:.4f}s")
                else:
                    print(f"  Failed: {result.error_message}")

    except ImportError as e:
        print(f"Required library not installed: {e}")


def example_3_error_correction():
    """Example 3: Quantum error correction"""
    print("\n" + "="*60)
    print("Example 3: Quantum Error Correction")
    print("="*60)

    # Create surface code
    code = SurfaceCode(code_distance=5)

    # Get parameters
    params = code.get_code_parameters()

    print(f"\nSurface Code Parameters:")
    print(f"  Code distance: {params['code_distance']}")
    print(f"  Data qubits: {params['num_data_qubits']}")
    print(f"  Syndrome qubits: {params['num_syndrome_qubits']}")
    print(f"  Physical error rate: {params['physical_error_rate']:.4f}")
    print(f"  Logical error rate: {params['logical_error_rate']:.2e}")
    print(f"  Can correct errors: {params['can_correct_errors']}")

    # Calculate requirements for target error rate
    from quantum_os.error_correction.surface_codes import get_error_correction_requirements

    target_rate = 1e-9  # 1 error per billion operations
    requirements = get_error_correction_requirements(target_rate)

    print(f"\nRequirements for {target_rate:.0e} error rate:")
    for backend, req in requirements.items():
        if 'error' not in req:
            print(f"  {backend}:")
            print(f"    Code distance needed: {req['code_distance']}")
            print(f"    Total qubits: {req['total_qubits_needed']}")
            print(f"    Achievable: {req['achievable']}")


def example_4_resource_management():
    """Example 4: Resource management and monitoring"""
    print("\n" + "="*60)
    print("Example 4: Resource Management")
    print("="*60)

    qos = create_quantum_os()

    # Get system status
    status = qos.get_system_status()

    print(f"\nSystem Status:")
    print(f"  Version: {status['version']}")
    print(f"  Backends: {len(status['backends'])}")

    print(f"\nBackend Status:")
    for name, backend_status in status['backends'].items():
        print(f"  {name}:")
        print(f"    Available: {backend_status['available']}")
        print(f"    Type: {backend_status['type']}")
        print(f"    Mode: {backend_status['mode']}")

    print(f"\nResource Status:")
    resources = status['resources']
    print(f"  Max qubits: {resources['max_qubits']}")
    print(f"  Available qubits: {resources['available_qubits']}")
    print(f"  Active allocations: {resources['active_allocations']}")

    # System resources
    sys_res = resources['system_resources']
    print(f"\nSystem Resources:")
    print(f"  CPU usage: {sys_res['cpu_percent']:.1f}%")
    print(f"  Memory available: {sys_res['memory_available_gb']:.2f} GB")
    print(f"  GPU available: {sys_res.get('gpu_available', False)}")


def example_5_backend_properties():
    """Example 5: Query backend properties"""
    print("\n" + "="*60)
    print("Example 5: Backend Properties")
    print("="*60)

    qos = create_quantum_os()

    for backend_name in qos.list_backends():
        print(f"\n{backend_name}:")
        props = qos.get_backend_properties(backend_name)

        for key, value in props.items():
            if isinstance(value, (list, dict)):
                continue  # Skip complex structures
            print(f"  {key}: {value}")


def example_6_plugin_system():
    """Example 6: Load plugins from existing quantum projects"""
    print("\n" + "="*60)
    print("Example 6: Plugin System")
    print("="*60)

    from quantum_os import PluginLoader

    loader = PluginLoader()

    # Try to load L.L.M.A algorithms
    llma_path = "/mnt/c/Quantum-A.I.-Large-Language-Model-Agent-L.L.M.A--main"

    import os
    if os.path.exists(llma_path):
        print(f"\nLoading algorithms from L.L.M.A...")
        algorithms = loader.load_llma_algorithms(llma_path)

        print(f"Loaded {len(algorithms)} algorithm modules:")
        for name in algorithms.keys():
            print(f"  - {name}")

            # List functions in module
            functions = loader.get_plugin_functions(name)
            if functions:
                print(f"    Functions: {', '.join(functions[:5])}")
    else:
        print(f"L.L.M.A path not found: {llma_path}")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         Quantum OS - Basic Usage Examples                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Run examples
    example_1_basic_circuit()
    example_2_multiple_backends()
    example_3_error_correction()
    example_4_resource_management()
    example_5_backend_properties()
    example_6_plugin_system()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)

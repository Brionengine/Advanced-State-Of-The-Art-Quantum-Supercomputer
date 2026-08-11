"""
Test Quantum Backends
"""

import pytest
import sys
sys.path.insert(0, '..')

from quantum_os import QuantumOS, QuantumOSConfig


def test_backend_initialization():
    """Test backend initialization"""
    qos = QuantumOS()
    assert len(qos.backends) > 0, "No backends initialized"


def test_cirq_backend():
    """Test Cirq backend"""
    try:
        import cirq
        from quantum_os import CirqBackend, ExecutionMode

        backend = CirqBackend(execution_mode=ExecutionMode.SIMULATION)
        assert backend.initialize(), "Cirq backend failed to initialize"

        # Create and execute circuit
        circuit = backend.create_circuit(num_qubits=2)
        qubits = sorted(circuit.all_qubits())

        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))

        result = backend.execute(circuit, shots=1024)
        assert result.success, "Cirq execution failed"
        assert len(result.counts) > 0, "No measurement results"

    except ImportError:
        pytest.skip("Cirq not installed")


def test_qiskit_backend():
    """Test Qiskit backend"""
    try:
        from qiskit import QuantumCircuit
        from quantum_os import QiskitBackend, ExecutionMode

        backend = QiskitBackend(backend_name='aer_simulator', execution_mode=ExecutionMode.SIMULATION)
        assert backend.initialize(), "Qiskit backend failed to initialize"

        # Create and execute circuit
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        result = backend.execute(circuit, shots=1024)
        assert result.success, "Qiskit execution failed"
        assert len(result.counts) > 0, "No measurement results"

    except ImportError:
        pytest.skip("Qiskit not installed")


def test_backend_properties():
    """Test backend properties retrieval"""
    qos = QuantumOS()

    for backend_name in qos.list_backends():
        props = qos.get_backend_properties(backend_name)
        assert 'backend_name' in props
        assert 'backend_type' in props
        assert 'is_available' in props


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

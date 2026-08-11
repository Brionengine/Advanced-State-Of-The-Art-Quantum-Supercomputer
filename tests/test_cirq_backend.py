"""
Tests for CirqBackend circuit construction.

Cirq is an optional dependency, so every test here skips cleanly without it —
which is itself part of what is under test: the backends package must import
whether or not Cirq is installed.
"""

import pytest

cirq = pytest.importorskip("cirq", reason="Cirq is an optional backend dependency")

from quantum_os.backends import CirqBackend, ExecutionMode  # noqa: E402


@pytest.fixture
def backend():
    return CirqBackend(execution_mode=ExecutionMode.SIMULATION)


def test_created_circuit_exposes_its_qubits(backend):
    """Regression: qubits were stashed on a private attribute Cirq ignores."""
    circuit = backend.create_circuit(num_qubits=2)

    assert len(circuit.all_qubits()) == 2


@pytest.mark.parametrize("size", [1, 2, 5, 12])
def test_qubit_count_matches_the_request(backend, size):
    circuit = backend.create_circuit(num_qubits=size)

    assert len(circuit.all_qubits()) == size


def test_zero_qubit_circuit_is_empty(backend):
    circuit = backend.create_circuit(num_qubits=0)

    assert len(circuit.all_qubits()) == 0


def test_qubits_are_sortable_for_indexing(backend):
    """Callers do `sorted(circuit.all_qubits())[0]` to address the register."""
    circuit = backend.create_circuit(num_qubits=3)
    qubits = sorted(circuit.all_qubits())

    assert len(qubits) == 3
    assert qubits[0] != qubits[1]


def test_circuit_accepts_gates_on_its_own_qubits(backend):
    circuit = backend.create_circuit(num_qubits=2)
    qubits = sorted(circuit.all_qubits())

    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))

    assert len(circuit.all_qubits()) == 2


def test_identity_padding_does_not_disturb_measurements(backend):
    """The pinning gates are identities, so outcomes must be unaffected."""
    circuit = backend.create_circuit(num_qubits=2)
    qubits = sorted(circuit.all_qubits())
    circuit.append(cirq.X(qubits[0]))
    circuit.append(cirq.measure(*qubits, key="m"))

    result = cirq.Simulator().run(circuit, repetitions=20)
    outcomes = result.measurements["m"]

    # Qubit 0 was flipped, qubit 1 was not — identities changed nothing.
    assert all(shot[0] == 1 for shot in outcomes)
    assert all(shot[1] == 0 for shot in outcomes)


def test_backend_initializes(backend):
    assert backend.initialize() is True

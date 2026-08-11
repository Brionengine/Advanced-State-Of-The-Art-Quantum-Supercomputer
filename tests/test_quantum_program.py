"""
Tests for QuantumProgram scheduling and circuit metrics.

Depth is the property hardware feasibility is judged against, so the parallel
layer schedule behind it gets the most attention here.
"""

import pytest

from quantum_os.core.quantum_vm import (
    QuantumGateType,
    QuantumInstruction,
    QuantumProgram,
)


@pytest.fixture
def program():
    return QuantumProgram(num_qubits=4)


# -- Layer scheduling -------------------------------------------------------


def test_empty_program_has_zero_depth(program):
    assert program.depth() == 0
    assert program.layers() == []


def test_single_gate_has_depth_one(program):
    program.h(0)
    assert program.depth() == 1


def test_disjoint_gates_share_one_layer(program):
    """Gates on different qubits run concurrently — this is the whole point."""
    program.h(0)
    program.h(1)
    program.h(2)
    program.h(3)

    assert program.depth() == 1
    assert len(program.layers()[0]) == 4


def test_sequential_gates_on_one_qubit_stack(program):
    program.h(0)
    program.x(0)
    program.z(0)

    assert program.depth() == 3


def test_two_qubit_gate_serializes_both_operands(program):
    program.h(0)          # layer 0
    program.h(1)          # layer 0
    program.cnot(0, 1)    # layer 1 — must wait for both

    assert program.depth() == 2
    assert len(program.layers()[1]) == 1


def test_gate_waits_for_its_latest_operand(program):
    program.h(0)
    program.x(0)
    program.h(1)          # layer 0, independent
    program.cnot(0, 1)    # must wait for qubit 0's two gates

    assert program.depth() == 3


def test_independent_pairs_run_in_parallel(program):
    program.cnot(0, 1)
    program.cnot(2, 3)

    assert program.depth() == 1


def test_bell_pair_plus_measurement(program):
    bell = QuantumProgram(num_qubits=2)
    bell.h(0)
    bell.cnot(0, 1)
    bell.measure_all()

    # h -> cnot -> measurements (both measures are parallel)
    assert bell.depth() == 3


def test_layers_partition_every_instruction(program):
    program.h(0)
    program.cnot(0, 1)
    program.x(2)
    program.measure_all()

    scheduled = sum(len(layer) for layer in program.layers())
    assert scheduled == len(program.instructions)


def test_no_layer_reuses_a_qubit(program):
    program.h(0)
    program.cnot(0, 1)
    program.x(1)
    program.toffoli(0, 1, 2)
    program.measure_all()

    for layer in program.layers():
        seen = [q for inst in layer for q in inst.qubits]
        assert len(seen) == len(set(seen))


def test_depth_never_exceeds_instruction_count(program):
    program.h(0)
    program.cnot(0, 1)
    program.x(2)

    assert program.depth() <= len(program.instructions)


def test_depth_is_below_gate_count_for_a_wide_circuit():
    """The old implementation returned gate count and would fail this."""
    wide = QuantumProgram(num_qubits=50)
    for qubit in range(50):
        wide.h(qubit)

    assert wide.gate_count() == 50
    assert wide.depth() == 1


# -- Gate metrics -----------------------------------------------------------


def test_gate_count_excludes_measurements(program):
    program.h(0)
    program.x(1)
    program.measure_all()

    assert program.gate_count() == 2


def test_two_qubit_gate_count(program):
    program.h(0)
    program.cnot(0, 1)
    program.cz(2, 3)
    program.toffoli(0, 1, 2)

    assert program.two_qubit_gate_count() == 2


def test_two_qubit_count_ignores_measurements(program):
    program.measure_all()
    assert program.two_qubit_gate_count() == 0


def test_gate_histogram_counts_by_type(program):
    program.h(0)
    program.h(1)
    program.cnot(0, 1)

    histogram = program.gate_histogram()
    assert histogram["H"] == 2
    assert histogram["CNOT"] == 1


def test_gate_histogram_includes_measurements(program):
    program.measure(0)
    assert program.gate_histogram()["MEASURE"] == 1


def test_gate_histogram_of_empty_program(program):
    assert program.gate_histogram() == {}


# -- Program construction ---------------------------------------------------


def test_classical_bits_default_to_qubit_count():
    assert QuantumProgram(num_qubits=5).num_classical_bits == 5


def test_classical_bits_can_be_set_independently():
    assert QuantumProgram(num_qubits=5, num_classical_bits=2).num_classical_bits == 2


def test_measure_all_covers_every_qubit(program):
    program.measure_all()

    measured = {
        inst.qubits[0] for inst in program.instructions
        if inst.gate_type == QuantumGateType.MEASURE
    }
    assert measured == {0, 1, 2, 3}


def test_rotation_gates_carry_their_angle(program):
    program.rx(0, 1.57)

    assert program.instructions[0].parameters == [1.57]


def test_barrier_is_stored_as_metadata_not_an_instruction(program):
    program.barrier()

    assert program.instructions == []
    assert program.metadata["barriers"] == [[0, 1, 2, 3]]


def test_add_gate_accepts_a_bare_int(program):
    program.add_gate(QuantumGateType.HADAMARD, 2)

    assert program.instructions[0].qubits == [2]


def test_to_dict_round_trips_shape(program):
    program.h(0)
    data = program.to_dict()

    assert data["num_qubits"] == 4
    assert len(data["instructions"]) == 1

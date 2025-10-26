"""
Quantum Virtual Machine (QVM)

Provides a general-purpose quantum computing interface that abstracts
away backend-specific details, enabling any quantum algorithm to run
on any quantum hardware.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
from enum import Enum


class QuantumGateType(Enum):
    """Universal quantum gate types"""
    # Single-qubit gates
    HADAMARD = "H"
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    S_GATE = "S"
    T_GATE = "T"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    U = "U"  # Universal single-qubit gate

    # Two-qubit gates
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"
    ISWAP = "ISWAP"

    # Three-qubit gates
    TOFFOLI = "TOFFOLI"
    FREDKIN = "FREDKIN"

    # Measurement
    MEASURE = "MEASURE"


class QuantumInstruction:
    """
    Universal quantum instruction

    Represents a single quantum operation that can be executed
    on any quantum backend
    """

    def __init__(
        self,
        gate_type: QuantumGateType,
        qubits: List[int],
        parameters: Optional[List[float]] = None,
        classical_bits: Optional[List[int]] = None
    ):
        """
        Initialize quantum instruction

        Args:
            gate_type: Type of quantum gate
            qubits: Qubit indices
            parameters: Gate parameters (for parameterized gates)
            classical_bits: Classical bit indices (for measurements)
        """
        self.gate_type = gate_type
        self.qubits = qubits
        self.parameters = parameters or []
        self.classical_bits = classical_bits or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'gate': self.gate_type.value,
            'qubits': self.qubits,
            'parameters': self.parameters,
            'classical_bits': self.classical_bits
        }

    def __repr__(self) -> str:
        params = f"({', '.join(map(str, self.parameters))})" if self.parameters else ""
        qubits = ', '.join(map(str, self.qubits))
        return f"{self.gate_type.value}{params} q[{qubits}]"


class QuantumProgram:
    """
    General-purpose quantum program

    A backend-agnostic quantum program that can be compiled
    and executed on any quantum computer
    """

    def __init__(self, num_qubits: int, num_classical_bits: Optional[int] = None):
        """
        Initialize quantum program

        Args:
            num_qubits: Number of qubits
            num_classical_bits: Number of classical bits (defaults to num_qubits)
        """
        self.num_qubits = num_qubits
        self.num_classical_bits = num_classical_bits or num_qubits
        self.instructions: List[QuantumInstruction] = []
        self.metadata: Dict[str, Any] = {}

    def add_gate(
        self,
        gate_type: QuantumGateType,
        qubits: Union[int, List[int]],
        parameters: Optional[List[float]] = None
    ):
        """
        Add a quantum gate

        Args:
            gate_type: Type of gate
            qubits: Qubit index or list of indices
            parameters: Gate parameters
        """
        if isinstance(qubits, int):
            qubits = [qubits]

        instruction = QuantumInstruction(gate_type, qubits, parameters)
        self.instructions.append(instruction)

    # Convenience methods for common gates
    def h(self, qubit: int):
        """Apply Hadamard gate"""
        self.add_gate(QuantumGateType.HADAMARD, qubit)

    def x(self, qubit: int):
        """Apply Pauli-X gate"""
        self.add_gate(QuantumGateType.PAULI_X, qubit)

    def y(self, qubit: int):
        """Apply Pauli-Y gate"""
        self.add_gate(QuantumGateType.PAULI_Y, qubit)

    def z(self, qubit: int):
        """Apply Pauli-Z gate"""
        self.add_gate(QuantumGateType.PAULI_Z, qubit)

    def rx(self, qubit: int, angle: float):
        """Apply rotation around X axis"""
        self.add_gate(QuantumGateType.RX, qubit, [angle])

    def ry(self, qubit: int, angle: float):
        """Apply rotation around Y axis"""
        self.add_gate(QuantumGateType.RY, qubit, [angle])

    def rz(self, qubit: int, angle: float):
        """Apply rotation around Z axis"""
        self.add_gate(QuantumGateType.RZ, qubit, [angle])

    def cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        self.add_gate(QuantumGateType.CNOT, [control, target])

    def cz(self, control: int, target: int):
        """Apply CZ gate"""
        self.add_gate(QuantumGateType.CZ, [control, target])

    def swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate"""
        self.add_gate(QuantumGateType.SWAP, [qubit1, qubit2])

    def toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli (CCNOT) gate"""
        self.add_gate(QuantumGateType.TOFFOLI, [control1, control2, target])

    def measure(self, qubit: int, classical_bit: Optional[int] = None):
        """
        Measure a qubit

        Args:
            qubit: Qubit to measure
            classical_bit: Classical bit to store result (defaults to same index)
        """
        if classical_bit is None:
            classical_bit = qubit

        instruction = QuantumInstruction(
            QuantumGateType.MEASURE,
            [qubit],
            classical_bits=[classical_bit]
        )
        self.instructions.append(instruction)

    def measure_all(self):
        """Measure all qubits"""
        for i in range(self.num_qubits):
            self.measure(i)

    def barrier(self, qubits: Optional[List[int]] = None):
        """Add a barrier (for visualization/optimization purposes)"""
        # Barriers are often backend-specific, store as metadata
        if 'barriers' not in self.metadata:
            self.metadata['barriers'] = []
        self.metadata['barriers'].append(qubits or list(range(self.num_qubits)))

    def depth(self) -> int:
        """Calculate circuit depth (approximation)"""
        # Simplified depth calculation
        return len(self.instructions)

    def gate_count(self) -> int:
        """Count total number of gates"""
        return len([i for i in self.instructions if i.gate_type != QuantumGateType.MEASURE])

    def to_dict(self) -> Dict[str, Any]:
        """Convert program to dictionary"""
        return {
            'num_qubits': self.num_qubits,
            'num_classical_bits': self.num_classical_bits,
            'instructions': [inst.to_dict() for inst in self.instructions],
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        return f"QuantumProgram({self.num_qubits} qubits, {len(self.instructions)} instructions)"


class QuantumVirtualMachine:
    """
    Quantum Virtual Machine

    Provides a general-purpose quantum computing interface that can:
    - Compile quantum programs to any backend
    - Execute on multiple quantum computers simultaneously
    - Distribute workload across available quantum resources
    - Abstract hardware-specific details
    """

    def __init__(self, quantum_os):
        """
        Initialize QVM

        Args:
            quantum_os: QuantumOS instance
        """
        self.quantum_os = quantum_os
        self.program_cache: Dict[str, Any] = {}

    def create_program(self, num_qubits: int) -> QuantumProgram:
        """
        Create a new quantum program

        Args:
            num_qubits: Number of qubits

        Returns:
            QuantumProgram instance
        """
        return QuantumProgram(num_qubits)

    def compile(
        self,
        program: QuantumProgram,
        backend_name: Optional[str] = None
    ) -> Any:
        """
        Compile quantum program to backend-specific circuit

        Args:
            program: Quantum program to compile
            backend_name: Target backend (auto-select if None)

        Returns:
            Backend-specific circuit
        """
        backend = self.quantum_os.get_backend(backend_name)
        if not backend:
            raise RuntimeError("No backend available")

        backend_type = backend.backend_type.value

        if backend_type == 'cirq':
            return self._compile_to_cirq(program)
        elif backend_type == 'qiskit':
            return self._compile_to_qiskit(program)
        elif backend_type == 'tfq':
            return self._compile_to_cirq(program)  # TFQ uses Cirq

        raise ValueError(f"Unsupported backend type: {backend_type}")

    def _compile_to_cirq(self, program: QuantumProgram) -> Any:
        """Compile to Cirq circuit"""
        import cirq

        qubits = [cirq.GridQubit(i // 10, i % 10) for i in range(program.num_qubits)]
        circuit = cirq.Circuit()

        gate_map = {
            QuantumGateType.HADAMARD: lambda q: cirq.H(qubits[q[0]]),
            QuantumGateType.PAULI_X: lambda q: cirq.X(qubits[q[0]]),
            QuantumGateType.PAULI_Y: lambda q: cirq.Y(qubits[q[0]]),
            QuantumGateType.PAULI_Z: lambda q: cirq.Z(qubits[q[0]]),
            QuantumGateType.S_GATE: lambda q: cirq.S(qubits[q[0]]),
            QuantumGateType.T_GATE: lambda q: cirq.T(qubits[q[0]]),
            QuantumGateType.CNOT: lambda q: cirq.CNOT(qubits[q[0]], qubits[q[1]]),
            QuantumGateType.CZ: lambda q: cirq.CZ(qubits[q[0]], qubits[q[1]]),
            QuantumGateType.SWAP: lambda q: cirq.SWAP(qubits[q[0]], qubits[q[1]]),
            QuantumGateType.TOFFOLI: lambda q: cirq.TOFFOLI(qubits[q[0]], qubits[q[1]], qubits[q[2]]),
        }

        for inst in program.instructions:
            if inst.gate_type == QuantumGateType.RX:
                circuit.append(cirq.rx(inst.parameters[0])(qubits[inst.qubits[0]]))
            elif inst.gate_type == QuantumGateType.RY:
                circuit.append(cirq.ry(inst.parameters[0])(qubits[inst.qubits[0]]))
            elif inst.gate_type == QuantumGateType.RZ:
                circuit.append(cirq.rz(inst.parameters[0])(qubits[inst.qubits[0]]))
            elif inst.gate_type == QuantumGateType.MEASURE:
                # Measurements handled separately in Cirq
                pass
            elif inst.gate_type in gate_map:
                circuit.append(gate_map[inst.gate_type](inst.qubits))

        # Add measurements
        measure_qubits = [
            qubits[inst.qubits[0]]
            for inst in program.instructions
            if inst.gate_type == QuantumGateType.MEASURE
        ]
        if measure_qubits:
            circuit.append(cirq.measure(*measure_qubits, key='result'))

        return circuit

    def _compile_to_qiskit(self, program: QuantumProgram) -> Any:
        """Compile to Qiskit circuit"""
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(program.num_qubits, program.num_classical_bits)

        for inst in program.instructions:
            q = inst.qubits

            if inst.gate_type == QuantumGateType.HADAMARD:
                circuit.h(q[0])
            elif inst.gate_type == QuantumGateType.PAULI_X:
                circuit.x(q[0])
            elif inst.gate_type == QuantumGateType.PAULI_Y:
                circuit.y(q[0])
            elif inst.gate_type == QuantumGateType.PAULI_Z:
                circuit.z(q[0])
            elif inst.gate_type == QuantumGateType.S_GATE:
                circuit.s(q[0])
            elif inst.gate_type == QuantumGateType.T_GATE:
                circuit.t(q[0])
            elif inst.gate_type == QuantumGateType.RX:
                circuit.rx(inst.parameters[0], q[0])
            elif inst.gate_type == QuantumGateType.RY:
                circuit.ry(inst.parameters[0], q[0])
            elif inst.gate_type == QuantumGateType.RZ:
                circuit.rz(inst.parameters[0], q[0])
            elif inst.gate_type == QuantumGateType.CNOT:
                circuit.cx(q[0], q[1])
            elif inst.gate_type == QuantumGateType.CZ:
                circuit.cz(q[0], q[1])
            elif inst.gate_type == QuantumGateType.SWAP:
                circuit.swap(q[0], q[1])
            elif inst.gate_type == QuantumGateType.TOFFOLI:
                circuit.ccx(q[0], q[1], q[2])
            elif inst.gate_type == QuantumGateType.MEASURE:
                circuit.measure(q[0], inst.classical_bits[0])

        return circuit

    def execute(
        self,
        program: QuantumProgram,
        shots: int = 1024,
        backend_name: Optional[str] = None
    ) -> Any:
        """
        Execute quantum program

        Args:
            program: Quantum program
            shots: Number of shots
            backend_name: Backend to use

        Returns:
            Execution result
        """
        # Compile program
        circuit = self.compile(program, backend_name)

        # Execute on quantum OS
        return self.quantum_os.execute(circuit, shots, backend_name)

    def execute_distributed(
        self,
        program: QuantumProgram,
        shots: int = 1024,
        backend_names: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Execute program on multiple backends simultaneously

        Args:
            program: Quantum program
            shots: Shots per backend
            backend_names: List of backends to use

        Returns:
            List of results from each backend
        """
        if backend_names is None:
            backend_names = self.quantum_os.list_backends()

        results = []
        for backend_name in backend_names:
            try:
                result = self.execute(program, shots, backend_name)
                results.append({
                    'backend': backend_name,
                    'result': result,
                    'success': result.success if result else False
                })
            except Exception as e:
                results.append({
                    'backend': backend_name,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })

        return results

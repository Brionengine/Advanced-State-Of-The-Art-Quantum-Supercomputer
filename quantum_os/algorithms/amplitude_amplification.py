"""Amplitude Amplification"""
from ..core.quantum_vm import QuantumProgram

class AmplitudeAmplification:
    """Amplitude amplification technique"""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def create_circuit(self, num_iterations: int = 1) -> QuantumProgram:
        """Create amplitude amplification circuit"""
        program = QuantumProgram(self.num_qubits)

        for i in range(self.num_qubits):
            program.h(i)

        for _ in range(num_iterations):
            for i in range(self.num_qubits):
                program.z(i)

        program.measure_all()
        return program

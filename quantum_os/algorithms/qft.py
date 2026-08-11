"""Quantum Fourier Transform"""
from ..core.quantum_vm import QuantumProgram
import numpy as np

class QuantumFourierTransform:
    """Quantum Fourier Transform"""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def create_circuit(self) -> QuantumProgram:
        """Create QFT circuit"""
        program = QuantumProgram(self.num_qubits)

        for i in range(self.num_qubits):
            program.h(i)
            for j in range(i + 1, self.num_qubits):
                angle = np.pi / (2 ** (j - i))
                program.rz(j, angle)

        program.measure_all()
        return program

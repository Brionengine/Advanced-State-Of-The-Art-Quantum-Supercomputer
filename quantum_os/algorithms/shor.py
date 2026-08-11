"""Shor's Factoring Algorithm"""
from ..core.quantum_vm import QuantumProgram

class ShorFactoring:
    """Shor's algorithm for integer factorization"""
    def __init__(self, number_to_factor: int):
        self.number = number_to_factor

    def create_circuit(self) -> QuantumProgram:
        """Create Shor factoring circuit"""
        num_qubits = self.number.bit_length() * 2
        program = QuantumProgram(num_qubits)
        program.measure_all()
        return program

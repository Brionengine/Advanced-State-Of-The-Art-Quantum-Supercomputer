"""
Quantum Echoes Applications
===========================

Real-world applications using Quantum Echoes framework.
"""

from .quantum_sensing import QuantumSensor, PrecisionMeasurement
from .quantum_communication import (
    QuantumKeyDistribution,
    QuantumTeleportation,
    SecureQuantumChannel,
    QuantumKey,
    TeleportationResult
)
from .quantum_simulation import (
    QuantumChemistrySimulator,
    MaterialsSimulator,
    MolecularSystem,
    SimulationResult,
    simulate_h2_molecule,
    simulate_water_molecule
)
from .quantum_cryptography import (
    QuantumRandomNumberGenerator,
    LatticeCryptography,
    QuantumSignatureScheme,
    QuantumSecureHash,
    QuantumSignature,
    LatticeKey,
    demonstrate_post_quantum_crypto
)

__all__ = [
    # Sensing
    'QuantumSensor',
    'PrecisionMeasurement',

    # Communication
    'QuantumKeyDistribution',
    'QuantumTeleportation',
    'SecureQuantumChannel',
    'QuantumKey',
    'TeleportationResult',

    # Simulation
    'QuantumChemistrySimulator',
    'MaterialsSimulator',
    'MolecularSystem',
    'SimulationResult',
    'simulate_h2_molecule',
    'simulate_water_molecule',

    # Cryptography
    'QuantumRandomNumberGenerator',
    'LatticeCryptography',
    'QuantumSignatureScheme',
    'QuantumSecureHash',
    'QuantumSignature',
    'LatticeKey',
    'demonstrate_post_quantum_crypto'
]

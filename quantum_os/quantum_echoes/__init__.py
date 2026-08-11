"""
Quantum Echoes Module
====================

Topological quantum computation using specialized ion emissions and particle echoes.

This module implements the Quantum Echoes algorithm for practical,
fault-tolerant quantum computing using topological qubits created from
specialized ions that emit cascades of sub-particles.

Core Features:
- Specialized ion emission system
- Topological qubits with inherent fault tolerance
- Quantum echo propagation and interference
- Topological error correction
- Practical quantum applications

Usage:
    from quantum_echoes import QuantumEchoesAlgorithm, build_circuit

    # Create algorithm instance
    qe = QuantumEchoesAlgorithm(num_qubits=4)

    # Initialize qubits
    qe.initialize_qubits('zero')

    # Build circuit
    circuit = build_circuit(4)
    circuit.h(0).cnot(0, 1).cnot(0, 2).cnot(0, 3).measure_all()

    # Run circuit
    results = qe.run_circuit(circuit.build())

Author: Quantum Echoes Development Team
Version: 1.0.0
"""

__version__ = '1.0.0'

# Core components
from .core.particle_emitter import (
    SpecializedIonEmitter,
    ParticleType,
    EmissionPattern
)

from .core.topological_qubit import (
    TopologicalQubit,
    AnyonType,
    BraidingOperation
)

from .core.echo_propagation import (
    EchoPropagator,
    EchoState,
    PropagationMode
)

from .core.particle_registry import (
    ParticleRegistry,
    ParticleProperties,
    get_particle,
    register_particle
)

# Algorithms
from .algorithms.quantum_echoes import (
    QuantumEchoesAlgorithm,
    EchoCircuit
)

# Circuit builder
from .circuits.echo_circuit_builder import (
    EchoCircuitBuilder,
    build_circuit,
    create_bell_circuit,
    create_ghz_circuit
)

# Error correction
from .error_correction.topological_protection import (
    TopologicalErrorCorrection,
    SurfaceCode
)

# Configuration
from .config.echo_settings import (
    EchoSettings,
    get_default_settings,
    get_high_fidelity_settings,
    get_fast_simulation_settings
)

# Applications
from .applications.quantum_sensing import (
    QuantumSensor,
    PrecisionMeasurement
)

__all__ = [
    # Core
    'SpecializedIonEmitter',
    'ParticleType',
    'EmissionPattern',
    'TopologicalQubit',
    'AnyonType',
    'BraidingOperation',
    'EchoPropagator',
    'EchoState',
    'PropagationMode',
    'ParticleRegistry',
    'ParticleProperties',
    'get_particle',
    'register_particle',

    # Algorithms
    'QuantumEchoesAlgorithm',
    'EchoCircuit',

    # Circuit Builder
    'EchoCircuitBuilder',
    'build_circuit',
    'create_bell_circuit',
    'create_ghz_circuit',

    # Error Correction
    'TopologicalErrorCorrection',
    'SurfaceCode',

    # Configuration
    'EchoSettings',
    'get_default_settings',
    'get_high_fidelity_settings',
    'get_fast_simulation_settings',

    # Applications
    'QuantumSensor',
    'PrecisionMeasurement',
]


def print_info():
    """Print information about Quantum Echoes module."""
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Quantum Echoes v{__version__}                         ║
    ║   Topological Quantum Computing with Ion Emissions        ║
    ╚═══════════════════════════════════════════════════════════╝

    Core Components:
      - Specialized Ion Emitter
      - Topological Qubits (Anyonic)
      - Echo Propagation Engine
      - Particle Registry

    Algorithms:
      - Quantum Echoes Algorithm
      - Topological Gate Set
      - Error Correction (Surface Code)

    Applications:
      - Quantum Sensing
      - Quantum Communication
      - Quantum Simulation

    Quick Start:
      >>> from quantum_echoes import QuantumEchoesAlgorithm, build_circuit
      >>> qe = QuantumEchoesAlgorithm(num_qubits=2)
      >>> qe.initialize_qubits('zero')
      >>> circuit = build_circuit(2).h(0).cnot(0, 1).measure_all().build()
      >>> results = qe.run_circuit(circuit)

    For more information, see documentation at quantum_echoes/examples/
    """)


# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

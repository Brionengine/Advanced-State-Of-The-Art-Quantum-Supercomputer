"""
Quantum Echoes Core Module
==========================

Core particle and qubit systems for Quantum Echoes algorithm.
Implements specialized ion emission, topological qubits, and echo propagation.
"""

from .particle_emitter import SpecializedIonEmitter, ParticleType, EmissionPattern
from .topological_qubit import TopologicalQubit, AnyonType, BraidingOperation
from .echo_propagation import EchoPropagator, EchoState, PropagationMode
from .particle_registry import ParticleRegistry, register_particle, get_particle

__all__ = [
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
    'register_particle',
    'get_particle'
]

__version__ = '1.0.0'

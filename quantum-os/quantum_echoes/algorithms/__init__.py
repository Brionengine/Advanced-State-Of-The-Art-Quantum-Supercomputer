"""
Quantum Echoes Algorithms Module
=================================

Implementation of quantum algorithms using the Quantum Echoes framework.
"""

from .quantum_echoes import QuantumEchoesAlgorithm, EchoCircuit
from .echo_interference import InterferenceEngine, InterferencePattern
from .topological_gates import TopologicalGateSet, AnyonicBraiding
from .echo_amplification import SignalAmplifier, CoherentAmplification
from .quantum_echoes_search import (
    GroverSearch,
    AmplitudeAmplification,
    EchoEnhancedSearch,
    grover_search_example,
    amplitude_amplification_example,
    echo_enhanced_search_example
)

__all__ = [
    'QuantumEchoesAlgorithm',
    'EchoCircuit',
    'InterferenceEngine',
    'InterferencePattern',
    'TopologicalGateSet',
    'AnyonicBraiding',
    'SignalAmplifier',
    'CoherentAmplification',
    'GroverSearch',
    'AmplitudeAmplification',
    'EchoEnhancedSearch',
    'grover_search_example',
    'amplitude_amplification_example',
    'echo_enhanced_search_example'
]

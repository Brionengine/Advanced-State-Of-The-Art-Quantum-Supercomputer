"""
General-Purpose Quantum Algorithm Library

Provides implementations of fundamental quantum algorithms that can
run on any quantum backend
"""

from .grover import GroverSearch
from .shor import ShorFactoring
from .vqe import VariationalQuantumEigensolver
from .qaoa import QuantumApproximateOptimization
from .qft import QuantumFourierTransform
from .amplitude_amplification import AmplitudeAmplification

__all__ = [
    'GroverSearch',
    'ShorFactoring',
    'VariationalQuantumEigensolver',
    'QuantumApproximateOptimization',
    'QuantumFourierTransform',
    'AmplitudeAmplification',
]

"""
Classical Computing Engine

Provides classical computing capabilities within the quantum supercomputer,
enabling hybrid quantum-classical operations and automatic selection of
the best computing paradigm for each task.
"""

from .engine import ClassicalComputingEngine
from .algorithms import ClassicalAlgorithms
from .optimizer import HybridOptimizer

__all__ = ['ClassicalComputingEngine', 'ClassicalAlgorithms', 'HybridOptimizer']

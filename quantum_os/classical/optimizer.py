"""
Hybrid Quantum-Classical Optimizer

Automatically determines whether to use quantum or classical computing
for maximum performance
"""

import numpy as np
from typing import Dict, Any, Callable, Optional
from enum import Enum


class ComputingParadigm(Enum):
    """Computing paradigm selection"""
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"  # Use both


class QuantumAdvantage(Enum):
    """Quantum advantage categories"""
    EXPONENTIAL = "exponential"  # Exponential speedup (e.g., Shor, Grover)
    QUADRATIC = "quadratic"  # Quadratic speedup (e.g., Grover)
    POLYNOMIAL = "polynomial"  # Polynomial speedup
    NONE = "none"  # No quantum advantage
    UNKNOWN = "unknown"  # Advantage not yet determined


class HybridOptimizer:
    """
    Hybrid Quantum-Classical Optimizer

    Intelligently decides whether to use quantum or classical computing
    for each task, maximizing performance
    """

    def __init__(self, quantum_os):
        """
        Initialize hybrid optimizer

        Args:
            quantum_os: QuantumOS instance
        """
        self.quantum_os = quantum_os

        # Problem size thresholds for quantum advantage
        self.quantum_thresholds = {
            'search': 1000,  # Grover advantage kicks in around N > 1000
            'factoring': 100,  # Shor needs large numbers
            'optimization': 50,  # QAOA/VQE for medium problems
            'simulation': 20,  # Quantum simulation always better
            'linear_algebra': 1000,  # HHL algorithm threshold
        }

    def analyze_problem(
        self,
        problem_type: str,
        problem_size: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze problem to determine quantum advantage

        Args:
            problem_type: Type of problem (search, sort, optimization, etc.)
            problem_size: Size of the problem
            **kwargs: Additional problem parameters

        Returns:
            Analysis results
        """
        analysis = {
            'problem_type': problem_type,
            'problem_size': problem_size,
            'recommended_paradigm': ComputingParadigm.CLASSICAL,
            'quantum_advantage': QuantumAdvantage.NONE,
            'speedup_factor': 1.0,
            'reasoning': []
        }

        # Analyze based on problem type
        if problem_type == 'search':
            analysis.update(self._analyze_search(problem_size, kwargs))

        elif problem_type == 'sort':
            # Classical sorting is always better (no known quantum advantage)
            analysis['reasoning'].append("Classical sorting algorithms are optimal")
            analysis['speedup_factor'] = 1.0

        elif problem_type == 'factoring':
            analysis.update(self._analyze_factoring(problem_size))

        elif problem_type == 'optimization':
            analysis.update(self._analyze_optimization(problem_size, kwargs))

        elif problem_type == 'simulation':
            analysis.update(self._analyze_simulation(problem_size))

        elif problem_type == 'matrix_operations':
            analysis.update(self._analyze_matrix_operations(problem_size, kwargs))

        elif problem_type == 'machine_learning':
            analysis.update(self._analyze_ml(problem_size, kwargs))

        return analysis

    def _analyze_search(self, size: int, params: dict) -> dict:
        """Analyze search problem"""
        is_unstructured = params.get('unstructured', True)

        if is_unstructured and size > self.quantum_thresholds['search']:
            # Grover's algorithm provides quadratic speedup
            speedup = np.sqrt(size)
            return {
                'recommended_paradigm': ComputingParadigm.QUANTUM,
                'quantum_advantage': QuantumAdvantage.QUADRATIC,
                'speedup_factor': speedup,
                'reasoning': [
                    f"Grover's algorithm provides O(√N) speedup",
                    f"Classical: O({size}) operations",
                    f"Quantum: O({int(np.sqrt(size))}) operations",
                    f"Speedup: {speedup:.2f}x"
                ]
            }
        else:
            return {
                'recommended_paradigm': ComputingParadigm.CLASSICAL,
                'quantum_advantage': QuantumAdvantage.NONE,
                'speedup_factor': 1.0,
                'reasoning': [
                    "Problem size too small for quantum advantage",
                    "Classical search more efficient"
                ]
            }

    def _analyze_factoring(self, number_bits: int) -> dict:
        """Analyze factoring problem"""
        if number_bits > self.quantum_thresholds['factoring']:
            # Shor's algorithm provides exponential speedup
            classical_time = 2 ** (number_bits ** (1/3))
            quantum_time = number_bits ** 2

            speedup = classical_time / quantum_time

            return {
                'recommended_paradigm': ComputingParadigm.QUANTUM,
                'quantum_advantage': QuantumAdvantage.EXPONENTIAL,
                'speedup_factor': speedup,
                'reasoning': [
                    "Shor's algorithm provides exponential speedup",
                    f"Classical: O(2^(n^(1/3))) = ~{classical_time:.0e}",
                    f"Quantum: O(n^2) = {quantum_time}",
                    f"Speedup: {speedup:.2e}x"
                ]
            }
        else:
            return {
                'recommended_paradigm': ComputingParadigm.CLASSICAL,
                'quantum_advantage': QuantumAdvantage.NONE,
                'speedup_factor': 1.0,
                'reasoning': ["Number too small for quantum advantage"]
            }

    def _analyze_optimization(self, size: int, params: dict) -> dict:
        """Analyze optimization problem"""
        is_combinatorial = params.get('combinatorial', False)

        if is_combinatorial and size > self.quantum_thresholds['optimization']:
            return {
                'recommended_paradigm': ComputingParadigm.HYBRID,
                'quantum_advantage': QuantumAdvantage.POLYNOMIAL,
                'speedup_factor': size ** 0.5,
                'reasoning': [
                    "QAOA/VQE can provide polynomial speedup",
                    "Hybrid quantum-classical approach recommended",
                    f"Potential speedup: {size**0.5:.2f}x"
                ]
            }
        else:
            return {
                'recommended_paradigm': ComputingParadigm.CLASSICAL,
                'quantum_advantage': QuantumAdvantage.NONE,
                'speedup_factor': 1.0,
                'reasoning': ["Classical optimization more efficient for this size"]
            }

    def _analyze_simulation(self, num_particles: int) -> dict:
        """Analyze quantum simulation problem"""
        # Quantum simulation of quantum systems is exponentially better
        speedup = 2 ** num_particles

        return {
            'recommended_paradigm': ComputingParadigm.QUANTUM,
            'quantum_advantage': QuantumAdvantage.EXPONENTIAL,
            'speedup_factor': speedup,
            'reasoning': [
                "Quantum simulation provides exponential advantage",
                f"Classical: O(2^n) = {speedup:.2e}",
                f"Quantum: O(n) = {num_particles}",
                "This is ideal for quantum computing!"
            ]
        }

    def _analyze_matrix_operations(self, size: int, params: dict) -> dict:
        """Analyze matrix operations"""
        operation = params.get('operation', 'multiply')

        if operation == 'inversion' and size > self.quantum_thresholds['linear_algebra']:
            return {
                'recommended_paradigm': ComputingParadigm.QUANTUM,
                'quantum_advantage': QuantumAdvantage.EXPONENTIAL,
                'speedup_factor': size ** 2,
                'reasoning': [
                    "HHL algorithm can provide exponential speedup",
                    f"Classical: O(n^3) = {size**3}",
                    f"Quantum: O(log n) = {int(np.log2(size))}",
                    "Significant advantage for large matrices"
                ]
            }
        else:
            return {
                'recommended_paradigm': ComputingParadigm.CLASSICAL,
                'quantum_advantage': QuantumAdvantage.NONE,
                'speedup_factor': 1.0,
                'reasoning': [
                    "Classical matrix operations are well-optimized",
                    "GPU acceleration provides best performance"
                ]
            }

    def _analyze_ml(self, dataset_size: int, params: dict) -> dict:
        """Analyze machine learning problem"""
        model_type = params.get('model_type', 'classical')

        if model_type == 'quantum' or dataset_size > 10000:
            return {
                'recommended_paradigm': ComputingParadigm.HYBRID,
                'quantum_advantage': QuantumAdvantage.POLYNOMIAL,
                'speedup_factor': np.sqrt(dataset_size),
                'reasoning': [
                    "Quantum machine learning can provide polynomial speedup",
                    "Hybrid approach recommended for best results",
                    "Use quantum for feature extraction, classical for training"
                ]
            }
        else:
            return {
                'recommended_paradigm': ComputingParadigm.CLASSICAL,
                'quantum_advantage': QuantumAdvantage.NONE,
                'speedup_factor': 1.0,
                'reasoning': ["Classical ML is more mature and efficient for small datasets"]
            }

    def recommend_execution(
        self,
        problem_type: str,
        problem_size: int,
        **kwargs
    ) -> ComputingParadigm:
        """
        Recommend computing paradigm for a problem

        Args:
            problem_type: Type of problem
            problem_size: Size of problem
            **kwargs: Additional parameters

        Returns:
            Recommended computing paradigm
        """
        analysis = self.analyze_problem(problem_type, problem_size, **kwargs)
        return analysis['recommended_paradigm']

    def get_speedup_estimate(
        self,
        problem_type: str,
        problem_size: int,
        **kwargs
    ) -> float:
        """
        Estimate quantum speedup for a problem

        Args:
            problem_type: Type of problem
            problem_size: Size of problem
            **kwargs: Additional parameters

        Returns:
            Estimated speedup factor
        """
        analysis = self.analyze_problem(problem_type, problem_size, **kwargs)
        return analysis['speedup_factor']

    def compare_approaches(
        self,
        problem_type: str,
        problem_size: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare quantum vs classical approaches

        Args:
            problem_type: Type of problem
            problem_size: Size of problem
            **kwargs: Additional parameters

        Returns:
            Comparison results
        """
        analysis = self.analyze_problem(problem_type, problem_size, **kwargs)

        return {
            'quantum_time_complexity': self._get_quantum_complexity(problem_type, problem_size),
            'classical_time_complexity': self._get_classical_complexity(problem_type, problem_size),
            'quantum_space_complexity': self._get_quantum_space(problem_type, problem_size),
            'classical_space_complexity': self._get_classical_space(problem_type, problem_size),
            'speedup_factor': analysis['speedup_factor'],
            'quantum_advantage': analysis['quantum_advantage'].value,
            'recommendation': analysis['recommended_paradigm'].value,
            'reasoning': analysis['reasoning']
        }

    def _get_quantum_complexity(self, problem_type: str, size: int) -> str:
        """Get quantum time complexity"""
        complexities = {
            'search': f"O(√{size})",
            'factoring': f"O({int(np.log2(size))}²)",
            'sort': f"O({size})",  # No advantage
            'simulation': f"O({int(np.log2(size))})",
            'optimization': f"O({size}^0.5)",
        }
        return complexities.get(problem_type, "O(?)")

    def _get_classical_complexity(self, problem_type: str, size: int) -> str:
        """Get classical time complexity"""
        complexities = {
            'search': f"O({size})",
            'factoring': f"O(2^({size}^(1/3)))",
            'sort': f"O({size} log {size})",
            'simulation': f"O(2^{size})",
            'optimization': f"O({size}²)",
        }
        return complexities.get(problem_type, "O(?)")

    def _get_quantum_space(self, problem_type: str, size: int) -> str:
        """Get quantum space complexity"""
        return f"O(log {size})" if problem_type in ['search', 'factoring'] else f"O({int(np.log2(size))})"

    def _get_classical_space(self, problem_type: str, size: int) -> str:
        """Get classical space complexity"""
        return f"O({size})"

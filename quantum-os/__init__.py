"""
Quantum Operating System (Quantum OS)

A unified quantum computing framework supporting:
- Google Quantum AI (Willow)
- IBM Quantum (Brisbane, Torino)
- TensorFlow Quantum
- Quantum error correction
- Hybrid quantum-classical computing
- GPU acceleration
- Distributed quantum execution

Author: Brionengine Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Brionengine Team"

# Core components
from .core import (
    QuantumOS,
    QuantumScheduler,
    QuantumResourceManager,
    QuantumVirtualMachine,
    QuantumProgram,
    QuantumGateType,
    UnifiedQuantumResourcePool
)
from .config import QuantumOSConfig, BackendConfig

# Backend interfaces
from .backends import (
    QuantumBackend,
    CirqBackend,
    QiskitBackend,
    TFQBackend,
    QuantumResult,
    QuantumCircuit,
    BackendType,
    ExecutionMode
)

# Error correction
from .error_correction import (
    SurfaceCode,
    StabilizerCode,
    ErrorMitigation
)

# Plugin system
from .plugins import PluginLoader, PluginRegistry

# Security
from .security import CodeObfuscator

# Quantum Algorithms
from .algorithms import (
    GroverSearch,
    ShorFactoring,
    VariationalQuantumEigensolver,
    QuantumApproximateOptimization,
    QuantumFourierTransform,
    AmplitudeAmplification
)

# Classical Computing
from .classical import (
    ClassicalComputingEngine,
    ClassicalAlgorithms,
    HybridOptimizer
)

__all__ = [
    # Core
    'QuantumOS',
    'QuantumScheduler',
    'QuantumResourceManager',
    'QuantumVirtualMachine',
    'QuantumProgram',
    'QuantumGateType',
    'UnifiedQuantumResourcePool',
    'QuantumOSConfig',
    'BackendConfig',

    # Backends
    'QuantumBackend',
    'CirqBackend',
    'QiskitBackend',
    'TFQBackend',
    'QuantumResult',
    'QuantumCircuit',
    'BackendType',
    'ExecutionMode',

    # Error Correction
    'SurfaceCode',
    'StabilizerCode',
    'ErrorMitigation',

    # Plugins
    'PluginLoader',
    'PluginRegistry',

    # Security
    'CodeObfuscator',

    # Algorithms
    'GroverSearch',
    'ShorFactoring',
    'VariationalQuantumEigensolver',
    'QuantumApproximateOptimization',
    'QuantumFourierTransform',
    'AmplitudeAmplification',

    # Classical Computing
    'ClassicalComputingEngine',
    'ClassicalAlgorithms',
    'HybridOptimizer',
]


def create_quantum_os(config_path=None):
    """
    Create and initialize a Quantum OS instance

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        Initialized QuantumOS instance

    Example:
        >>> qos = create_quantum_os()
        >>> circuit = qos.create_circuit(num_qubits=5)
        >>> result = qos.execute(circuit, shots=1024)
    """
    if config_path:
        config = QuantumOSConfig(config_path)
    else:
        config = QuantumOSConfig()

    return QuantumOS(config)


# Quick start helper
def quick_start():
    """
    Quick start guide for Quantum OS

    Prints basic usage information
    """
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              Quantum OS v{__version__}                            ║
    ║        Advanced Quantum Supercomputer Framework               ║
    ╚═══════════════════════════════════════════════════════════════╝

    Quick Start:

    1. Create Quantum OS instance:
       >>> from quantum_os import create_quantum_os
       >>> qos = create_quantum_os()

    2. Check available backends:
       >>> qos.list_backends()

    3. Create a quantum circuit:
       >>> circuit = qos.create_circuit(num_qubits=5)

    4. Execute on quantum hardware:
       >>> result = qos.execute(circuit, shots=1024)

    5. Use error correction:
       >>> from quantum_os import SurfaceCode
       >>> code = SurfaceCode(code_distance=3)
       >>> params = code.get_code_parameters()

    Supported Backends:
    - Google Cirq (Willow simulator)
    - IBM Qiskit (Brisbane 127q, Torino 133q)
    - TensorFlow Quantum (GPU-accelerated)

    For documentation: See README.md
    Repository: https://github.com/Brionengine
    """)


if __name__ == "__main__":
    quick_start()

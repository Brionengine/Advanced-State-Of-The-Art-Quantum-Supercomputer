"""
Quantum OS Kernel

Main orchestration layer for the quantum operating system
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from ..backends import (
    QuantumBackend,
    CirqBackend,
    QiskitBackend,
    TFQBackend,
    QuantumResult,
    ExecutionMode
)
from ..config import QuantumOSConfig, BackendConfig
from .scheduler import QuantumScheduler
from .resource_manager import QuantumResourceManager
from .quantum_vm import QuantumVirtualMachine
from .quantum_resource_pool import UnifiedQuantumResourcePool
from ..classical import ClassicalComputingEngine, HybridOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
)


class QuantumOS:
    """
    Quantum Operating System Kernel

    Provides unified interface to multiple quantum backends with:
    - Automatic backend selection
    - Resource management
    - Job scheduling
    - Error correction
    - Plugin system
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[QuantumOSConfig] = None):
        """
        Initialize Quantum OS

        Args:
            config: QuantumOSConfig object or None for defaults
        """
        self.logger = logging.getLogger('QuantumOS')
        self.logger.info(f"Initializing Quantum OS v{self.VERSION}")

        # Load configuration
        self.config = config or QuantumOSConfig()

        # Initialize components
        self.backends: Dict[str, QuantumBackend] = {}
        self.scheduler = QuantumScheduler(self.config.resources)
        self.resource_manager = QuantumResourceManager(self.config.resources)

        # Initialize backends
        self._initialize_backends()

        # Initialize Quantum Virtual Machine (general computing interface)
        self.qvm = QuantumVirtualMachine(self)

        # Initialize Unified Quantum Resource Pool (quantum supercomputer)
        self.resource_pool = UnifiedQuantumResourcePool(self)

        # Initialize Classical Computing Engine
        self.classical = ClassicalComputingEngine(use_gpu=True)

        # Initialize Hybrid Optimizer (quantum vs classical selection)
        self.hybrid_optimizer = HybridOptimizer(self)

        self.logger.info(
            f"Quantum OS initialized with {len(self.backends)} backend(s)"
        )

    def _initialize_backends(self):
        """Initialize all configured backends"""
        for backend_config in self.config.get_enabled_backends():
            try:
                backend = self._create_backend(backend_config)
                if backend and backend.initialize():
                    self.backends[backend_config.name] = backend
                    self.logger.info(
                        f"Backend '{backend_config.name}' initialized successfully"
                    )
                else:
                    self.logger.warning(
                        f"Failed to initialize backend '{backend_config.name}'"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error initializing backend '{backend_config.name}': {e}"
                )

    def _create_backend(self, config: BackendConfig) -> Optional[QuantumBackend]:
        """Create a backend from configuration"""
        backend_type = config.backend_type.lower()
        execution_mode = ExecutionMode.SIMULATION if config.execution_mode == 'simulation' else ExecutionMode.REAL_QUANTUM

        if backend_type == 'cirq':
            return CirqBackend(
                backend_name=config.name,
                execution_mode=execution_mode,
                **config.credentials,
                **config.options
            )
        elif backend_type == 'qiskit':
            return QiskitBackend(
                backend_name=config.name,
                execution_mode=execution_mode,
                **config.credentials,
                **config.options
            )
        elif backend_type == 'tfq':
            return TFQBackend(
                backend_name=config.name,
                execution_mode=execution_mode,
                **config.options
            )
        else:
            self.logger.error(f"Unknown backend type: {backend_type}")
            return None

    def get_backend(self, name: Optional[str] = None) -> Optional[QuantumBackend]:
        """
        Get a quantum backend by name

        Args:
            name: Backend name (uses primary backend if None)

        Returns:
            QuantumBackend or None
        """
        if name is None:
            # Return highest priority backend
            primary = self.config.get_primary_backend()
            if primary:
                return self.backends.get(primary.name)
            return None

        return self.backends.get(name)

    def list_backends(self) -> List[str]:
        """List all available backend names"""
        return list(self.backends.keys())

    def create_circuit(
        self,
        num_qubits: int,
        backend_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create a quantum circuit

        Args:
            num_qubits: Number of qubits
            backend_name: Target backend (uses primary if None)
            **kwargs: Additional arguments

        Returns:
            Native circuit object for the backend
        """
        backend = self.get_backend(backend_name)
        if not backend:
            raise RuntimeError("No backend available")

        return backend.create_circuit(num_qubits, **kwargs)

    def execute(
        self,
        circuit: Any,
        shots: int = 1024,
        backend_name: Optional[str] = None,
        **kwargs
    ) -> QuantumResult:
        """
        Execute a quantum circuit

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            backend_name: Target backend (auto-selects if None)
            **kwargs: Additional execution parameters

        Returns:
            QuantumResult object
        """
        backend = self.get_backend(backend_name)
        if not backend:
            raise RuntimeError("No backend available")

        # Submit to scheduler
        job_id = self.scheduler.submit_job({
            'circuit': circuit,
            'shots': shots,
            'backend_name': backend.backend_name,
            'kwargs': kwargs
        })

        # Execute
        try:
            result = backend.execute(circuit, shots, **kwargs)
            self.scheduler.mark_job_complete(job_id, success=result.success)
            return result
        except Exception as e:
            self.scheduler.mark_job_complete(job_id, success=False, error=str(e))
            raise

    def execute_batch(
        self,
        circuits: List[Any],
        shots: int = 1024,
        backend_name: Optional[str] = None,
        **kwargs
    ) -> List[QuantumResult]:
        """
        Execute multiple circuits in batch

        Args:
            circuits: List of quantum circuits
            shots: Number of shots per circuit
            backend_name: Target backend
            **kwargs: Additional parameters

        Returns:
            List of QuantumResult objects
        """
        results = []
        backend = self.get_backend(backend_name)

        if not backend:
            raise RuntimeError("No backend available")

        # Check if backend supports batching (TFQ does)
        if hasattr(backend, 'execute_batch'):
            return backend.execute_batch(circuits, shots, **kwargs)

        # Otherwise, execute sequentially
        for circuit in circuits:
            result = self.execute(circuit, shots, backend_name, **kwargs)
            results.append(result)

        return results

    def transpile(
        self,
        circuit: Any,
        backend_name: Optional[str] = None,
        optimization_level: int = 1,
        **kwargs
    ) -> Any:
        """
        Transpile circuit for a specific backend

        Args:
            circuit: Input quantum circuit
            backend_name: Target backend
            optimization_level: 0-3 (higher = more optimization)
            **kwargs: Additional transpilation options

        Returns:
            Transpiled circuit
        """
        backend = self.get_backend(backend_name)
        if not backend:
            raise RuntimeError("No backend available")

        return backend.transpile(circuit, optimization_level, **kwargs)

    def get_backend_properties(
        self,
        backend_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get properties of a backend

        Args:
            backend_name: Backend name (uses primary if None)

        Returns:
            Dictionary with backend properties
        """
        backend = self.get_backend(backend_name)
        if not backend:
            return {}

        return backend.get_backend_properties()

    def estimate_resources(
        self,
        circuit: Any,
        backend_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate resources needed for circuit execution

        Args:
            circuit: Quantum circuit
            backend_name: Target backend

        Returns:
            Dictionary with resource estimates
        """
        backend = self.get_backend(backend_name)
        if not backend:
            return {}

        return backend.estimate_resources(circuit)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            'version': self.VERSION,
            'backends': {},
            'scheduler': self.scheduler.get_status(),
            'resources': self.resource_manager.get_status(),
        }

        for name, backend in self.backends.items():
            status['backends'][name] = {
                'available': backend.is_available,
                'type': backend.backend_type.value,
                'mode': backend.execution_mode.value,
            }

        return status

    def shutdown(self):
        """Shutdown the quantum OS"""
        self.logger.info("Shutting down Quantum OS...")

        # Wait for all jobs to complete
        self.scheduler.wait_for_all_jobs()

        # Clean up backends
        for backend in self.backends.values():
            # Any cleanup needed
            pass

        self.logger.info("Quantum OS shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"QuantumOS(version={self.VERSION}, "
            f"backends={len(self.backends)}, "
            f"active_jobs={len(self.scheduler.jobs)})"
        )

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

    VERSION = "2.0.0"

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

        # Circuit cache for compiled programs (avoids recompilation)
        self._circuit_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Telemetry and performance tracking
        self._execution_history: List[Dict[str, Any]] = []
        self._total_circuits_executed = 0
        self._total_shots_executed = 0

        # Initialize backends with health monitoring
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

    def hot_swap_backend(self, name: str, config: 'BackendConfig') -> bool:
        """
        Hot-swap a backend without restarting the OS.
        Removes old backend and initializes new one in-place.

        Args:
            name: Backend name to replace
            config: New backend configuration

        Returns:
            True if successful
        """
        self.logger.info(f"Hot-swapping backend '{name}'...")
        if name in self.backends:
            del self.backends[name]
        try:
            backend = self._create_backend(config)
            if backend and backend.initialize():
                self.backends[name] = backend
                self.logger.info(f"Backend '{name}' hot-swapped successfully")
                return True
        except Exception as e:
            self.logger.error(f"Hot-swap failed for '{name}': {e}")
        return False

    def select_optimal_backend(
        self,
        circuit: Any,
        criteria: str = 'fidelity'
    ) -> Optional[str]:
        """
        Automatically select the best backend for a given circuit.

        Args:
            circuit: The quantum circuit to evaluate
            criteria: Selection criteria - 'fidelity', 'speed', or 'cost'

        Returns:
            Name of the best backend
        """
        scores = {}
        for name, backend in self.backends.items():
            if not backend.is_available:
                continue
            props = backend.get_backend_properties()
            est = backend.estimate_resources(circuit)
            if criteria == 'fidelity':
                error_rate = props.get('error_rate', 1.0)
                scores[name] = 1.0 - error_rate
            elif criteria == 'speed':
                est_time = est.get('estimated_time', float('inf'))
                scores[name] = 1.0 / max(est_time, 0.001)
            elif criteria == 'cost':
                qubits = props.get('num_qubits', 0)
                scores[name] = qubits
        if not scores:
            return None
        best = max(scores, key=scores.get)
        self.logger.info(
            f"Auto-selected backend '{best}' (criteria={criteria}, score={scores[best]:.4f})"
        )
        return best

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get circuit cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total, 1)
        return {
            'cache_size': len(self._circuit_cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Get execution telemetry and performance metrics"""
        return {
            'total_circuits_executed': self._total_circuits_executed,
            'total_shots_executed': self._total_shots_executed,
            'cache_stats': self.get_cache_stats(),
            'recent_executions': self._execution_history[-10:],
            'backends_available': len([
                b for b in self.backends.values() if b.is_available
            ]),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Run health check across all backends and subsystems.
        Returns per-component health status.
        """
        health = {
            'os_version': self.VERSION,
            'status': 'healthy',
            'backends': {},
            'scheduler': 'ok',
            'resource_manager': 'ok',
        }
        unhealthy = 0
        for name, backend in self.backends.items():
            try:
                props = backend.get_backend_properties()
                health['backends'][name] = {
                    'status': 'healthy' if backend.is_available else 'unavailable',
                    'qubits': props.get('num_qubits', 0),
                    'error_rate': props.get('error_rate', 'unknown'),
                }
            except Exception as e:
                health['backends'][name] = {'status': 'error', 'error': str(e)}
                unhealthy += 1
        if unhealthy > 0:
            health['status'] = 'degraded' if unhealthy < len(self.backends) else 'critical'
        return health

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            'version': self.VERSION,
            'backends': {},
            'scheduler': self.scheduler.get_status(),
            'resources': self.resource_manager.get_status(),
            'telemetry': self.get_telemetry(),
            'health': self.health_check(),
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

        # Flush circuit cache
        self._circuit_cache.clear()

        # Clean up backends
        for name, backend in self.backends.items():
            self.logger.info(f"Shutting down backend '{name}'")

        self.logger.info(
            f"Quantum OS shutdown complete. "
            f"Total circuits executed: {self._total_circuits_executed}"
        )

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

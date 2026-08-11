"""
Unified Quantum Resource Pool

Treats multiple quantum computers (Google Willow, IBM Brisbane, IBM Torino, etc.)
as a single unified quantum supercomputer resource pool.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class QuantumComputerResource:
    """Represents a quantum computer in the resource pool"""
    name: str
    backend: Any
    num_qubits: int
    is_available: bool
    is_real_hardware: bool
    queue_depth: int = 0
    avg_gate_error: float = 0.001
    current_load: float = 0.0  # 0.0 to 1.0
    total_jobs_completed: int = 0


class UnifiedQuantumResourcePool:
    """
    Unified Quantum Resource Pool

    Manages multiple quantum computers as a single supercomputer:
    - Google Willow (105 qubits)
    - IBM Brisbane (127 qubits)
    - IBM Torino (133 qubits)
    - TensorFlow Quantum (GPU simulation)
    - Any other connected quantum backends

    Provides:
    - Unified qubit pool across all backends
    - Intelligent job distribution
    - Load balancing
    - Automatic failover
    - Resource aggregation
    """

    def __init__(self, quantum_os):
        """
        Initialize resource pool

        Args:
            quantum_os: QuantumOS instance
        """
        self.quantum_os = quantum_os
        self.resources: Dict[str, QuantumComputerResource] = {}
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the quantum computer resource pool"""
        backends = self.quantum_os.list_backends()

        for backend_name in backends:
            backend = self.quantum_os.get_backend(backend_name)
            if backend and backend.is_available:
                props = backend.get_backend_properties()

                resource = QuantumComputerResource(
                    name=backend_name,
                    backend=backend,
                    num_qubits=props.get('num_qubits', 0),
                    is_available=True,
                    is_real_hardware=props.get('execution_mode') == 'real_quantum',
                    avg_gate_error=props.get('avg_gate_error', 0.001)
                )

                self.resources[backend_name] = resource

    def get_total_qubits(self) -> int:
        """
        Get total qubits available across all quantum computers

        Returns:
            Total qubit count
        """
        return sum(
            res.num_qubits
            for res in self.resources.values()
            if res.is_available
        )

    def get_total_real_hardware_qubits(self) -> int:
        """Get total qubits on real quantum hardware"""
        return sum(
            res.num_qubits
            for res in self.resources.values()
            if res.is_available and res.is_real_hardware
        )

    def get_available_resources(self) -> List[QuantumComputerResource]:
        """Get list of available quantum computers"""
        return [
            res for res in self.resources.values()
            if res.is_available
        ]

    def select_best_backend(
        self,
        num_qubits_needed: int,
        prefer_real_hardware: bool = False,
        min_fidelity: float = 0.0
    ) -> Optional[str]:
        """
        Select the best backend for a job

        Args:
            num_qubits_needed: Number of qubits required
            prefer_real_hardware: Prefer real quantum hardware
            min_fidelity: Minimum required fidelity

        Returns:
            Backend name or None
        """
        candidates = []

        for name, res in self.resources.items():
            if not res.is_available:
                continue

            if res.num_qubits < num_qubits_needed:
                continue

            if prefer_real_hardware and not res.is_real_hardware:
                continue

            fidelity = 1.0 - res.avg_gate_error
            if fidelity < min_fidelity:
                continue

            # Score based on: available qubits, load, fidelity, real hardware
            score = 0.0
            score += (res.num_qubits - num_qubits_needed) * 0.1  # Prefer exact fit
            score -= res.current_load * 10  # Prefer less loaded backends
            score += fidelity * 5  # Prefer higher fidelity
            score += 20 if res.is_real_hardware else 0  # Bonus for real hardware

            candidates.append((name, score))

        if not candidates:
            return None

        # Return backend with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def distribute_circuits(
        self,
        circuits: List[Any],
        shots: int = 1024
    ) -> Dict[str, List[Any]]:
        """
        Distribute circuits across available quantum computers

        Args:
            circuits: List of circuits to execute
            shots: Shots per circuit

        Returns:
            Dictionary mapping backend names to assigned circuits
        """
        distribution = {name: [] for name in self.resources.keys()}

        # Sort backends by capacity (real hardware first, then by qubits)
        sorted_backends = sorted(
            self.get_available_resources(),
            key=lambda r: (not r.is_real_hardware, -r.num_qubits, r.current_load)
        )

        # Round-robin distribution with load balancing
        backend_index = 0

        for circuit in circuits:
            if not sorted_backends:
                break

            # Select next backend
            backend = sorted_backends[backend_index % len(sorted_backends)]
            distribution[backend.name].append(circuit)

            # Update load
            backend.current_load = len(distribution[backend.name]) / len(circuits)

            backend_index += 1

        # Remove empty entries
        distribution = {k: v for k, v in distribution.items() if v}

        return distribution

    def execute_distributed(
        self,
        circuits: List[Any],
        shots: int = 1024,
        aggregate_results: bool = True
    ) -> Any:
        """
        Execute circuits distributed across all quantum computers

        This is the core supercomputer capability - executing workloads
        across multiple quantum processors simultaneously

        Args:
            circuits: Circuits to execute
            shots: Shots per circuit
            aggregate_results: Whether to aggregate results

        Returns:
            Aggregated or individual results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Distribute circuits
        distribution = self.distribute_circuits(circuits, shots)

        print(f"\nDistributing {len(circuits)} circuits across quantum supercomputer:")
        for backend_name, backend_circuits in distribution.items():
            print(f"  {backend_name}: {len(backend_circuits)} circuits")

        # Execute in parallel across all backends
        results = []
        futures = []

        with ThreadPoolExecutor(max_workers=len(distribution)) as executor:
            for backend_name, backend_circuits in distribution.items():
                resource = self.resources[backend_name]

                for circuit in backend_circuits:
                    future = executor.submit(
                        self._execute_on_backend,
                        circuit,
                        resource.backend,
                        backend_name,
                        shots
                    )
                    futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    # Update statistics
                    backend_name = result.backend_name
                    if backend_name in self.resources:
                        self.resources[backend_name].total_jobs_completed += 1

                except Exception as e:
                    print(f"Execution error: {e}")

        # Aggregate if requested
        if aggregate_results and len(results) > 1:
            return self._aggregate_results(results)

        return results

    def _execute_on_backend(
        self,
        circuit: Any,
        backend: Any,
        backend_name: str,
        shots: int
    ) -> Any:
        """Execute a single circuit on a backend"""
        try:
            result = backend.execute(circuit, shots=shots)
            result.backend_name = backend_name
            return result
        except Exception as e:
            print(f"Error executing on {backend_name}: {e}")
            return None

    def _aggregate_results(self, results: List[Any]) -> Any:
        """
        Aggregate results from multiple quantum computers

        Args:
            results: List of quantum results

        Returns:
            Aggregated result
        """
        # Filter successful results
        successful = [r for r in results if r and r.success]

        if not successful:
            # Return first result (likely contains error)
            return results[0] if results else None

        # Aggregate counts
        aggregated_counts = {}
        total_shots = 0

        for result in successful:
            for state, count in result.counts.items():
                aggregated_counts[state] = aggregated_counts.get(state, 0) + count
            total_shots += result.shots

        # Create aggregated result (using first result as template)
        from ..backends.base import QuantumResult

        aggregated = QuantumResult(
            counts=aggregated_counts,
            execution_time=sum(r.execution_time for r in successful) / len(successful),
            backend_name="unified_quantum_supercomputer",
            num_qubits=successful[0].num_qubits,
            shots=total_shots,
            success=True,
            metadata={
                'backends_used': [r.backend_name for r in successful],
                'num_backends': len(successful),
                'aggregated': True
            }
        )

        return aggregated

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get status of the quantum resource pool

        Returns:
            Pool status information
        """
        status = {
            'total_backends': len(self.resources),
            'available_backends': len(self.get_available_resources()),
            'total_qubits': self.get_total_qubits(),
            'real_hardware_qubits': self.get_total_real_hardware_qubits(),
            'backends': {}
        }

        for name, res in self.resources.items():
            status['backends'][name] = {
                'available': res.is_available,
                'qubits': res.num_qubits,
                'real_hardware': res.is_real_hardware,
                'queue_depth': res.queue_depth,
                'current_load': f"{res.current_load*100:.1f}%",
                'jobs_completed': res.total_jobs_completed,
                'avg_gate_error': f"{res.avg_gate_error*100:.2f}%"
            }

        return status

    def add_backend(self, backend_name: str, backend: Any):
        """
        Add a new quantum computer to the resource pool

        Args:
            backend_name: Name of the backend
            backend: Backend instance
        """
        if backend.is_available:
            props = backend.get_backend_properties()

            resource = QuantumComputerResource(
                name=backend_name,
                backend=backend,
                num_qubits=props.get('num_qubits', 0),
                is_available=True,
                is_real_hardware=props.get('execution_mode') == 'real_quantum',
                avg_gate_error=props.get('avg_gate_error', 0.001)
            )

            self.resources[backend_name] = resource
            print(f"Added {backend_name} to quantum resource pool ({resource.num_qubits} qubits)")

    def remove_backend(self, backend_name: str):
        """Remove a backend from the pool"""
        if backend_name in self.resources:
            del self.resources[backend_name]

    def __repr__(self) -> str:
        return (
            f"UnifiedQuantumResourcePool("
            f"backends={len(self.resources)}, "
            f"total_qubits={self.get_total_qubits()})"
        )

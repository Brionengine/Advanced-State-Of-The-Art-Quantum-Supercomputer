"""
Distributed Quantum Circuit Executor

Distributes quantum jobs across multiple quantum backends and compute nodes
"""

from typing import List, Dict, Any, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class DistributedExecutor:
    """
    Distributed executor for quantum circuits

    Enables:
    - Multi-backend parallel execution
    - Load balancing across quantum processors
    - Fault tolerance and retry logic
    - Result aggregation
    """

    def __init__(self, max_workers: int = 5):
        """
        Initialize distributed executor

        Args:
            max_workers: Maximum parallel workers
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def distribute_circuits(
        self,
        circuits: List[Any],
        backends: List[Any],
        shots: int = 1024
    ) -> List[Any]:
        """
        Distribute circuits across multiple backends

        Args:
            circuits: List of quantum circuits
            backends: List of available backends
            shots: Shots per circuit

        Returns:
            List of results
        """
        if not backends:
            raise ValueError("No backends available")

        # Distribute circuits across backends (round-robin)
        tasks = []
        for i, circuit in enumerate(circuits):
            backend = backends[i % len(backends)]
            tasks.append((circuit, backend, shots))

        # Execute in parallel
        futures = []
        for circuit, backend, shots in tasks:
            future = self.executor.submit(
                self._execute_circuit,
                circuit,
                backend,
                shots
            )
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Circuit execution failed: {e}")
                results.append(None)

        return results

    def _execute_circuit(
        self,
        circuit: Any,
        backend: Any,
        shots: int
    ) -> Any:
        """Execute a single circuit on a backend"""
        try:
            return backend.execute(circuit, shots=shots)
        except Exception as e:
            print(f"Execution error: {e}")
            return None

    def execute_with_retry(
        self,
        circuit: Any,
        backends: List[Any],
        shots: int = 1024,
        max_retries: int = 3
    ) -> Any:
        """
        Execute circuit with automatic retry on failure

        Args:
            circuit: Quantum circuit
            backends: List of backends (fallback order)
            shots: Number of shots
            max_retries: Maximum retry attempts

        Returns:
            Execution result
        """
        for attempt in range(max_retries):
            for backend in backends:
                try:
                    result = backend.execute(circuit, shots=shots)
                    if result and result.success:
                        return result
                except Exception as e:
                    print(f"Attempt {attempt + 1} on {backend.backend_name} failed: {e}")
                    continue

        raise RuntimeError("All execution attempts failed")

    def parallel_parameter_sweep(
        self,
        circuit_generator: callable,
        parameters: List[Any],
        backend: Any,
        shots: int = 1024
    ) -> List[Any]:
        """
        Execute parameter sweep in parallel

        Args:
            circuit_generator: Function that creates circuit from parameters
            parameters: List of parameter sets
            backend: Backend to use
            shots: Shots per circuit

        Returns:
            List of results for each parameter set
        """
        futures = []

        for params in parameters:
            circuit = circuit_generator(params)
            future = self.executor.submit(
                backend.execute,
                circuit,
                shots
            )
            futures.append((params, future))

        # Collect results
        results = []
        for params, future in futures:
            try:
                result = future.result()
                results.append({
                    'parameters': params,
                    'result': result
                })
            except Exception as e:
                print(f"Parameter sweep failed for {params}: {e}")
                results.append({
                    'parameters': params,
                    'result': None,
                    'error': str(e)
                })

        return results

    def aggregate_results(
        self,
        results: List[Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple executions

        Args:
            results: List of QuantumResult objects

        Returns:
            Aggregated counts and statistics
        """
        aggregated = {
            'total_shots': 0,
            'combined_counts': {},
            'success_rate': 0.0,
            'avg_execution_time': 0.0
        }

        successful_results = [r for r in results if r and r.success]

        if not successful_results:
            return aggregated

        # Aggregate counts
        for result in successful_results:
            aggregated['total_shots'] += result.shots

            for state, count in result.counts.items():
                aggregated['combined_counts'][state] = \
                    aggregated['combined_counts'].get(state, 0) + count

        # Calculate statistics
        aggregated['success_rate'] = len(successful_results) / len(results) * 100

        avg_time = sum(r.execution_time for r in successful_results) / len(successful_results)
        aggregated['avg_execution_time'] = avg_time

        return aggregated

    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

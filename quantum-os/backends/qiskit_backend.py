"""
Qiskit Backend Implementation for IBM Quantum

Supports:
- IBM Quantum processors (Brisbane 127q, Torino 133q)
- Aer simulator (local simulation)
- Qiskit Runtime for optimized execution
"""

import time
from typing import Any, Dict, Optional, List
import numpy as np

try:
    from qiskit import QuantumCircuit as QiskitCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .base import (
    QuantumBackend,
    QuantumCircuit,
    QuantumResult,
    BackendType,
    ExecutionMode
)


class QiskitBackend(QuantumBackend):
    """
    Qiskit-based quantum backend for IBM Quantum

    Supports:
    - IBM Brisbane (127 qubits)
    - IBM Torino (133 qubits)
    - Aer Simulator (local simulation)
    - Qiskit Runtime Service
    """

    def __init__(
        self,
        backend_name: str = "aer_simulator",
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        api_token: Optional[str] = None,
        instance: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Qiskit backend

        Args:
            backend_name: Backend name ('aer_simulator', 'ibm_brisbane', 'ibm_torino')
            execution_mode: Simulation or real quantum
            api_token: IBM Quantum API token
            instance: IBM Quantum instance (hub/group/project)
        """
        super().__init__(
            backend_name=backend_name,
            backend_type=BackendType.QISKIT,
            execution_mode=execution_mode,
            **kwargs
        )

        self.api_token = api_token
        self.instance = instance or "ibm-q/open/main"
        self.use_runtime = kwargs.get('use_runtime', True)

        # Will be initialized in initialize()
        self._service = None
        self._session = None

    def initialize(self) -> bool:
        """Initialize the Qiskit backend"""
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is not installed. "
                "Install with: pip install qiskit qiskit-aer qiskit-ibm-runtime"
            )

        try:
            if self.execution_mode == ExecutionMode.SIMULATION:
                # Initialize Aer simulator
                self._native_backend = AerSimulator()

            elif self.execution_mode == ExecutionMode.REAL_QUANTUM:
                # Initialize IBM Quantum service
                if not self.api_token:
                    # Try to load saved credentials
                    try:
                        self._service = QiskitRuntimeService()
                    except:
                        raise ValueError(
                            "IBM Quantum API token required for real quantum execution. "
                            "Provide api_token or save credentials with "
                            "QiskitRuntimeService.save_account()"
                        )
                else:
                    # Save and load credentials
                    QiskitRuntimeService.save_account(
                        channel="ibm_quantum",
                        token=self.api_token,
                        instance=self.instance,
                        overwrite=True
                    )
                    self._service = QiskitRuntimeService()

                # Get the specified backend
                self._native_backend = self._service.backend(self.backend_name)

            self._is_initialized = True
            return True

        except Exception as e:
            print(f"Error initializing Qiskit backend: {e}")
            self._is_initialized = False
            return False

    def create_circuit(
        self,
        num_qubits: int,
        num_classical_bits: Optional[int] = None
    ) -> QiskitCircuit:
        """
        Create a Qiskit quantum circuit

        Args:
            num_qubits: Number of qubits
            num_classical_bits: Number of classical bits (defaults to num_qubits)

        Returns:
            QiskitCircuit object
        """
        if num_classical_bits is None:
            num_classical_bits = num_qubits

        circuit = QiskitCircuit(num_qubits, num_classical_bits)
        return circuit

    def execute(
        self,
        circuit: QiskitCircuit,
        shots: int = 1024,
        **kwargs
    ) -> QuantumResult:
        """
        Execute a Qiskit circuit

        Args:
            circuit: Qiskit circuit to execute
            shots: Number of measurement shots
            **kwargs: Additional execution parameters

        Returns:
            QuantumResult with execution results
        """
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Transpile circuit for the backend
            transpiled_circuit = transpile(
                circuit,
                backend=self._native_backend,
                optimization_level=kwargs.get('optimization_level', 1)
            )

            # Execute based on mode
            if self.execution_mode == ExecutionMode.SIMULATION:
                # Run on Aer simulator
                job = self._native_backend.run(transpiled_circuit, shots=shots)
                result = job.result()
                counts = result.get_counts()

                # Get statevector if available
                statevector = None
                try:
                    # Create circuit without measurements
                    circuit_no_measure = circuit.remove_final_measurements(inplace=False)
                    sv = Statevector(circuit_no_measure)
                    statevector = np.array(sv.data)
                except:
                    statevector = None

            elif self.execution_mode == ExecutionMode.REAL_QUANTUM:
                # Use Qiskit Runtime for optimized execution
                if self.use_runtime:
                    with Session(service=self._service, backend=self._native_backend) as session:
                        sampler = Sampler(session=session)
                        job = sampler.run(transpiled_circuit, shots=shots)
                        result = job.result()
                        # Extract counts from Runtime result
                        quasi_dists = result.quasi_dists[0]
                        # Convert quasi-distribution to counts
                        counts = {
                            format(state, f'0{circuit.num_qubits}b'): int(prob * shots)
                            for state, prob in quasi_dists.items()
                        }
                else:
                    # Direct execution (legacy)
                    job = self._native_backend.run(transpiled_circuit, shots=shots)
                    result = job.result()
                    counts = result.get_counts()

                statevector = None  # Not available on real hardware

            execution_time = time.time() - start_time

            return QuantumResult(
                counts=counts,
                statevector=statevector,
                execution_time=execution_time,
                backend_name=self.backend_name,
                num_qubits=circuit.num_qubits,
                shots=shots,
                success=True,
                metadata={
                    'backend_type': 'qiskit',
                    'execution_mode': self.execution_mode.value,
                    'transpiled_depth': transpiled_circuit.depth(),
                    'transpiled_gates': dict(transpiled_circuit.count_ops())
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return QuantumResult(
                counts={},
                execution_time=execution_time,
                backend_name=self.backend_name,
                num_qubits=circuit.num_qubits if hasattr(circuit, 'num_qubits') else 0,
                shots=shots,
                success=False,
                error_message=str(e),
                metadata={'backend_type': 'qiskit'}
            )

    def transpile(
        self,
        circuit: QiskitCircuit,
        optimization_level: int = 1,
        **kwargs
    ) -> QiskitCircuit:
        """
        Transpile circuit for IBM Quantum backend

        Args:
            circuit: Input Qiskit circuit
            optimization_level: 0-3 (higher = more optimization)
            **kwargs: Additional transpilation options

        Returns:
            Transpiled Qiskit circuit
        """
        return transpile(
            circuit,
            backend=self._native_backend,
            optimization_level=optimization_level,
            **kwargs
        )

    def get_backend_properties(self) -> Dict[str, Any]:
        """Get properties of the IBM Quantum backend"""
        props = {
            'backend_name': self.backend_name,
            'backend_type': self.backend_type.value,
            'execution_mode': self.execution_mode.value,
            'is_available': self.is_available,
        }

        if self._native_backend:
            config = self._native_backend.configuration()
            props.update({
                'num_qubits': config.n_qubits,
                'basis_gates': config.basis_gates,
                'coupling_map': config.coupling_map.get_edges() if hasattr(config, 'coupling_map') and config.coupling_map else None,
                'max_shots': config.max_shots,
                'max_experiments': config.max_experiments if hasattr(config, 'max_experiments') else 1,
            })

            # Add backend-specific properties for real hardware
            if self.execution_mode == ExecutionMode.REAL_QUANTUM:
                try:
                    backend_props = self._native_backend.properties()
                    if backend_props:
                        # Get average gate error rates
                        gate_errors = []
                        for gate in backend_props.gates:
                            for param in gate.parameters:
                                if param.name == 'gate_error':
                                    gate_errors.append(param.value)

                        if gate_errors:
                            props['avg_gate_error'] = np.mean(gate_errors)

                        # Get average readout error
                        readout_errors = [
                            qubit.readout_error
                            for qubit in backend_props.qubits
                            if hasattr(qubit, 'readout_error') and qubit.readout_error is not None
                        ]
                        if readout_errors:
                            props['avg_readout_error'] = np.mean(readout_errors)

                        # Get T1 and T2 coherence times
                        t1_times = []
                        t2_times = []
                        for qubit in backend_props.qubits:
                            for prop in qubit:
                                if prop.name == 'T1':
                                    t1_times.append(prop.value)
                                elif prop.name == 'T2':
                                    t2_times.append(prop.value)

                        if t1_times:
                            props['avg_t1'] = np.mean(t1_times)
                        if t2_times:
                            props['avg_t2'] = np.mean(t2_times)

                except Exception as e:
                    print(f"Could not fetch backend properties: {e}")

        return props

    def convert_to_universal(self, native_circuit: QiskitCircuit) -> QuantumCircuit:
        """Convert Qiskit circuit to universal representation"""
        gates = []

        for instruction in native_circuit.data:
            gate_info = {
                'type': instruction.operation.name,
                'qubits': [native_circuit.find_bit(q).index for q in instruction.qubits],
                'params': [float(p) for p in instruction.operation.params]
            }
            gates.append(gate_info)

        return QuantumCircuit(
            num_qubits=native_circuit.num_qubits,
            num_classical_bits=native_circuit.num_clbits,
            gates=gates,
            name=native_circuit.name
        )

    def estimate_resources(self, circuit: QiskitCircuit) -> Dict[str, Any]:
        """Estimate resources for circuit execution"""
        # Transpile to get realistic estimates
        try:
            transpiled = self.transpile(circuit, optimization_level=1)

            gate_counts = transpiled.count_ops()
            depth = transpiled.depth()

            # Estimate error rate based on backend properties
            props = self.get_backend_properties()
            avg_gate_error = props.get('avg_gate_error', 0.001)

            total_gates = sum(gate_counts.values())
            estimated_error_rate = 1 - (1 - avg_gate_error) ** total_gates

            # Estimate execution time (very rough)
            # Assume ~100ns per gate + overhead
            estimated_time = total_gates * 100e-9 + 0.001  # 1ms overhead

            return {
                'num_qubits': circuit.num_qubits,
                'depth': depth,
                'gate_count': total_gates,
                'gate_breakdown': gate_counts,
                'estimated_time': estimated_time,
                'estimated_error_rate': estimated_error_rate
            }

        except Exception as e:
            return {
                'num_qubits': circuit.num_qubits,
                'depth': circuit.depth(),
                'gate_count': sum(circuit.count_ops().values()),
                'estimated_time': 0.0,
                'estimated_error_rate': 0.0,
                'error': str(e)
            }

    def get_available_backends(self) -> List[str]:
        """Get list of available IBM Quantum backends"""
        if self._service:
            backends = self._service.backends()
            return [backend.name for backend in backends]
        return []

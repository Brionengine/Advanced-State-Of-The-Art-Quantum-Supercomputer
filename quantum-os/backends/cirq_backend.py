"""
Cirq Backend Implementation for Google Quantum (Willow)

Supports:
- Cirq simulator (local simulation)
- Google Quantum Engine (real hardware when available)
- Willow quantum processor (via simulator for now)
"""

import time
from typing import Any, Dict, Optional, List
import numpy as np

try:
    import cirq
    import cirq_google
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

from .base import (
    QuantumBackend,
    QuantumCircuit,
    QuantumResult,
    BackendType,
    ExecutionMode
)


class CirqBackend(QuantumBackend):
    """
    Cirq-based quantum backend for Google Quantum AI

    This backend supports:
    - Local simulation via Cirq simulator
    - Google Willow processor (simulated)
    - Google Quantum Engine (when credentials available)
    """

    def __init__(
        self,
        backend_name: str = "cirq_simulator",
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        device_name: Optional[str] = None,
        project_id: Optional[str] = None,
        processor_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Cirq backend

        Args:
            backend_name: Name of the backend
            execution_mode: Simulation or real quantum
            device_name: Google device name (e.g., 'rainbow', 'weber')
            project_id: Google Cloud project ID
            processor_id: Quantum processor ID
        """
        super().__init__(
            backend_name=backend_name,
            backend_type=BackendType.CIRQ,
            execution_mode=execution_mode,
            **kwargs
        )

        self.device_name = device_name or "willow_simulated"
        self.project_id = project_id
        self.processor_id = processor_id
        self.simulator_type = kwargs.get('simulator_type', 'density_matrix')

        # Will be initialized in initialize()
        self._simulator = None
        self._device = None
        self._engine = None

    def initialize(self) -> bool:
        """Initialize the Cirq backend"""
        if not CIRQ_AVAILABLE:
            raise ImportError(
                "Cirq is not installed. "
                "Install with: pip install cirq cirq-google"
            )

        try:
            if self.execution_mode == ExecutionMode.SIMULATION:
                # Initialize simulator
                if self.simulator_type == 'density_matrix':
                    self._simulator = cirq.DensityMatrixSimulator()
                elif self.simulator_type == 'state_vector':
                    self._simulator = cirq.Simulator()
                else:
                    # Default to density matrix for better noise modeling
                    self._simulator = cirq.DensityMatrixSimulator()

                # Create a simulated Willow-like device
                self._device = self._create_willow_simulated_device()
                self._native_backend = self._simulator

            elif self.execution_mode == ExecutionMode.REAL_QUANTUM:
                # Initialize Google Quantum Engine
                if not self.project_id:
                    raise ValueError(
                        "project_id required for real quantum execution"
                    )

                # This requires Google Cloud credentials
                self._engine = cirq_google.Engine(
                    project_id=self.project_id,
                    proto_version=cirq_google.ProtoVersion.V2
                )

                if self.processor_id:
                    processor = self._engine.get_processor(self.processor_id)
                    self._device = processor.get_device()
                    self._native_backend = processor
                else:
                    # Use default processor
                    self._device = cirq_google.Sycamore
                    self._native_backend = self._engine

            self._is_initialized = True
            return True

        except Exception as e:
            print(f"Error initializing Cirq backend: {e}")
            self._is_initialized = False
            return False

    def _create_willow_simulated_device(self) -> cirq.Device:
        """
        Create a simulated device with Willow-like properties

        Willow characteristics:
        - 105 qubits (announced)
        - High connectivity
        - Error rates: ~0.1% for two-qubit gates
        """
        # For now, use an unrestricted device
        # In future, this can be configured with actual Willow topology
        return cirq.UNCONSTRAINED_DEVICE

    def create_circuit(
        self,
        num_qubits: int,
        num_classical_bits: Optional[int] = None
    ) -> cirq.Circuit:
        """
        Create a Cirq circuit

        Args:
            num_qubits: Number of qubits
            num_classical_bits: Ignored for Cirq (measurements handled differently)

        Returns:
            cirq.Circuit object
        """
        # Create qubits - using GridQubits for realistic topology
        qubits = [cirq.GridQubit(i // 10, i % 10) for i in range(num_qubits)]
        circuit = cirq.Circuit()

        # Store qubits as circuit metadata
        circuit._qubits = qubits

        return circuit

    def execute(
        self,
        circuit: cirq.Circuit,
        shots: int = 1024,
        **kwargs
    ) -> QuantumResult:
        """
        Execute a Cirq circuit

        Args:
            circuit: Cirq circuit to execute
            shots: Number of measurement repetitions
            **kwargs: Additional execution parameters

        Returns:
            QuantumResult with execution results
        """
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Add measurements if not present
            if not any(isinstance(op.gate, cirq.MeasurementGate)
                      for op in circuit.all_operations()):
                # Add measurement to all qubits
                qubits = sorted(circuit.all_qubits())
                circuit = circuit + cirq.measure(*qubits, key='result')

            # Run simulation or real execution
            if self.execution_mode == ExecutionMode.SIMULATION:
                result = self._simulator.run(circuit, repetitions=shots)
            else:
                # Real quantum execution via Google Engine
                job = self._engine.run(
                    circuit,
                    repetitions=shots,
                    processor_id=self.processor_id or 'rainbow'
                )
                result = job.results()[0]

            # Extract results
            measurements = result.measurements

            # Convert to counts dictionary
            counts = {}
            for key in measurements:
                bitstrings = measurements[key]
                for bitstring in bitstrings:
                    key_str = ''.join(str(b) for b in bitstring)
                    counts[key_str] = counts.get(key_str, 0) + 1

            # Get statevector if available (simulation only)
            statevector = None
            if self.execution_mode == ExecutionMode.SIMULATION:
                try:
                    # Remove measurements for statevector calculation
                    circuit_no_measure = circuit[:-1] if circuit else circuit
                    sim_result = cirq.Simulator().simulate(circuit_no_measure)
                    statevector = sim_result.final_state_vector
                except:
                    statevector = None

            execution_time = time.time() - start_time

            # Get number of qubits
            num_qubits = len(sorted(circuit.all_qubits()))

            return QuantumResult(
                counts=counts,
                statevector=statevector,
                execution_time=execution_time,
                backend_name=self.backend_name,
                num_qubits=num_qubits,
                shots=shots,
                success=True,
                metadata={
                    'backend_type': 'cirq',
                    'execution_mode': self.execution_mode.value,
                    'device': str(self._device)
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return QuantumResult(
                counts={},
                execution_time=execution_time,
                backend_name=self.backend_name,
                num_qubits=0,
                shots=shots,
                success=False,
                error_message=str(e),
                metadata={'backend_type': 'cirq'}
            )

    def transpile(
        self,
        circuit: cirq.Circuit,
        optimization_level: int = 1,
        **kwargs
    ) -> cirq.Circuit:
        """
        Transpile/optimize circuit for Cirq backend

        Args:
            circuit: Input Cirq circuit
            optimization_level: 0-3 (higher = more optimization)
            **kwargs: Additional options

        Returns:
            Optimized Cirq circuit
        """
        if optimization_level == 0:
            return circuit

        optimized = circuit

        if optimization_level >= 1:
            # Basic optimization - merge single-qubit gates
            optimized = cirq.merge_single_qubit_gates_to_phxz(optimized)

        if optimization_level >= 2:
            # Drop negligible operations
            optimized = cirq.drop_negligible_operations(optimized)
            # Drop empty moments
            optimized = cirq.drop_empty_moments(optimized)

        if optimization_level >= 3:
            # Aggressive optimization
            optimized = cirq.optimize_for_target_gateset(
                optimized,
                gateset=cirq_google.SycamoreTargetGateset()
            )

        return optimized

    def get_backend_properties(self) -> Dict[str, Any]:
        """Get properties of the Cirq backend"""
        props = {
            'backend_name': self.backend_name,
            'backend_type': self.backend_type.value,
            'execution_mode': self.execution_mode.value,
            'simulator_type': self.simulator_type,
            'device_name': self.device_name,
            'is_available': self.is_available,
        }

        if self.execution_mode == ExecutionMode.SIMULATION:
            # Simulated Willow properties
            props.update({
                'num_qubits': 105,  # Willow has 105 qubits
                'basis_gates': ['rx', 'ry', 'rz', 'cz', 'sqrt_iswap'],
                'coupling_map': 'all_to_all',  # Simplified
                'gate_error_1q': 0.001,  # 0.1%
                'gate_error_2q': 0.001,  # 0.1%
                'readout_error': 0.001,
                'coherence_time_t1': 100e-6,  # 100 microseconds
                'coherence_time_t2': 150e-6,  # 150 microseconds
            })
        elif self._device:
            # Real device properties
            props.update({
                'num_qubits': len(self._device.qubits) if hasattr(self._device, 'qubits') else 0,
                'device': str(self._device),
            })

        return props

    def convert_to_universal(self, native_circuit: cirq.Circuit) -> QuantumCircuit:
        """Convert Cirq circuit to universal representation"""
        num_qubits = len(sorted(native_circuit.all_qubits()))

        gates = []
        for moment in native_circuit:
            for op in moment:
                gate_info = {
                    'type': str(op.gate),
                    'qubits': [q.row * 10 + q.col for q in op.qubits],
                    'params': {}
                }
                gates.append(gate_info)

        return QuantumCircuit(
            num_qubits=num_qubits,
            gates=gates,
            name=f"cirq_circuit_{id(native_circuit)}"
        )

    def estimate_resources(self, circuit: cirq.Circuit) -> Dict[str, Any]:
        """Estimate resources for circuit execution"""
        num_qubits = len(sorted(circuit.all_qubits()))
        depth = len(circuit)
        gate_count = sum(len(moment) for moment in circuit)

        # Estimate error rate (simplified)
        # Assume 0.1% per 2-qubit gate, 0.01% per 1-qubit gate
        two_qubit_gates = sum(
            1 for op in circuit.all_operations()
            if len(op.qubits) == 2
        )
        one_qubit_gates = gate_count - two_qubit_gates

        estimated_error_rate = (
            two_qubit_gates * 0.001 +
            one_qubit_gates * 0.0001
        )

        # Estimate execution time (simplified)
        # Assume 100ns per gate
        estimated_time = gate_count * 100e-9

        return {
            'num_qubits': num_qubits,
            'depth': depth,
            'gate_count': gate_count,
            '1q_gates': one_qubit_gates,
            '2q_gates': two_qubit_gates,
            'estimated_time': estimated_time,
            'estimated_error_rate': estimated_error_rate
        }

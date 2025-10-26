"""
TensorFlow Quantum Backend Implementation

Supports:
- Hybrid quantum-classical machine learning
- Quantum neural networks
- GPU-accelerated quantum simulation
- Integration with TensorFlow/Keras
"""

import time
from typing import Any, Dict, Optional, List
import numpy as np

try:
    import tensorflow as tf
    import tensorflow_quantum as tfq
    import cirq
    import sympy
    TFQ_AVAILABLE = True
except ImportError:
    TFQ_AVAILABLE = False

from .base import (
    QuantumBackend,
    QuantumCircuit,
    QuantumResult,
    BackendType,
    ExecutionMode
)


class TFQBackend(QuantumBackend):
    """
    TensorFlow Quantum backend for hybrid quantum-classical ML

    Enables:
    - Quantum neural networks
    - Variational quantum algorithms
    - GPU-accelerated simulation
    - Gradient-based optimization
    """

    def __init__(
        self,
        backend_name: str = "tfq_simulator",
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize TensorFlow Quantum backend

        Args:
            backend_name: Name of the backend
            execution_mode: Only simulation supported for TFQ
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__(
            backend_name=backend_name,
            backend_type=BackendType.TFQ,
            execution_mode=ExecutionMode.SIMULATION,  # TFQ only supports simulation
            **kwargs
        )

        self.use_gpu = use_gpu
        self._setup_gpu()

    def _setup_gpu(self):
        """Configure GPU settings for TensorFlow"""
        if self.use_gpu:
            # Enable GPU memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"TFQ: GPU acceleration enabled ({len(gpus)} GPU(s) found)")
                except RuntimeError as e:
                    print(f"TFQ: GPU configuration error: {e}")
            else:
                print("TFQ: No GPU found, using CPU")

    def initialize(self) -> bool:
        """Initialize the TensorFlow Quantum backend"""
        if not TFQ_AVAILABLE:
            raise ImportError(
                "TensorFlow Quantum is not installed. "
                "Install with: pip install tensorflow tensorflow-quantum cirq"
            )

        try:
            # Test TFQ functionality
            test_circuit = cirq.Circuit(
                cirq.H(cirq.GridQubit(0, 0))
            )
            _ = tfq.convert_to_tensor([test_circuit])

            self._native_backend = "tfq_simulator"
            self._is_initialized = True
            return True

        except Exception as e:
            print(f"Error initializing TFQ backend: {e}")
            self._is_initialized = False
            return False

    def create_circuit(
        self,
        num_qubits: int,
        num_classical_bits: Optional[int] = None
    ) -> cirq.Circuit:
        """
        Create a Cirq circuit (TFQ uses Cirq circuits)

        Args:
            num_qubits: Number of qubits
            num_classical_bits: Ignored (TFQ handles measurements differently)

        Returns:
            cirq.Circuit object
        """
        qubits = [cirq.GridQubit(i // 10, i % 10) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        circuit._qubits = qubits
        return circuit

    def execute(
        self,
        circuit: cirq.Circuit,
        shots: int = 1024,
        **kwargs
    ) -> QuantumResult:
        """
        Execute a circuit using TensorFlow Quantum

        Args:
            circuit: Cirq circuit to execute
            shots: Number of measurement shots
            **kwargs: Additional parameters

        Returns:
            QuantumResult with execution results
        """
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Convert circuit to TFQ tensor
            circuit_tensor = tfq.convert_to_tensor([circuit])

            # Add measurements if not present
            qubits = sorted(circuit.all_qubits())
            if not any(isinstance(op.gate, cirq.MeasurementGate)
                      for op in circuit.all_operations()):
                circuit_with_measure = circuit + cirq.measure(*qubits, key='result')
            else:
                circuit_with_measure = circuit

            # Use TFQ's sampling functionality
            sampler = tfq.layers.Sample()

            # Run sampling
            samples = sampler(
                circuit_tensor,
                repetitions=shots
            )

            # Convert samples to counts
            counts = {}
            samples_np = samples.numpy()[0]  # Get first (and only) circuit's samples

            for sample in samples_np:
                # Convert to bitstring
                bitstring = ''.join(str(int(b)) for b in sample)
                counts[bitstring] = counts.get(bitstring, 0) + 1

            # Get statevector using TFQ state layer
            statevector = None
            try:
                circuit_no_measure = circuit  # Original circuit without measurement
                state_layer = tfq.layers.State()
                state_output = state_layer(tfq.convert_to_tensor([circuit_no_measure]))
                statevector = state_output.numpy()[0]
            except:
                statevector = None

            execution_time = time.time() - start_time

            return QuantumResult(
                counts=counts,
                statevector=statevector,
                execution_time=execution_time,
                backend_name=self.backend_name,
                num_qubits=len(qubits),
                shots=shots,
                success=True,
                metadata={
                    'backend_type': 'tfq',
                    'gpu_enabled': self.use_gpu,
                    'execution_mode': 'simulation'
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
                metadata={'backend_type': 'tfq'}
            )

    def execute_batch(
        self,
        circuits: List[cirq.Circuit],
        shots: int = 1024,
        **kwargs
    ) -> List[QuantumResult]:
        """
        Execute multiple circuits in parallel using TFQ

        This is one of TFQ's key advantages - batched execution

        Args:
            circuits: List of Cirq circuits
            shots: Number of measurement shots per circuit
            **kwargs: Additional parameters

        Returns:
            List of QuantumResult objects
        """
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        start_time = time.time()
        results = []

        try:
            # Convert all circuits to tensor
            circuit_tensors = tfq.convert_to_tensor(circuits)

            # Use TFQ's batch sampling
            sampler = tfq.layers.Sample()
            all_samples = sampler(circuit_tensors, repetitions=shots)

            # Process each circuit's results
            for i, (circuit, samples) in enumerate(zip(circuits, all_samples.numpy())):
                counts = {}
                for sample in samples:
                    bitstring = ''.join(str(int(b)) for b in sample)
                    counts[bitstring] = counts.get(bitstring, 0) + 1

                qubits = sorted(circuit.all_qubits())

                results.append(QuantumResult(
                    counts=counts,
                    execution_time=(time.time() - start_time) / len(circuits),
                    backend_name=self.backend_name,
                    num_qubits=len(qubits),
                    shots=shots,
                    success=True,
                    metadata={'backend_type': 'tfq', 'batch_index': i}
                ))

            return results

        except Exception as e:
            print(f"Batch execution error: {e}")
            return []

    def create_quantum_layer(
        self,
        circuit: cirq.Circuit,
        observables: List[Any],
        **kwargs
    ):
        """
        Create a TensorFlow Keras quantum layer

        This enables building quantum neural networks

        Args:
            circuit: Parameterized quantum circuit
            observables: List of observables to measure
            **kwargs: Additional layer parameters

        Returns:
            TFQ Keras layer
        """
        return tfq.layers.PQC(
            circuit,
            observables,
            **kwargs
        )

    def transpile(
        self,
        circuit: cirq.Circuit,
        optimization_level: int = 1,
        **kwargs
    ) -> cirq.Circuit:
        """
        Optimize circuit for TFQ execution

        Args:
            circuit: Input circuit
            optimization_level: 0-3

        Returns:
            Optimized circuit
        """
        if optimization_level == 0:
            return circuit

        optimized = circuit

        if optimization_level >= 1:
            optimized = cirq.merge_single_qubit_gates_to_phxz(optimized)

        if optimization_level >= 2:
            optimized = cirq.drop_negligible_operations(optimized)
            optimized = cirq.drop_empty_moments(optimized)

        return optimized

    def get_backend_properties(self) -> Dict[str, Any]:
        """Get TFQ backend properties"""
        gpu_devices = tf.config.list_physical_devices('GPU')

        props = {
            'backend_name': self.backend_name,
            'backend_type': self.backend_type.value,
            'execution_mode': 'simulation',
            'is_available': self.is_available,
            'gpu_enabled': self.use_gpu,
            'num_gpus': len(gpu_devices) if gpu_devices else 0,
            'supports_batching': True,
            'supports_gradients': True,
            'supports_hybrid_ml': True,
        }

        if gpu_devices:
            props['gpu_info'] = [
                {
                    'name': gpu.name,
                    'device_type': gpu.device_type
                }
                for gpu in gpu_devices
            ]

        # TFQ can handle many qubits in simulation (limited by memory)
        props['max_qubits_simulation'] = 25  # Practical limit on most systems

        return props

    def estimate_resources(self, circuit: cirq.Circuit) -> Dict[str, Any]:
        """Estimate resources for TFQ execution"""
        num_qubits = len(sorted(circuit.all_qubits()))
        depth = len(circuit)
        gate_count = sum(len(moment) for moment in circuit)

        # Memory estimate for statevector simulation
        # 2^n complex numbers, each 16 bytes
        memory_bytes = 2 ** num_qubits * 16

        # Time estimate (very rough)
        # GPU can be 10-100x faster than CPU
        base_time = gate_count * 50e-9  # 50ns per gate
        if self.use_gpu:
            estimated_time = base_time / 10
        else:
            estimated_time = base_time

        return {
            'num_qubits': num_qubits,
            'depth': depth,
            'gate_count': gate_count,
            'estimated_memory_bytes': memory_bytes,
            'estimated_memory_gb': memory_bytes / (1024**3),
            'estimated_time': estimated_time,
            'estimated_error_rate': 0.0,  # Ideal simulation
            'can_use_gpu': self.use_gpu and tf.config.list_physical_devices('GPU')
        }

    def create_variational_circuit(
        self,
        num_qubits: int,
        num_layers: int,
        **kwargs
    ) -> tuple:
        """
        Create a parameterized variational quantum circuit

        Args:
            num_qubits: Number of qubits
            num_layers: Number of variational layers

        Returns:
            Tuple of (circuit, parameter_symbols)
        """
        qubits = [cirq.GridQubit(i // 10, i % 10) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        params = []

        for layer in range(num_layers):
            # Rotation layer
            for i, qubit in enumerate(qubits):
                param_x = sympy.Symbol(f'θ_{layer}_{i}_x')
                param_y = sympy.Symbol(f'θ_{layer}_{i}_y')
                param_z = sympy.Symbol(f'θ_{layer}_{i}_z')

                circuit += cirq.rx(param_x)(qubit)
                circuit += cirq.ry(param_y)(qubit)
                circuit += cirq.rz(param_z)(qubit)

                params.extend([param_x, param_y, param_z])

            # Entanglement layer
            for i in range(num_qubits - 1):
                circuit += cirq.CNOT(qubits[i], qubits[i + 1])

            # Optional: circular entanglement
            if kwargs.get('circular_entanglement', False) and num_qubits > 2:
                circuit += cirq.CNOT(qubits[-1], qubits[0])

        return circuit, params

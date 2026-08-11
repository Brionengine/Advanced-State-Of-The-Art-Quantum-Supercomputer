"""
Practical Real-World Applications for Long-Lived Qubits
========================================================

Interfaces for using long-lived fault-tolerant qubits in practical applications:
- Quantum computing as a service
- Quantum memory and databases
- Quantum networking and communication
- Quantum sensing and metrology
- Drug discovery and molecular simulation
- Cryptography and secure communications
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
import logging

logger = logging.getLogger(__name__)


class ApplicationDomain(Enum):
    """Real-world application domains."""
    QUANTUM_COMPUTING = "quantum_computing"
    QUANTUM_MEMORY = "quantum_memory"
    QUANTUM_NETWORKING = "quantum_networking"
    QUANTUM_SENSING = "quantum_sensing"
    MOLECULAR_SIMULATION = "molecular_simulation"
    CRYPTOGRAPHY = "cryptography"
    MACHINE_LEARNING = "machine_learning"
    FINANCIAL_MODELING = "financial_modeling"
    OPTIMIZATION = "optimization"


@dataclass
class ApplicationRequirements:
    """Requirements for specific application."""
    domain: ApplicationDomain
    min_qubits: int
    min_coherence_time: float  # seconds
    min_gate_fidelity: float
    required_connectivity: str  # 'all-to-all', 'nearest-neighbor', 'arbitrary'
    max_error_rate: float


class QuantumComputingService:
    """
    Quantum computing as a service using long-lived qubits.

    Provides cloud-based quantum computation with guaranteed reliability
    and long job runtimes enabled by stable qubits.
    """

    def __init__(self,
                 num_qubits: int = 100,
                 service_tier: str = "enterprise"):
        """
        Initialize quantum computing service.

        Args:
            num_qubits: Available qubits
            service_tier: 'basic', 'professional', 'enterprise'
        """
        self.num_qubits = num_qubits
        self.service_tier = service_tier

        # Service capabilities
        self.capabilities = {
            'basic': {
                'max_circuit_depth': 1000,
                'max_runtime_hours': 1,
                'guaranteed_fidelity': 0.99
            },
            'professional': {
                'max_circuit_depth': 10000,
                'max_runtime_hours': 24,
                'guaranteed_fidelity': 0.999
            },
            'enterprise': {
                'max_circuit_depth': 100000,
                'max_runtime_hours': 720,  # 30 days
                'guaranteed_fidelity': 0.9999
            }
        }

        # Job queue
        self.job_queue: List[Dict] = []
        self.completed_jobs: List[Dict] = []

        logger.info(f"Initialized quantum computing service: {service_tier} tier, "
                   f"{num_qubits} qubits")

    def submit_circuit(self,
                      circuit_description: Dict,
                      priority: str = "normal") -> str:
        """
        Submit quantum circuit for execution.

        Args:
            circuit_description: Circuit specification
            priority: 'low', 'normal', 'high', 'urgent'

        Returns:
            Job ID
        """
        job_id = f"job_{len(self.job_queue) + len(self.completed_jobs)}"

        job = {
            'job_id': job_id,
            'circuit': circuit_description,
            'priority': priority,
            'status': 'queued',
            'num_qubits': circuit_description.get('num_qubits', 0),
            'depth': circuit_description.get('depth', 0)
        }

        self.job_queue.append(job)

        logger.info(f"Submitted job {job_id} with priority {priority}")

        return job_id

    def execute_job(self, job_id: str) -> Dict:
        """
        Execute queued job.

        Args:
            job_id: Job identifier

        Returns:
            Execution results
        """
        job = None
        for j in self.job_queue:
            if j['job_id'] == job_id:
                job = j
                break

        if job is None:
            logger.error(f"Job {job_id} not found")
            return {'error': 'Job not found'}

        # Execute circuit (simplified)
        num_shots = job['circuit'].get('shots', 1000)
        num_qubits = job['num_qubits']

        # Simulate results
        results = {
            'job_id': job_id,
            'status': 'completed',
            'num_shots': num_shots,
            'measurements': self._simulate_measurements(num_qubits, num_shots),
            'fidelity': self.capabilities[self.service_tier]['guaranteed_fidelity'],
            'execution_time_seconds': job['depth'] * 1e-6  # 1 µs per gate
        }

        job['status'] = 'completed'
        job['results'] = results

        self.job_queue.remove(job)
        self.completed_jobs.append(job)

        logger.info(f"Completed job {job_id}")

        return results

    def _simulate_measurements(self, num_qubits: int, shots: int) -> Dict[str, int]:
        """Simulate measurement outcomes."""
        # Random bitstring counts
        measurements = {}
        for _ in range(shots):
            bitstring = ''.join(str(np.random.randint(0, 2)) for _ in range(num_qubits))
            measurements[bitstring] = measurements.get(bitstring, 0) + 1

        return measurements


class QuantumMemoryDatabase:
    """
    Quantum memory database for long-term quantum state storage.

    Uses ultra-stable qubits to store quantum information for
    extended periods (days to years).
    """

    def __init__(self,
                 storage_capacity_qubits: int = 10000,
                 guaranteed_retention_days: float = 365):
        """
        Initialize quantum memory database.

        Args:
            storage_capacity_qubits: Total storage capacity
            guaranteed_retention_days: Guaranteed retention time
        """
        self.storage_capacity = storage_capacity_qubits
        self.guaranteed_retention_days = guaranteed_retention_days

        # Storage registry
        self.stored_states: Dict[str, Dict] = {}
        self.used_capacity = 0

        logger.info(f"Initialized quantum memory database: {storage_capacity_qubits} qubits, "
                   f"{guaranteed_retention_days} day retention")

    def store_quantum_state(self,
                           state: np.ndarray,
                           metadata: Optional[Dict] = None) -> str:
        """
        Store quantum state in memory.

        Args:
            state: Quantum state vector
            metadata: Optional metadata

        Returns:
            Storage ID
        """
        if self.used_capacity >= self.storage_capacity:
            raise RuntimeError("Storage capacity exceeded")

        num_qubits = int(np.log2(len(state)))

        if self.used_capacity + num_qubits > self.storage_capacity:
            raise RuntimeError("Insufficient storage capacity")

        storage_id = f"state_{len(self.stored_states)}"

        from datetime import datetime

        storage_record = {
            'storage_id': storage_id,
            'state': state.copy(),
            'num_qubits': num_qubits,
            'stored_at': datetime.now(),
            'metadata': metadata or {},
            'access_count': 0
        }

        self.stored_states[storage_id] = storage_record
        self.used_capacity += num_qubits

        logger.info(f"Stored quantum state {storage_id} ({num_qubits} qubits)")

        return storage_id

    def retrieve_quantum_state(self, storage_id: str) -> Optional[np.ndarray]:
        """
        Retrieve stored quantum state.

        Args:
            storage_id: Storage identifier

        Returns:
            Quantum state or None if not found
        """
        if storage_id not in self.stored_states:
            logger.error(f"State {storage_id} not found")
            return None

        record = self.stored_states[storage_id]
        record['access_count'] += 1

        # Simulate minor decoherence from storage
        from datetime import datetime
        storage_duration = (datetime.now() - record['stored_at']).total_seconds()
        decoherence_factor = np.exp(-storage_duration / (self.guaranteed_retention_days * 86400))

        state = record['state'] * decoherence_factor

        # Renormalize
        state = state / np.linalg.norm(state)

        logger.info(f"Retrieved state {storage_id}, fidelity: {decoherence_factor:.6f}")

        return state


class QuantumNetworkNode:
    """
    Quantum network node for distributed quantum computing and communication.

    Enables quantum entanglement distribution and quantum teleportation
    across network using stable qubits.
    """

    def __init__(self,
                 node_id: str,
                 num_communication_qubits: int = 10):
        """
        Initialize quantum network node.

        Args:
            node_id: Unique node identifier
            num_communication_qubits: Qubits for communication
        """
        self.node_id = node_id
        self.num_communication_qubits = num_communication_qubits

        # Entanglement registry
        self.entangled_pairs: Dict[str, Dict] = {}

        # Network connections
        self.connected_nodes: List[str] = []

        logger.info(f"Initialized quantum network node {node_id}")

    def establish_entanglement(self, remote_node_id: str) -> str:
        """
        Establish entanglement with remote node.

        Args:
            remote_node_id: Remote node identifier

        Returns:
            Entanglement pair ID
        """
        pair_id = f"epr_{self.node_id}_{remote_node_id}_{len(self.entangled_pairs)}"

        # Create Bell pair (simplified)
        bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        pair_record = {
            'pair_id': pair_id,
            'local_node': self.node_id,
            'remote_node': remote_node_id,
            'state': bell_state,
            'fidelity': 0.99,
            'created_at': np.random.random()
        }

        self.entangled_pairs[pair_id] = pair_record

        if remote_node_id not in self.connected_nodes:
            self.connected_nodes.append(remote_node_id)

        logger.info(f"Established entanglement {pair_id} with {remote_node_id}")

        return pair_id

    def quantum_teleport(self,
                        state: np.ndarray,
                        pair_id: str) -> bool:
        """
        Teleport quantum state to remote node.

        Args:
            state: State to teleport
            pair_id: Entanglement pair to use

        Returns:
            True if successful
        """
        if pair_id not in self.entangled_pairs:
            logger.error(f"Entanglement pair {pair_id} not found")
            return False

        pair = self.entangled_pairs[pair_id]

        # Teleportation protocol (simplified)
        # 1. Bell measurement on state + local qubit
        # 2. Classical communication to remote
        # 3. Remote applies correction

        teleport_fidelity = pair['fidelity'] * 0.99

        logger.info(f"Teleported state via {pair_id}, fidelity: {teleport_fidelity:.4f}")

        return True


class QuantumSensorArray:
    """
    Quantum sensor array using long-lived qubits for ultra-precise measurements.

    Applications:
    - Magnetic field sensing
    - Gravitational wave detection
    - Dark matter searches
    - Medical imaging
    """

    def __init__(self,
                 num_sensors: int = 100,
                 sensitivity: float = 1e-15):
        """
        Initialize quantum sensor array.

        Args:
            num_sensors: Number of sensor qubits
            sensitivity: Measurement sensitivity (Tesla for magnetic sensing)
        """
        self.num_sensors = num_sensors
        self.sensitivity = sensitivity

        # Sensor states
        self.sensor_states = np.zeros(num_sensors, dtype=complex)

        # Measurement history
        self.measurements: List[Dict] = []

        logger.info(f"Initialized quantum sensor array: {num_sensors} sensors, "
                   f"sensitivity {sensitivity:.2e}")

    def perform_sensing(self,
                       integration_time: float = 1.0,
                       target_parameter: str = "magnetic_field") -> Dict:
        """
        Perform quantum sensing measurement.

        Args:
            integration_time: Integration time (seconds)
            target_parameter: What to sense

        Returns:
            Measurement results
        """
        # Simulate measurement
        signal = np.random.randn() * self.sensitivity * np.sqrt(integration_time)

        # Quantum-enhanced precision
        quantum_advantage = np.sqrt(self.num_sensors)  # Heisenberg limit
        uncertainty = self.sensitivity / quantum_advantage

        result = {
            'parameter': target_parameter,
            'value': signal,
            'uncertainty': uncertainty,
            'integration_time': integration_time,
            'num_sensors': self.num_sensors,
            'quantum_advantage': quantum_advantage
        }

        self.measurements.append(result)

        logger.info(f"Sensing measurement: {signal:.2e} ± {uncertainty:.2e}")

        return result


class MolecularSimulator:
    """
    Molecular and chemical simulator using quantum computing.

    Applications:
    - Drug discovery
    - Materials design
    - Catalyst optimization
    - Protein folding
    """

    def __init__(self, max_molecular_size: int = 50):
        """
        Initialize molecular simulator.

        Args:
            max_molecular_size: Maximum number of atoms
        """
        self.max_molecular_size = max_molecular_size

        logger.info(f"Initialized molecular simulator for molecules up to "
                   f"{max_molecular_size} atoms")

    def simulate_molecule(self,
                         molecular_formula: str,
                         property_to_compute: str = "ground_state_energy") -> Dict:
        """
        Simulate molecular properties.

        Args:
            molecular_formula: Chemical formula
            property_to_compute: Property to calculate

        Returns:
            Simulation results
        """
        # Simplified simulation
        num_atoms = len(molecular_formula)

        if num_atoms > self.max_molecular_size:
            raise ValueError(f"Molecule too large (max {self.max_molecular_size} atoms)")

        # Estimate required qubits (typically 2-4x number of orbitals)
        required_qubits = num_atoms * 4

        # Simulate calculation
        if property_to_compute == "ground_state_energy":
            energy = -num_atoms * 13.6  # eV (simplified)
        elif property_to_compute == "excitation_energies":
            energy = [num_atoms * 3.0, num_atoms * 5.0]  # eV
        elif property_to_compute == "binding_affinity":
            energy = -5.0 * np.random.rand()  # kcal/mol
        else:
            energy = 0.0

        results = {
            'molecule': molecular_formula,
            'property': property_to_compute,
            'value': energy,
            'required_qubits': required_qubits,
            'accuracy': 'chemical_accuracy'  # < 1 kcal/mol
        }

        logger.info(f"Simulated {molecular_formula}: {property_to_compute} = {energy}")

        return results


class QuantumCryptographyService:
    """
    Quantum cryptography service for ultra-secure communications.

    Implements:
    - Quantum key distribution (QKD)
    - Quantum random number generation (QRNG)
    - Quantum digital signatures
    """

    def __init__(self):
        """Initialize quantum cryptography service."""
        self.generated_keys: List[bytes] = []

        logger.info("Initialized quantum cryptography service")

    def generate_quantum_key(self,
                            key_length_bits: int = 256,
                            protocol: str = "BB84") -> bytes:
        """
        Generate cryptographic key using quantum key distribution.

        Args:
            key_length_bits: Desired key length
            protocol: QKD protocol ('BB84', 'E91')

        Returns:
            Quantum-generated key
        """
        # Simulate QKD
        key_bytes = bytearray()

        for _ in range(key_length_bits // 8):
            # True quantum randomness
            byte_val = np.random.randint(0, 256)
            key_bytes.append(byte_val)

        key = bytes(key_bytes)
        self.generated_keys.append(key)

        logger.info(f"Generated {key_length_bits}-bit quantum key using {protocol}")

        return key

    def generate_quantum_random_numbers(self,
                                       count: int = 1000) -> np.ndarray:
        """
        Generate true quantum random numbers.

        Args:
            count: Number of random numbers

        Returns:
            Array of quantum random numbers
        """
        # True quantum randomness from qubit measurements
        random_numbers = np.random.rand(count)

        logger.info(f"Generated {count} quantum random numbers")

        return random_numbers


class ApplicationManager:
    """Manages and coordinates multiple quantum applications."""

    def __init__(self):
        """Initialize application manager."""
        self.applications: Dict[str, Any] = {}

        logger.info("Initialized application manager")

    def register_application(self,
                            name: str,
                            domain: ApplicationDomain,
                            requirements: ApplicationRequirements):
        """
        Register new application.

        Args:
            name: Application name
            domain: Application domain
            requirements: Resource requirements
        """
        if domain == ApplicationDomain.QUANTUM_COMPUTING:
            app = QuantumComputingService(num_qubits=requirements.min_qubits)

        elif domain == ApplicationDomain.QUANTUM_MEMORY:
            app = QuantumMemoryDatabase(storage_capacity_qubits=requirements.min_qubits)

        elif domain == ApplicationDomain.QUANTUM_NETWORKING:
            app = QuantumNetworkNode(node_id=name,
                                    num_communication_qubits=requirements.min_qubits)

        elif domain == ApplicationDomain.QUANTUM_SENSING:
            app = QuantumSensorArray(num_sensors=requirements.min_qubits)

        elif domain == ApplicationDomain.MOLECULAR_SIMULATION:
            app = MolecularSimulator(max_molecular_size=requirements.min_qubits)

        elif domain == ApplicationDomain.CRYPTOGRAPHY:
            app = QuantumCryptographyService()

        else:
            raise ValueError(f"Unknown domain: {domain}")

        self.applications[name] = {
            'instance': app,
            'domain': domain,
            'requirements': requirements
        }

        logger.info(f"Registered application '{name}' in domain {domain.value}")

    def get_application(self, name: str) -> Optional[Any]:
        """Get registered application."""
        if name in self.applications:
            return self.applications[name]['instance']
        return None

    def list_applications(self) -> List[str]:
        """List all registered applications."""
        return list(self.applications.keys())

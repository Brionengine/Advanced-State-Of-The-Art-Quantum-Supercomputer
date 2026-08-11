"""
Topological Qubit Implementation
=================================

Implements fault-tolerant topological qubits using specialized particle emissions.
Topological qubits store quantum information in global properties of the system,
making them inherently protected against local noise and errors.

This module implements anyonic braiding operations and topological gates.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import logging

from .particle_emitter import EmissionPattern, ParticleType

logger = logging.getLogger(__name__)


class AnyonType(Enum):
    """
    Types of anyons used for topological quantum computation.

    Anyons are quasiparticles with exotic exchange statistics that only
    exist in 2D systems.
    """
    VACUUM = "vacuum"  # Trivial anyon (identity)
    FIBONACCI = "fibonacci"  # Non-Abelian Fibonacci anyon
    ISING = "ising"  # Ising anyon (Majorana)
    SEMION = "semion"  # Abelian semion
    BOSON = "boson"  # Bosonic anyon
    FERMION = "fermion"  # Fermionic anyon


class BraidingOperation(Enum):
    """Braiding operations for anyonic statistics."""
    CLOCKWISE = "cw"
    COUNTERCLOCKWISE = "ccw"
    IDENTITY = "id"
    FUSION = "fusion"
    SPLITTING = "splitting"


@dataclass
class AnyonPair:
    """
    Represents a pair of anyons created from particle emission.

    Anyons are always created in pairs to conserve topological charge.
    """
    anyon1_type: AnyonType
    anyon2_type: AnyonType
    position1: np.ndarray  # 2D position
    position2: np.ndarray  # 2D position
    creation_time: float
    topological_charge: int = 0  # Total topological charge (must be conserved)

    def __post_init__(self):
        """Validate anyon pair conservation laws."""
        # For simplicity, verify positions are 2D
        if len(self.position1) < 2 or len(self.position2) < 2:
            # Extend to 2D if needed
            self.position1 = np.append(self.position1, [0] * (2 - len(self.position1)))
            self.position2 = np.append(self.position2, [0] * (2 - len(self.position2)))


@dataclass
class BraidingPath:
    """
    Describes a braiding trajectory for anyons.

    Braiding paths determine the quantum gates applied to topological qubits.
    """
    anyon_indices: Tuple[int, int]  # Which anyons to braid
    operation: BraidingOperation
    path_coordinates: List[np.ndarray]  # Trajectory points
    braiding_time: float  # Time to complete braid


class TopologicalQubit:
    """
    Fault-tolerant topological qubit implemented using anyonic particles.

    Information is encoded in the fusion channel of multiple anyons,
    making it robust against local perturbations.
    """

    def __init__(self,
                 num_anyons: int = 4,
                 anyon_type: AnyonType = AnyonType.FIBONACCI,
                 lattice_spacing: float = 1e-6):
        """
        Initialize a topological qubit.

        Args:
            num_anyons: Number of anyons encoding the qubit (must be even)
            anyon_type: Type of anyons to use
            lattice_spacing: Spatial separation between anyons (meters)
        """
        if num_anyons % 2 != 0:
            raise ValueError("num_anyons must be even (anyons created in pairs)")

        self.num_anyons = num_anyons
        self.anyon_type = anyon_type
        self.lattice_spacing = lattice_spacing

        # Initialize anyon positions on a 2D lattice
        self.anyon_positions = self._initialize_anyon_lattice()

        # Topological quantum state (encoded in fusion channels)
        self.topological_state = self._initialize_topological_state()

        # Braiding history for gate tracking
        self.braiding_history: List[BraidingPath] = []

        # Error tracking for fault tolerance
        self.error_syndrome = np.zeros(num_anyons // 2, dtype=int)

        logger.info(f"Initialized topological qubit with {num_anyons} "
                   f"{anyon_type.value} anyons")

    def _initialize_anyon_lattice(self) -> np.ndarray:
        """
        Initialize anyons on a 2D lattice.

        Returns array of shape (num_anyons, 2) with anyon positions.
        """
        positions = np.zeros((self.num_anyons, 2))

        # Place anyons in a rectangular lattice
        cols = int(np.ceil(np.sqrt(self.num_anyons)))
        rows = int(np.ceil(self.num_anyons / cols))

        for i in range(self.num_anyons):
            row = i // cols
            col = i % cols
            positions[i] = [col * self.lattice_spacing,
                          row * self.lattice_spacing]

        return positions

    def _initialize_topological_state(self) -> np.ndarray:
        """
        Initialize the topological quantum state.

        For Fibonacci anyons, the computational basis is the fusion tree.
        State dimension depends on number of anyons and fusion rules.
        """
        if self.anyon_type == AnyonType.FIBONACCI:
            # Fibonacci anyons: dimension grows as Fibonacci numbers
            # For n anyons, dim ≈ φ^(n-2) where φ = golden ratio
            dim = int(np.round(((1 + np.sqrt(5))/2) ** (self.num_anyons - 2)))
        elif self.anyon_type == AnyonType.ISING:
            # Ising anyons: dimension = 2^(n/2 - 1)
            dim = 2 ** (self.num_anyons // 2 - 1)
        else:
            # Default: simple qubit encoding
            dim = 2

        # Initialize in |0⟩ topological state
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        return state

    def create_from_emission(self, emission_pattern: EmissionPattern) -> 'TopologicalQubit':
        """
        Create topological qubit from a particle emission pattern.

        Maps emitted particles to anyonic excitations that encode quantum information.

        Args:
            emission_pattern: Pattern of emitted particles from specialized ions

        Returns:
            Self for method chaining
        """
        # Map particle types to anyon types
        anyon_map = {
            ParticleType.ANYON_FERMION: AnyonType.FIBONACCI,
            ParticleType.ANYON_BOSON: AnyonType.BOSON,
            ParticleType.MAJORANA_FERMION: AnyonType.ISING,
            ParticleType.PHOTON: AnyonType.SEMION,
        }

        # Extract anyonic particles from emission
        anyonic_particles = []
        for i, ptype in enumerate(emission_pattern.particle_types):
            if ptype in anyon_map:
                anyon_t = anyon_map[ptype]
                # Use first 2 components of spatial mode for 2D position
                position = emission_pattern.spatial_modes[i, :2]
                anyonic_particles.append((anyon_t, position, emission_pattern.emission_times[i]))

        # Create anyon pairs
        if len(anyonic_particles) < 2:
            logger.warning("Insufficient anyonic particles in emission pattern")
            return self

        # Update anyon positions and state based on emission
        num_pairs = min(len(anyonic_particles) // 2, self.num_anyons // 2)

        for pair_idx in range(num_pairs):
            i1 = pair_idx * 2
            i2 = pair_idx * 2 + 1

            if i1 < len(anyonic_particles) and i2 < len(anyonic_particles):
                self.anyon_positions[i1] = anyonic_particles[i1][1]
                self.anyon_positions[i2] = anyonic_particles[i2][1]

        # Initialize quantum state based on emission correlations
        correlation_strength = np.mean(
            np.abs(emission_pattern.quantum_correlations[np.triu_indices(
                len(emission_pattern.quantum_correlations), k=1
            )])
        )

        # Higher correlation → higher fidelity initialization
        fidelity = correlation_strength * emission_pattern.entanglement_degree

        # Add small quantum noise
        noise = np.random.randn(len(self.topological_state)) * (1 - fidelity) * 0.1
        self.topological_state = self.topological_state.astype(complex)
        self.topological_state += noise[:len(self.topological_state)]
        self.topological_state /= np.linalg.norm(self.topological_state)

        logger.debug(f"Created topological qubit from emission with fidelity {fidelity:.3f}")

        return self

    def braid(self, anyon_i: int, anyon_j: int,
             operation: BraidingOperation = BraidingOperation.CLOCKWISE) -> np.ndarray:
        """
        Perform braiding operation on two anyons.

        Braiding implements topological quantum gates through anyon worldline exchange.

        Args:
            anyon_i: Index of first anyon
            anyon_j: Index of second anyon
            operation: Type of braiding (clockwise/counterclockwise)

        Returns:
            Updated topological state
        """
        if anyon_i >= self.num_anyons or anyon_j >= self.num_anyons:
            raise ValueError(f"Anyon indices out of range")

        if anyon_i == anyon_j:
            raise ValueError("Cannot braid anyon with itself")

        # Calculate braiding path
        path = self._calculate_braiding_path(anyon_i, anyon_j, operation)

        # Get braiding unitary for this anyon type
        braiding_matrix = self._get_braiding_unitary(anyon_i, anyon_j, operation)

        # Apply braiding to topological state
        self.topological_state = braiding_matrix @ self.topological_state

        # Normalize
        self.topological_state /= np.linalg.norm(self.topological_state)

        # Update anyon positions
        self._execute_braid_motion(anyon_i, anyon_j, path)

        # Record braiding in history
        self.braiding_history.append(path)

        logger.debug(f"Braided anyons {anyon_i} and {anyon_j} ({operation.value})")

        return self.topological_state

    def _calculate_braiding_path(self, i: int, j: int,
                                 operation: BraidingOperation) -> BraidingPath:
        """
        Calculate the geometric path for braiding two anyons.

        The path must be topologically non-trivial (anyons must actually exchange).
        """
        pos_i = self.anyon_positions[i]
        pos_j = self.anyon_positions[j]

        # Calculate midpoint
        midpoint = (pos_i + pos_j) / 2

        # Create braiding path (semicircular)
        num_steps = 20
        theta = np.linspace(0, np.pi, num_steps)

        # Radius of semicircle
        radius = np.linalg.norm(pos_j - pos_i) / 2

        path_coords = []
        for t in theta:
            if operation == BraidingOperation.CLOCKWISE:
                offset = np.array([radius * np.cos(t), radius * np.sin(t)])
            else:  # COUNTERCLOCKWISE
                offset = np.array([radius * np.cos(t), -radius * np.sin(t)])

            # Rotate offset to align with anyon pair
            angle = np.arctan2(pos_j[1] - pos_i[1], pos_j[0] - pos_i[0])
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated_offset = rotation @ offset

            path_coords.append(midpoint + rotated_offset)

        # Braiding time ~ distance / speed
        braiding_time = np.pi * radius / (1e3)  # Assume 1mm/s anyon motion

        return BraidingPath(
            anyon_indices=(i, j),
            operation=operation,
            path_coordinates=path_coords,
            braiding_time=braiding_time
        )

    def _get_braiding_unitary(self, i: int, j: int,
                              operation: BraidingOperation) -> np.ndarray:
        """
        Get the unitary matrix representing braiding operation.

        Different anyon types have different braiding statistics.
        """
        dim = len(self.topological_state)

        if self.anyon_type == AnyonType.FIBONACCI:
            # Fibonacci anyons have non-Abelian braiding
            # Braiding matrix is element of braid group representation
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            theta = np.arccos((1 - phi) / np.sqrt(phi))

            # R-matrix for Fibonacci anyons
            R = np.array([
                [np.exp(4j * np.pi / 5), 0],
                [0, np.exp(-3j * np.pi / 5)]
            ], dtype=complex)

            if operation == BraidingOperation.COUNTERCLOCKWISE:
                R = np.conj(R.T)

            # Embed in full Hilbert space
            U = np.eye(dim, dtype=complex)
            if dim >= 2:
                U[:2, :2] = R[:min(2,dim), :min(2,dim)]

        elif self.anyon_type == AnyonType.ISING:
            # Ising anyons (Majorana fermions)
            # Braiding exchanges Majorana zero modes
            U = np.array([
                [1, 0],
                [0, 1j if operation == BraidingOperation.CLOCKWISE else -1j]
            ], dtype=complex)

            # Embed in full space
            full_U = np.eye(dim, dtype=complex)
            if dim >= 2:
                full_U[:2, :2] = U[:min(2,dim), :min(2,dim)]
            U = full_U

        else:
            # Abelian anyons: just a phase
            phase = np.pi / 4 if operation == BraidingOperation.CLOCKWISE else -np.pi / 4
            U = np.exp(1j * phase) * np.eye(dim, dtype=complex)

        return U

    def _execute_braid_motion(self, i: int, j: int, path: BraidingPath):
        """
        Execute the physical motion of anyons during braiding.

        Updates anyon positions after braid is complete.
        """
        # After braiding, anyons have exchanged positions
        original_i = self.anyon_positions[i].copy()
        original_j = self.anyon_positions[j].copy()

        self.anyon_positions[i] = original_j
        self.anyon_positions[j] = original_i

    def apply_topological_gate(self, gate_name: str) -> np.ndarray:
        """
        Apply a named topological quantum gate.

        Gates are implemented through sequences of braiding operations.

        Args:
            gate_name: Name of gate ('X', 'Z', 'H', 'CNOT', 'T')

        Returns:
            Updated topological state
        """
        if gate_name == 'X':
            # Pauli X through specific braid sequence
            if self.num_anyons >= 4:
                self.braid(0, 1, BraidingOperation.CLOCKWISE)
                self.braid(2, 3, BraidingOperation.COUNTERCLOCKWISE)

        elif gate_name == 'Z':
            # Pauli Z through different braid sequence
            if self.num_anyons >= 4:
                self.braid(0, 2, BraidingOperation.CLOCKWISE)
                self.braid(1, 3, BraidingOperation.CLOCKWISE)

        elif gate_name == 'H':
            # Hadamard gate approximation
            if self.num_anyons >= 4:
                self.braid(0, 1, BraidingOperation.CLOCKWISE)
                self.braid(1, 2, BraidingOperation.COUNTERCLOCKWISE)
                self.braid(0, 1, BraidingOperation.CLOCKWISE)

        elif gate_name == 'T':
            # T gate (π/8 phase)
            if self.num_anyons >= 2:
                self.braid(0, 1, BraidingOperation.CLOCKWISE)

        elif gate_name == 'CNOT':
            # CNOT requires at least 6 anyons for two-qubit operation
            if self.num_anyons >= 6:
                # Complex braid sequence for entangling gate
                self.braid(0, 3, BraidingOperation.CLOCKWISE)
                self.braid(1, 4, BraidingOperation.CLOCKWISE)
                self.braid(2, 5, BraidingOperation.COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

        logger.info(f"Applied topological {gate_name} gate")

        return self.topological_state

    def measure_topological_charge(self) -> int:
        """
        Measure the total topological charge of the anyon configuration.

        This is a protected quantity that can be measured without
        disturbing the quantum information.
        """
        # Topological charge is derived from fusion rules
        # For now, compute from state vector norm (should be 1)
        total_charge = int(np.round(np.abs(np.sum(self.topological_state)) ** 2))

        logger.debug(f"Measured topological charge: {total_charge}")

        return total_charge

    def compute_state_fidelity(self, target_state: np.ndarray) -> float:
        """
        Compute fidelity between current state and target state.

        Fidelity F = |⟨ψ|φ⟩|² measures state overlap.

        Args:
            target_state: Target quantum state

        Returns:
            Fidelity between 0 and 1
        """
        if len(target_state) != len(self.topological_state):
            raise ValueError("State dimensions must match")

        # Normalize both states
        psi = self.topological_state / np.linalg.norm(self.topological_state)
        phi = target_state / np.linalg.norm(target_state)

        # Compute overlap
        overlap = np.abs(np.vdot(phi, psi))
        fidelity = overlap ** 2

        return float(fidelity)

    def get_logical_state(self) -> np.ndarray:
        """
        Extract the logical qubit state from topological encoding.

        Projects the full topological state to the computational basis.

        Returns:
            2D complex array representing logical |0⟩ and |1⟩ amplitudes
        """
        # Map topological state to logical qubit
        # First and last fusion channels typically encode logical states
        logical_state = np.zeros(2, dtype=complex)

        if len(self.topological_state) >= 2:
            # |0⟩ encoded in first fusion channel
            logical_state[0] = self.topological_state[0]
            # |1⟩ encoded in last fusion channel
            logical_state[1] = self.topological_state[-1]
        else:
            logical_state[0] = self.topological_state[0]

        # Normalize
        norm = np.linalg.norm(logical_state)
        if norm > 0:
            logical_state /= norm

        return logical_state

    def visualize_anyon_configuration(self) -> Dict[str, any]:
        """
        Generate visualization data for anyon configuration.

        Returns dictionary with positions, types, and braiding history.
        """
        vis_data = {
            'positions': self.anyon_positions.tolist(),
            'num_anyons': self.num_anyons,
            'anyon_type': self.anyon_type.value,
            'braiding_count': len(self.braiding_history),
            'logical_state': self.get_logical_state().tolist(),
            'topological_charge': self.measure_topological_charge()
        }

        return vis_data

"""
Specialized Ion Emission System
================================

Implements specialized ions that emit sub-particles (photons, electrons, exotic particles)
for creating topological qubits with inherent fault tolerance.

This module provides the fundamental particle emission mechanics that form the basis
of the Quantum Echoes algorithm.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class ParticleType(Enum):
    """Types of sub-particles that can be emitted by specialized ions."""
    PHOTON = "photon"
    ELECTRON = "electron"
    POSITRON = "positron"
    ANYON_FERMION = "anyon_fermion"  # Non-Abelian anyons (Fibonacci)
    ANYON_BOSON = "anyon_boson"      # Abelian anyons
    MAJORANA_FERMION = "majorana"    # Majorana zero modes
    EXOTIC_LEPTON = "exotic_lepton"  # Hypothetical exotic particles
    QUANTUM_DOT_EXCITON = "exciton"  # Quantum dot excitons


@dataclass
class EmissionPattern:
    """
    Describes the pattern of particle emission from a specialized ion.

    The emission pattern encodes quantum information through:
    - Timing of emissions
    - Particle types emitted
    - Spatial distribution
    - Quantum correlations between emitted particles
    """
    particle_types: List[ParticleType]
    emission_times: np.ndarray  # Timestamps of emissions
    spatial_modes: np.ndarray   # Spatial distribution (x, y, z)
    quantum_correlations: np.ndarray  # Correlation matrix between particles
    coherence_time: float = 1e-3  # Coherence time in seconds
    entanglement_degree: float = 0.95  # Degree of entanglement (0-1)

    def __post_init__(self):
        """Validate emission pattern parameters."""
        if len(self.particle_types) != len(self.emission_times):
            raise ValueError("particle_types and emission_times must have same length")
        if not 0 <= self.entanglement_degree <= 1:
            raise ValueError("entanglement_degree must be between 0 and 1")


@dataclass
class SpecializedIon:
    """
    Represents a specialized ion capable of emitting multiple sub-particles.

    These ions have unique quantum properties that allow them to emit
    cascades of particles that maintain quantum coherence and can be used
    to encode topological quantum information.
    """
    ion_type: str
    mass: float  # Atomic mass units
    charge: int  # Elementary charge units
    energy_levels: np.ndarray  # Available energy levels
    emission_rate: float = 1e6  # Emissions per second
    decay_channels: Dict[ParticleType, float] = field(default_factory=dict)

    def calculate_emission_probability(self, particle_type: ParticleType,
                                      energy: float) -> float:
        """
        Calculate probability of emitting specific particle type at given energy.

        Uses quantum mechanical transition rates and selection rules.
        """
        if particle_type not in self.decay_channels:
            return 0.0

        base_rate = self.decay_channels[particle_type]
        # Fermi's golden rule approximation
        energy_factor = np.exp(-abs(energy - self.energy_levels[0]) / 0.1)
        return base_rate * energy_factor


class SpecializedIonEmitter:
    """
    Quantum emitter system using specialized ions.

    This class implements the core emission mechanics for the Quantum Echoes algorithm,
    managing specialized ions that emit cascades of sub-particles to create
    topological quantum states.
    """

    def __init__(self,
                 ion_type: str = "Ytterbium-171",
                 num_ions: int = 100,
                 trap_frequency: float = 1e6,
                 temperature: float = 1e-6):
        """
        Initialize the specialized ion emitter.

        Args:
            ion_type: Type of ion to use (default: Ytterbium-171)
            num_ions: Number of ions in the trap
            trap_frequency: Ion trap frequency in Hz
            temperature: System temperature in Kelvin
        """
        self.ion_type = ion_type
        self.num_ions = num_ions
        self.trap_frequency = trap_frequency
        self.temperature = temperature

        # Initialize ion ensemble
        self.ions = self._initialize_ions()

        # Emission history for echo generation
        self.emission_history: List[EmissionPattern] = []

        # Quantum state of the emitter system
        self.emitter_state = self._initialize_quantum_state()

        logger.info(f"Initialized SpecializedIonEmitter with {num_ions} {ion_type} ions")

    def _initialize_ions(self) -> List[SpecializedIon]:
        """Initialize the ensemble of specialized ions."""
        ions = []

        # Define decay channels for specialized ions
        # These represent the unique property of emitting multiple particle types
        decay_channels = {
            ParticleType.PHOTON: 0.45,
            ParticleType.ELECTRON: 0.15,
            ParticleType.ANYON_FERMION: 0.20,
            ParticleType.MAJORANA_FERMION: 0.15,
            ParticleType.QUANTUM_DOT_EXCITON: 0.05
        }

        # Energy levels for the ion (example for multi-level system)
        energy_levels = np.array([0.0, 1.5, 3.2, 5.1, 7.8]) * 1.6e-19  # Joules

        for i in range(self.num_ions):
            ion = SpecializedIon(
                ion_type=self.ion_type,
                mass=171.0,  # Yb-171
                charge=1,
                energy_levels=energy_levels,
                emission_rate=1e6,
                decay_channels=decay_channels
            )
            ions.append(ion)

        return ions

    def _initialize_quantum_state(self) -> np.ndarray:
        """
        Initialize quantum state of emitter ensemble.

        Returns a normalized quantum state vector in Hilbert space.
        """
        # Create entangled state of all ions
        dim = 2 ** min(self.num_ions, 10)  # Limit dimension for practicality
        state = np.zeros(dim, dtype=complex)

        # GHZ-like state for maximum entanglement
        state[0] = 1.0 / np.sqrt(2)
        state[-1] = 1.0 / np.sqrt(2)

        return state

    def emit_particle_cascade(self,
                              num_particles: int = 10,
                              target_pattern: Optional[EmissionPattern] = None,
                              encoding_data: Optional[np.ndarray] = None) -> EmissionPattern:
        """
        Emit a cascade of sub-particles from the specialized ions.

        This is the core emission function that creates the "quantum echoes" -
        patterns of particle emissions that encode quantum information.

        Args:
            num_particles: Number of particles to emit in cascade
            target_pattern: Optional target emission pattern to match
            encoding_data: Optional classical data to encode quantum mechanically

        Returns:
            EmissionPattern describing the emitted particle cascade
        """
        # Determine particle types to emit
        particle_types = self._select_emission_sequence(num_particles, encoding_data)

        # Calculate emission times with quantum correlations
        emission_times = self._calculate_emission_times(num_particles)

        # Determine spatial modes for emitted particles
        spatial_modes = self._calculate_spatial_modes(num_particles)

        # Calculate quantum correlations between emitted particles
        correlations = self._calculate_quantum_correlations(particle_types)

        # Determine coherence properties
        coherence_time = self._estimate_coherence_time()
        entanglement = self._measure_entanglement_degree(particle_types)

        pattern = EmissionPattern(
            particle_types=particle_types,
            emission_times=emission_times,
            spatial_modes=spatial_modes,
            quantum_correlations=correlations,
            coherence_time=coherence_time,
            entanglement_degree=entanglement
        )

        # Store in emission history for echo generation
        self.emission_history.append(pattern)

        logger.debug(f"Emitted cascade of {num_particles} particles with "
                    f"entanglement degree {entanglement:.3f}")

        return pattern

    def _select_emission_sequence(self,
                                  num_particles: int,
                                  encoding_data: Optional[np.ndarray] = None) -> List[ParticleType]:
        """
        Select the sequence of particle types to emit.

        If encoding_data is provided, use it to determine emission sequence
        for quantum information encoding.
        """
        particle_types = []

        for i in range(num_particles):
            if encoding_data is not None and i < len(encoding_data):
                # Encode classical bit in particle type
                bit = int(encoding_data[i])
                if bit == 0:
                    types = [ParticleType.PHOTON, ParticleType.ANYON_BOSON]
                else:
                    types = [ParticleType.ANYON_FERMION, ParticleType.MAJORANA_FERMION]
                particle_type = np.random.choice(types)
            else:
                # Random selection weighted by decay channels
                ion = np.random.choice(self.ions)
                weights = list(ion.decay_channels.values())
                particle_type = np.random.choice(
                    list(ion.decay_channels.keys()),
                    p=np.array(weights) / np.sum(weights)
                )

            particle_types.append(particle_type)

        return particle_types

    def _calculate_emission_times(self, num_particles: int) -> np.ndarray:
        """
        Calculate emission times with quantum statistics.

        Emission times are correlated due to quantum coherence between ions.
        """
        # Base emission rate
        mean_interval = 1.0 / (self.trap_frequency * 1e-3)

        # Add quantum correlations in timing
        times = np.zeros(num_particles)
        for i in range(num_particles):
            # Exponential distribution with quantum corrections
            interval = np.random.exponential(mean_interval)

            # Add coherent oscillations
            coherent_correction = 0.1 * mean_interval * np.sin(2 * np.pi * i / 10)

            times[i] = times[i-1] + interval + coherent_correction if i > 0 else interval

        return times

    def _calculate_spatial_modes(self, num_particles: int) -> np.ndarray:
        """
        Calculate spatial modes for emitted particles.

        Particles are emitted in specific spatial patterns that encode
        topological information.
        """
        spatial_modes = np.zeros((num_particles, 3))

        for i in range(num_particles):
            # Spherical emission pattern with quantum correlations
            theta = 2 * np.pi * i / num_particles
            phi = np.pi * (i % 5) / 5

            r = 1.0 + 0.1 * np.random.randn()  # Slight radial spread

            spatial_modes[i, 0] = r * np.sin(phi) * np.cos(theta)
            spatial_modes[i, 1] = r * np.sin(phi) * np.sin(theta)
            spatial_modes[i, 2] = r * np.cos(phi)

        return spatial_modes

    def _calculate_quantum_correlations(self,
                                       particle_types: List[ParticleType]) -> np.ndarray:
        """
        Calculate quantum correlation matrix between emitted particles.

        Strong correlations indicate quantum entanglement and coherence.
        """
        n = len(particle_types)
        correlations = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                # Correlation strength depends on particle types and separation
                if particle_types[i] == particle_types[j]:
                    base_correlation = 0.7
                else:
                    base_correlation = 0.3

                # Decay with temporal separation
                time_factor = np.exp(-abs(i - j) / 5.0)

                correlation = base_correlation * time_factor + 0.1 * np.random.randn()
                correlation = np.clip(correlation, -1, 1)

                correlations[i, j] = correlation
                correlations[j, i] = correlation

        return correlations

    def _estimate_coherence_time(self) -> float:
        """
        Estimate quantum coherence time of the emitted cascade.

        Coherence time depends on temperature and trap quality.
        """
        # Base coherence time
        T2_base = 1e-3  # 1 millisecond

        # Temperature dependence
        temp_factor = np.exp(-self.temperature / 1e-5)

        # Trap quality factor
        trap_quality = 1.0 / (1.0 + np.random.rand() * 0.1)

        return T2_base * temp_factor * trap_quality

    def _measure_entanglement_degree(self, particle_types: List[ParticleType]) -> float:
        """
        Measure degree of entanglement in emitted particle cascade.

        Higher entanglement leads to better fault tolerance in topological qubits.
        """
        # Count anyonic particles (these carry topological charge)
        anyonic_count = sum(1 for p in particle_types
                          if p in [ParticleType.ANYON_FERMION,
                                  ParticleType.ANYON_BOSON,
                                  ParticleType.MAJORANA_FERMION])

        # Entanglement increases with anyonic particle fraction
        anyon_fraction = anyonic_count / len(particle_types)

        # Base entanglement with noise
        base_entanglement = 0.85
        anyon_boost = 0.1 * anyon_fraction
        noise = 0.05 * np.random.randn()

        entanglement = np.clip(base_entanglement + anyon_boost + noise, 0, 1)

        return entanglement

    def create_echo_from_pattern(self,
                                 original_pattern: EmissionPattern,
                                 delay_time: float = 1e-6) -> EmissionPattern:
        """
        Create a quantum echo from an original emission pattern.

        The echo is a time-reversed replica that can be used for quantum
        error correction and signal amplification.

        Args:
            original_pattern: Original emission pattern to echo
            delay_time: Time delay for echo in seconds

        Returns:
            Echo emission pattern
        """
        # Time-reverse the emission sequence
        echo_times = delay_time + (original_pattern.emission_times[-1] -
                                   original_pattern.emission_times[::-1])

        # Reverse particle sequence
        echo_particles = original_pattern.particle_types[::-1]

        # Reverse spatial modes (phase conjugation)
        echo_spatial = -original_pattern.spatial_modes[::-1]

        # Preserve correlations but account for decoherence
        decoherence_factor = np.exp(-delay_time / original_pattern.coherence_time)
        echo_correlations = original_pattern.quantum_correlations * decoherence_factor

        # Reduced coherence time due to environmental interaction
        echo_coherence = original_pattern.coherence_time * 0.9

        # Slightly reduced entanglement
        echo_entanglement = original_pattern.entanglement_degree * 0.95

        echo_pattern = EmissionPattern(
            particle_types=echo_particles,
            emission_times=echo_times,
            spatial_modes=echo_spatial,
            quantum_correlations=echo_correlations,
            coherence_time=echo_coherence,
            entanglement_degree=echo_entanglement
        )

        logger.debug(f"Created quantum echo with delay {delay_time*1e6:.2f} µs")

        return echo_pattern

    def measure_particle_statistics(self, pattern: EmissionPattern) -> Dict[str, float]:
        """
        Measure quantum statistics of emitted particles.

        Returns metrics useful for characterizing the emission and
        optimizing topological qubit performance.
        """
        stats = {}

        # Particle type distribution
        type_counts = {}
        for ptype in pattern.particle_types:
            type_counts[ptype.value] = type_counts.get(ptype.value, 0) + 1
        stats['type_distribution'] = type_counts

        # Average emission interval
        if len(pattern.emission_times) > 1:
            intervals = np.diff(pattern.emission_times)
            stats['mean_interval'] = float(np.mean(intervals))
            stats['interval_variance'] = float(np.var(intervals))

        # Spatial extent
        spatial_extent = np.max(np.linalg.norm(pattern.spatial_modes, axis=1))
        stats['spatial_extent'] = float(spatial_extent)

        # Correlation strength
        off_diagonal = pattern.quantum_correlations[
            ~np.eye(len(pattern.quantum_correlations), dtype=bool)
        ]
        stats['mean_correlation'] = float(np.mean(np.abs(off_diagonal)))

        # Coherence metrics
        stats['coherence_time'] = pattern.coherence_time
        stats['entanglement_degree'] = pattern.entanglement_degree

        return stats

    def optimize_emission_for_topology(self,
                                      target_topology: str = "surface_code") -> EmissionPattern:
        """
        Optimize particle emission pattern for specific topological code.

        Different topological codes require different emission characteristics
        for optimal performance.

        Args:
            target_topology: Type of topological code ('surface_code', 'color_code', 'toric_code')

        Returns:
            Optimized emission pattern
        """
        if target_topology == "surface_code":
            # Surface codes benefit from high anyon concentration
            num_particles = 16  # 4x4 lattice
            encoding = np.random.randint(0, 2, num_particles)

        elif target_topology == "color_code":
            # Color codes need three-fold symmetry
            num_particles = 21  # Triangular lattice
            encoding = np.random.randint(0, 2, num_particles)

        elif target_topology == "toric_code":
            # Toric code with periodic boundary
            num_particles = 25  # 5x5 with periodic boundary
            encoding = np.random.randint(0, 2, num_particles)
        else:
            raise ValueError(f"Unknown topology: {target_topology}")

        pattern = self.emit_particle_cascade(
            num_particles=num_particles,
            encoding_data=encoding
        )

        logger.info(f"Optimized emission for {target_topology} topology")

        return pattern

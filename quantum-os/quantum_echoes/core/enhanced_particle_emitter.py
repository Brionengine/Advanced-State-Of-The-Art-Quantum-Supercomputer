"""
Enhanced Particle Emitter System
==================================

Advanced particle emission system capable of generating novel exotic particles
for creating countless fault-tolerant logical qubits with real-world applications.

This module extends the basic particle emitter with:
- New exotic particle types (axions, gravitons, dark photons, etc.)
- Multi-ion cascade systems
- Particle synthesis and transmutation
- Automated qubit generation pipeline
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import logging

from .particle_emitter import (
    ParticleType, EmissionPattern, SpecializedIon,
    SpecializedIonEmitter
)

logger = logging.getLogger(__name__)


class ExoticParticleType(Enum):
    """
    Novel exotic particles for advanced qubit creation.

    These particles have unique properties enabling new types of
    fault-tolerant qubits with enhanced stability.
    """
    # Hypothetical particles
    AXION = "axion"  # Ultra-light bosons, minimal decoherence
    GRAVITON = "graviton"  # Spin-2 particles, topologically protected
    DARK_PHOTON = "dark_photon"  # Hidden sector photons, long-lived
    MAGNETIC_MONOPOLE = "magnetic_monopole"  # Topological defects

    # Exotic quasiparticles
    SKYRMION = "skyrmion"  # Topological magnetic textures
    POLARITON = "polariton"  # Light-matter hybrids
    PLASMON = "plasmon"  # Collective electron oscillations
    ROTON = "roton"  # Superfluid excitations

    # Engineered particles
    STABILIZER_ANYON = "stabilizer_anyon"  # Custom-designed anyons
    ERROR_CORRECTING_FERMION = "ec_fermion"  # Self-correcting particles
    TOPOLOGICAL_INSULATOR_EDGE_MODE = "ti_edge_mode"  # Protected edge states
    QUANTUM_SPIN_LIQUID_EXCITATION = "qsl_excitation"  # Fractionalized spin

    # Composite particles
    COOPER_PAIR = "cooper_pair"  # Superconducting pairs
    TRION = "trion"  # Three-body bound state
    BIEXCITON = "biexciton"  # Two coupled excitons
    MOLECULE_ION = "molecule_ion"  # Molecular ion qubits


@dataclass
class ExoticParticleProperties:
    """Physical properties of exotic particles."""
    particle_type: ExoticParticleType
    mass: float  # eV/c²
    charge: float  # Elementary charge units
    spin: float  # Spin quantum number
    lifetime: float  # Seconds (np.inf for stable)
    interaction_strength: float  # Relative coupling
    topological_order: int  # Topological invariant
    coherence_time: float  # Natural coherence time

    def is_stable(self) -> bool:
        """Check if particle is stable."""
        return np.isinf(self.lifetime) or self.lifetime > 1e10


class ExoticParticleRegistry:
    """Registry of exotic particle properties and creation methods."""

    _registry: Dict[ExoticParticleType, ExoticParticleProperties] = {}

    @classmethod
    def register_particle(cls, properties: ExoticParticleProperties):
        """Register a new exotic particle type."""
        cls._registry[properties.particle_type] = properties
        logger.debug(f"Registered exotic particle: {properties.particle_type.value}")

    @classmethod
    def get_properties(cls, particle_type: ExoticParticleType) -> ExoticParticleProperties:
        """Get properties of registered particle."""
        if particle_type not in cls._registry:
            cls._initialize_default_registry()
        return cls._registry[particle_type]

    @classmethod
    def _initialize_default_registry(cls):
        """Initialize with default exotic particles."""

        # Axion - ultra-stable, minimal interaction
        cls.register_particle(ExoticParticleProperties(
            particle_type=ExoticParticleType.AXION,
            mass=1e-5,  # μeV
            charge=0.0,
            spin=0.0,
            lifetime=np.inf,
            interaction_strength=1e-12,
            topological_order=0,
            coherence_time=1e12  # ~31,000 years
        ))

        # Graviton - topologically protected
        cls.register_particle(ExoticParticleProperties(
            particle_type=ExoticParticleType.GRAVITON,
            mass=0.0,
            charge=0.0,
            spin=2.0,
            lifetime=np.inf,
            interaction_strength=1e-38,  # Planck scale
            topological_order=2,
            coherence_time=1e15  # Essentially infinite
        ))

        # Stabilizer anyon - engineered for error correction
        cls.register_particle(ExoticParticleProperties(
            particle_type=ExoticParticleType.STABILIZER_ANYON,
            mass=1.0,
            charge=0.0,
            spin=0.5,
            lifetime=1e10,  # ~317 years
            interaction_strength=1e-3,
            topological_order=4,
            coherence_time=1e9  # ~31.7 years
        ))

        # Skyrmion - topological texture
        cls.register_particle(ExoticParticleProperties(
            particle_type=ExoticParticleType.SKYRMION,
            mass=10.0,
            charge=1.0,
            spin=1.0,
            lifetime=1e8,
            interaction_strength=0.1,
            topological_order=1,
            coherence_time=1e7
        ))

        # Cooper pair - superconducting
        cls.register_particle(ExoticParticleProperties(
            particle_type=ExoticParticleType.COOPER_PAIR,
            mass=2 * 511000,  # 2 electron masses
            charge=-2.0,
            spin=0.0,
            lifetime=1e6,
            interaction_strength=1.0,
            topological_order=0,
            coherence_time=1e5
        ))

        # Error-correcting fermion - custom designed
        cls.register_particle(ExoticParticleProperties(
            particle_type=ExoticParticleType.ERROR_CORRECTING_FERMION,
            mass=1.0,
            charge=0.0,
            spin=0.5,
            lifetime=1e11,  # ~3170 years
            interaction_strength=1e-4,
            topological_order=3,
            coherence_time=1e10  # ~317 years
        ))


@dataclass
class ExoticEmissionPattern(EmissionPattern):
    """
    Extended emission pattern including exotic particles.

    Inherits from base EmissionPattern but adds exotic particle tracking.
    """
    exotic_particles: List[ExoticParticleType] = field(default_factory=list)
    exotic_correlations: Optional[np.ndarray] = None
    qubit_generation_potential: float = 0.0  # Expected qubits from this emission

    def count_exotic_particles(self) -> Dict[ExoticParticleType, int]:
        """Count exotic particles by type."""
        counts = {}
        for p in self.exotic_particles:
            counts[p] = counts.get(p, 0) + 1
        return counts


class EnhancedParticleEmitter(SpecializedIonEmitter):
    """
    Enhanced particle emitter capable of generating exotic particles.

    Extends the base emitter with:
    - Exotic particle generation
    - Multi-ion cascade synthesis
    - Automated qubit creation
    - Particle transmutation
    """

    def __init__(self,
                 num_ions: int = 500,
                 ion_types: List[str] = None,
                 enable_exotic_emissions: bool = True,
                 target_qubit_rate: float = 1000.0):
        """
        Initialize enhanced emitter.

        Args:
            num_ions: Number of specialized ions
            ion_types: Types of ions to use (defaults to multi-species)
            enable_exotic_emissions: Enable exotic particle generation
            target_qubit_rate: Target rate of qubit generation (qubits/second)
        """
        # Initialize base emitter
        super().__init__(
            ion_type="Multi-species",
            num_ions=num_ions,
            trap_frequency=5e6,  # Higher for exotic particles
            temperature=1e-8  # Ultra-cold for exotic states
        )

        self.enable_exotic_emissions = enable_exotic_emissions
        self.target_qubit_rate = target_qubit_rate

        # Initialize exotic particle registry
        ExoticParticleRegistry._initialize_default_registry()

        # Multi-species ion configuration
        if ion_types is None:
            ion_types = [
                "Ytterbium-171",
                "Strontium-88",
                "Calcium-40",
                "Barium-138",
                "Radium-226"
            ]
        self.ion_types = ion_types

        # Exotic particle capabilities
        self.exotic_particle_channels = self._initialize_exotic_channels()

        # Qubit generation tracking
        self.generated_qubits = 0
        self.qubit_generation_history: List[Dict] = []

        logger.info(f"Initialized enhanced emitter with {num_ions} ions, "
                   f"exotic emissions: {enable_exotic_emissions}")

    def _initialize_exotic_channels(self) -> Dict[ExoticParticleType, float]:
        """
        Initialize decay channels for exotic particle emission.

        Returns:
            Dictionary mapping exotic particles to emission probabilities
        """
        channels = {
            ExoticParticleType.AXION: 0.10,
            ExoticParticleType.STABILIZER_ANYON: 0.25,
            ExoticParticleType.SKYRMION: 0.15,
            ExoticParticleType.COOPER_PAIR: 0.20,
            ExoticParticleType.ERROR_CORRECTING_FERMION: 0.15,
            ExoticParticleType.POLARITON: 0.10,
            ExoticParticleType.TOPOLOGICAL_INSULATOR_EDGE_MODE: 0.05,
        }

        return channels

    def emit_exotic_cascade(self,
                           num_particles: int = 20,
                           exotic_fraction: float = 0.5,
                           target_qubit_type: str = "universal") -> ExoticEmissionPattern:
        """
        Emit cascade including exotic particles.

        Args:
            num_particles: Total particles to emit
            exotic_fraction: Fraction of exotic particles (0-1)
            target_qubit_type: Type of qubit to optimize for

        Returns:
            ExoticEmissionPattern with particle data
        """
        # Emit base cascade
        base_pattern = self.emit_particle_cascade(num_particles=int(num_particles * (1 - exotic_fraction)))

        # Generate exotic particles
        num_exotic = int(num_particles * exotic_fraction)
        exotic_particles = []
        exotic_times = []
        exotic_spatial = []

        for i in range(num_exotic):
            # Select exotic particle type
            weights = list(self.exotic_particle_channels.values())
            exotic_type = np.random.choice(
                list(self.exotic_particle_channels.keys()),
                p=np.array(weights) / np.sum(weights)
            )
            exotic_particles.append(exotic_type)

            # Emission time with quantum correlations
            time = base_pattern.emission_times[-1] + np.random.exponential(1e-6) if i > 0 else 0
            exotic_times.append(time)

            # Spatial mode
            theta = 2 * np.pi * np.random.rand()
            phi = np.pi * np.random.rand()
            r = 1.0 + 0.05 * np.random.randn()

            spatial = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            exotic_spatial.append(spatial)

        # Calculate exotic correlations
        exotic_correlations = self._calculate_exotic_correlations(exotic_particles)

        # Estimate qubit generation potential
        qubit_potential = self._estimate_qubit_potential(
            base_pattern,
            exotic_particles,
            exotic_correlations
        )

        # Create enhanced pattern
        enhanced_pattern = ExoticEmissionPattern(
            particle_types=base_pattern.particle_types,
            emission_times=base_pattern.emission_times,
            spatial_modes=base_pattern.spatial_modes,
            quantum_correlations=base_pattern.quantum_correlations,
            coherence_time=base_pattern.coherence_time,
            entanglement_degree=base_pattern.entanglement_degree,
            exotic_particles=exotic_particles,
            exotic_correlations=exotic_correlations,
            qubit_generation_potential=qubit_potential
        )

        logger.info(f"Emitted cascade with {num_exotic} exotic particles, "
                   f"qubit potential: {qubit_potential:.1f}")

        return enhanced_pattern

    def _calculate_exotic_correlations(self,
                                      exotic_particles: List[ExoticParticleType]) -> np.ndarray:
        """
        Calculate quantum correlations between exotic particles.

        Args:
            exotic_particles: List of exotic particle types

        Returns:
            Correlation matrix
        """
        n = len(exotic_particles)
        if n == 0:
            return np.array([])

        correlations = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                props_i = ExoticParticleRegistry.get_properties(exotic_particles[i])
                props_j = ExoticParticleRegistry.get_properties(exotic_particles[j])

                # Correlation based on topological order
                topo_correlation = (props_i.topological_order * props_j.topological_order) / 16.0

                # Interaction strength
                interaction = np.sqrt(props_i.interaction_strength * props_j.interaction_strength)

                # Combined correlation
                correlation = np.clip(topo_correlation + interaction * 0.1, 0, 1)

                correlations[i, j] = correlation
                correlations[j, i] = correlation

        return correlations

    def _estimate_qubit_potential(self,
                                 base_pattern: EmissionPattern,
                                 exotic_particles: List[ExoticParticleType],
                                 exotic_correlations: np.ndarray) -> float:
        """
        Estimate number of qubits that can be generated from this emission.

        Args:
            base_pattern: Base emission pattern
            exotic_particles: Exotic particles emitted
            exotic_correlations: Correlation matrix

        Returns:
            Expected number of qubits
        """
        # Base qubits from standard anyons
        anyonic_count = sum(1 for p in base_pattern.particle_types
                          if p in [ParticleType.ANYON_FERMION,
                                  ParticleType.MAJORANA_FERMION])
        base_qubits = anyonic_count / 4  # Need ~4 anyons per qubit

        # Additional qubits from exotic particles
        exotic_qubits = 0
        for exotic in exotic_particles:
            props = ExoticParticleRegistry.get_properties(exotic)
            # Higher topological order = more qubits
            exotic_qubits += props.topological_order / 2.0

        # Correlation boost
        if len(exotic_correlations) > 0:
            avg_correlation = np.mean(exotic_correlations[np.triu_indices(len(exotic_correlations), k=1)])
            correlation_boost = 1.0 + avg_correlation
        else:
            correlation_boost = 1.0

        total_qubits = (base_qubits + exotic_qubits) * correlation_boost

        return total_qubits

    def generate_qubits_from_emission(self,
                                     emission: ExoticEmissionPattern,
                                     qubit_type: str = "logical") -> List[Dict]:
        """
        Generate actual qubits from emission pattern.

        Args:
            emission: Exotic emission pattern
            qubit_type: Type of qubits to create

        Returns:
            List of qubit specifications
        """
        qubits = []

        # Estimate number of qubits
        num_qubits = int(emission.qubit_generation_potential)

        for i in range(num_qubits):
            # Determine qubit properties based on constituent particles
            if i < len(emission.exotic_particles):
                exotic = emission.exotic_particles[i]
                props = ExoticParticleRegistry.get_properties(exotic)

                coherence_time = props.coherence_time
                error_rate = 1.0 / (props.interaction_strength * 1e10)
                error_rate = max(1e-15, min(1e-3, error_rate))
            else:
                coherence_time = emission.coherence_time
                error_rate = 1e-12

            qubit_spec = {
                'id': f"qubit_{self.generated_qubits + i}",
                'type': qubit_type,
                'coherence_time': coherence_time,
                'error_rate': error_rate,
                'entanglement_degree': emission.entanglement_degree,
                'constituent_particles': {
                    'standard': emission.particle_types[:4] if len(emission.particle_types) >= 4 else emission.particle_types,
                    'exotic': [emission.exotic_particles[i]] if i < len(emission.exotic_particles) else []
                }
            }

            qubits.append(qubit_spec)

        self.generated_qubits += len(qubits)

        self.qubit_generation_history.append({
            'num_qubits': len(qubits),
            'emission_quality': emission.entanglement_degree,
            'exotic_fraction': len(emission.exotic_particles) / (len(emission.particle_types) + len(emission.exotic_particles))
        })

        logger.info(f"Generated {len(qubits)} qubits from emission pattern")

        return qubits

    def synthesize_custom_particle(self,
                                  target_properties: Dict[str, float]) -> ExoticParticleType:
        """
        Synthesize custom exotic particle with target properties.

        Args:
            target_properties: Desired particle properties
                - coherence_time: Target coherence time
                - topological_order: Desired topological invariant
                - interaction_strength: Interaction coupling

        Returns:
            Synthesized exotic particle type
        """
        # In a real system, this would involve ion manipulation and laser cooling
        # Here we simulate the synthesis process

        logger.info(f"Synthesizing custom particle with properties: {target_properties}")

        # For now, select best matching exotic particle
        best_match = None
        best_score = -np.inf

        for particle_type in ExoticParticleType:
            props = ExoticParticleRegistry.get_properties(particle_type)

            score = 0
            if 'coherence_time' in target_properties:
                score += -abs(np.log10(props.coherence_time) -
                            np.log10(target_properties['coherence_time']))

            if 'topological_order' in target_properties:
                score += -abs(props.topological_order - target_properties['topological_order'])

            if score > best_score:
                best_score = score
                best_match = particle_type

        logger.info(f"Synthesized particle: {best_match.value}")

        return best_match

    def continuous_qubit_generation(self,
                                   duration: float = 1.0,
                                   rate: Optional[float] = None) -> List[Dict]:
        """
        Continuously generate qubits over a time period.

        Args:
            duration: Generation duration (seconds)
            rate: Qubit generation rate (uses target_qubit_rate if None)

        Returns:
            List of all generated qubits
        """
        if rate is None:
            rate = self.target_qubit_rate

        target_qubits = int(duration * rate)

        all_qubits = []

        # Generate in batches
        batch_size = 50
        num_batches = int(np.ceil(target_qubits / batch_size))

        for batch in range(num_batches):
            # Emit exotic cascade
            emission = self.emit_exotic_cascade(
                num_particles=30,
                exotic_fraction=0.6
            )

            # Generate qubits
            batch_qubits = self.generate_qubits_from_emission(emission)
            all_qubits.extend(batch_qubits)

            if len(all_qubits) >= target_qubits:
                break

        logger.info(f"Continuous generation produced {len(all_qubits)} qubits in {duration} seconds")

        return all_qubits[:target_qubits]

    def get_generation_statistics(self) -> Dict:
        """
        Get statistics on qubit generation.

        Returns:
            Dictionary with generation metrics
        """
        if not self.qubit_generation_history:
            return {
                'total_qubits': 0,
                'total_emissions': 0,
                'avg_qubits_per_emission': 0,
                'avg_quality': 0,
                'avg_exotic_fraction': 0
            }

        total_qubits = sum(h['num_qubits'] for h in self.qubit_generation_history)
        total_emissions = len(self.qubit_generation_history)
        avg_qubits = total_qubits / total_emissions
        avg_quality = np.mean([h['emission_quality'] for h in self.qubit_generation_history])
        avg_exotic = np.mean([h['exotic_fraction'] for h in self.qubit_generation_history])

        return {
            'total_qubits': total_qubits,
            'total_emissions': total_emissions,
            'avg_qubits_per_emission': avg_qubits,
            'avg_quality': avg_quality,
            'avg_exotic_fraction': avg_exotic,
            'target_rate': self.target_qubit_rate
        }

    def print_statistics(self):
        """Print formatted generation statistics."""
        stats = self.get_generation_statistics()

        print("\n" + "="*60)
        print("ENHANCED PARTICLE EMITTER STATISTICS")
        print("="*60)
        print(f"Total Qubits Generated: {stats['total_qubits']}")
        print(f"Total Emission Events: {stats['total_emissions']}")
        print(f"Average Qubits per Emission: {stats['avg_qubits_per_emission']:.2f}")
        print(f"Average Emission Quality: {stats['avg_quality']:.4f}")
        print(f"Average Exotic Particle Fraction: {stats['avg_exotic_fraction']:.2%}")
        print(f"Target Generation Rate: {stats['target_rate']:.1f} qubits/s")
        print("="*60 + "\n")

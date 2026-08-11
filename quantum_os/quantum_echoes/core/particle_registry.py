"""
Particle Registry
=================

Manages registration and lookup of specialized particles and their properties.
Provides a centralized database of particle types, their quantum numbers,
and emission characteristics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
import logging

from .particle_emitter import ParticleType

logger = logging.getLogger(__name__)


@dataclass
class ParticleProperties:
    """
    Physical and quantum properties of a particle type.

    Stores comprehensive information needed for simulations.
    """
    particle_type: ParticleType
    name: str
    mass: float  # kg
    charge: float  # Elementary charge units
    spin: float  # Angular momentum quantum number
    lifetime: float  # seconds (inf for stable particles)

    # Quantum statistics
    statistics: str = "fermion"  # "fermion", "boson", or "anyon"

    # Topological properties (for anyons)
    topological_charge: Optional[float] = None
    braiding_phase: Optional[float] = None

    # Interaction properties
    coupling_constants: Dict[str, float] = field(default_factory=dict)

    # Emission properties
    emission_spectrum: Optional[np.ndarray] = None
    decay_modes: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate particle properties."""
        if self.lifetime <= 0:
            raise ValueError(f"Lifetime must be positive, got {self.lifetime}")

        if self.statistics not in ["fermion", "boson", "anyon"]:
            raise ValueError(f"Unknown statistics: {self.statistics}")


class ParticleRegistry:
    """
    Global registry for specialized particles.

    Provides lookup and management of particle properties used
    throughout the Quantum Echoes system.
    """

    def __init__(self):
        """Initialize particle registry with standard particles."""
        self._registry: Dict[ParticleType, ParticleProperties] = {}

        # Register standard particles
        self._register_standard_particles()

        logger.info("Initialized ParticleRegistry")

    def _register_standard_particles(self):
        """Register standard particle types with known properties."""

        # Photon
        self.register(ParticleProperties(
            particle_type=ParticleType.PHOTON,
            name="Photon",
            mass=0.0,
            charge=0.0,
            spin=1.0,
            lifetime=float('inf'),
            statistics="boson",
            coupling_constants={'electromagnetic': 1.0}
        ))

        # Electron
        self.register(ParticleProperties(
            particle_type=ParticleType.ELECTRON,
            name="Electron",
            mass=9.10938356e-31,
            charge=-1.0,
            spin=0.5,
            lifetime=float('inf'),
            statistics="fermion",
            coupling_constants={'electromagnetic': 1.0, 'weak': 0.1}
        ))

        # Positron
        self.register(ParticleProperties(
            particle_type=ParticleType.POSITRON,
            name="Positron",
            mass=9.10938356e-31,
            charge=1.0,
            spin=0.5,
            lifetime=float('inf'),
            statistics="fermion",
            coupling_constants={'electromagnetic': 1.0, 'weak': 0.1}
        ))

        # Fibonacci Anyon (Non-Abelian)
        self.register(ParticleProperties(
            particle_type=ParticleType.ANYON_FERMION,
            name="Fibonacci Anyon",
            mass=0.0,  # Effective mass in condensed matter system
            charge=0.0,
            spin=0.5,  # Effective spin
            lifetime=1e-3,  # ~1 ms coherence time
            statistics="anyon",
            topological_charge=1.618,  # Golden ratio
            braiding_phase=4 * np.pi / 5,  # Fibonacci braiding phase
            coupling_constants={'topological': 1.0}
        ))

        # Abelian Anyon
        self.register(ParticleProperties(
            particle_type=ParticleType.ANYON_BOSON,
            name="Abelian Anyon",
            mass=0.0,
            charge=0.0,
            spin=0.0,
            lifetime=1e-3,
            statistics="anyon",
            topological_charge=1.0,
            braiding_phase=np.pi / 4,  # Abelian phase
            coupling_constants={'topological': 0.5}
        ))

        # Majorana Fermion
        self.register(ParticleProperties(
            particle_type=ParticleType.MAJORANA_FERMION,
            name="Majorana Fermion",
            mass=0.0,  # Zero mode
            charge=0.0,  # Neutral
            spin=0.5,
            lifetime=1e-3,
            statistics="fermion",
            topological_charge=0.5,
            coupling_constants={'superconducting': 1.0}
        ))

        # Exotic Lepton (hypothetical)
        self.register(ParticleProperties(
            particle_type=ParticleType.EXOTIC_LEPTON,
            name="Exotic Lepton",
            mass=1e-28,  # Very light
            charge=-1.0,
            spin=0.5,
            lifetime=1e-6,
            statistics="fermion",
            coupling_constants={'weak': 0.01}
        ))

        # Quantum Dot Exciton
        self.register(ParticleProperties(
            particle_type=ParticleType.QUANTUM_DOT_EXCITON,
            name="Quantum Dot Exciton",
            mass=0.0,  # Bound state
            charge=0.0,  # Neutral
            spin=1.0,  # Triplet state
            lifetime=1e-9,  # ~1 ns
            statistics="boson",
            coupling_constants={'electromagnetic': 0.5}
        ))

    def register(self, properties: ParticleProperties):
        """
        Register a new particle type.

        Args:
            properties: Particle properties to register
        """
        if properties.particle_type in self._registry:
            logger.warning(f"Overwriting existing registration for "
                         f"{properties.particle_type.value}")

        self._registry[properties.particle_type] = properties
        logger.debug(f"Registered {properties.name}")

    def get(self, particle_type: ParticleType) -> Optional[ParticleProperties]:
        """
        Get properties for a particle type.

        Args:
            particle_type: Type of particle to look up

        Returns:
            ParticleProperties if registered, None otherwise
        """
        return self._registry.get(particle_type)

    def get_mass(self, particle_type: ParticleType) -> float:
        """Get mass of particle type in kg."""
        props = self.get(particle_type)
        return props.mass if props else 0.0

    def get_charge(self, particle_type: ParticleType) -> float:
        """Get charge of particle type in elementary charge units."""
        props = self.get(particle_type)
        return props.charge if props else 0.0

    def get_lifetime(self, particle_type: ParticleType) -> float:
        """Get lifetime of particle type in seconds."""
        props = self.get(particle_type)
        return props.lifetime if props else 0.0

    def is_anyon(self, particle_type: ParticleType) -> bool:
        """Check if particle type is an anyon."""
        props = self.get(particle_type)
        return props.statistics == "anyon" if props else False

    def get_topological_charge(self, particle_type: ParticleType) -> Optional[float]:
        """Get topological charge (for anyons)."""
        props = self.get(particle_type)
        return props.topological_charge if props else None

    def get_all_anyons(self) -> List[ParticleType]:
        """Get list of all registered anyon types."""
        return [
            ptype for ptype, props in self._registry.items()
            if props.statistics == "anyon"
        ]

    def get_all_particles(self) -> List[ParticleType]:
        """Get list of all registered particle types."""
        return list(self._registry.keys())

    def compute_interaction_strength(self,
                                    particle1: ParticleType,
                                    particle2: ParticleType,
                                    interaction_type: str = "electromagnetic") -> float:
        """
        Compute interaction strength between two particles.

        Args:
            particle1: First particle type
            particle2: Second particle type
            interaction_type: Type of interaction

        Returns:
            Interaction strength (dimensionless)
        """
        props1 = self.get(particle1)
        props2 = self.get(particle2)

        if not props1 or not props2:
            return 0.0

        # Get coupling constants
        coupling1 = props1.coupling_constants.get(interaction_type, 0.0)
        coupling2 = props2.coupling_constants.get(interaction_type, 0.0)

        # Interaction strength ~ product of couplings
        strength = coupling1 * coupling2

        return strength

    def print_registry(self):
        """Print all registered particles and their properties."""
        print("Particle Registry")
        print("=" * 80)

        for ptype, props in self._registry.items():
            print(f"\n{props.name} ({ptype.value}):")
            print(f"  Mass: {props.mass:.3e} kg")
            print(f"  Charge: {props.charge:+.1f} e")
            print(f"  Spin: {props.spin:.1f}")
            print(f"  Lifetime: {props.lifetime:.3e} s")
            print(f"  Statistics: {props.statistics}")

            if props.topological_charge is not None:
                print(f"  Topological Charge: {props.topological_charge:.3f}")

            if props.braiding_phase is not None:
                print(f"  Braiding Phase: {props.braiding_phase:.3f} rad")

            if props.coupling_constants:
                print(f"  Couplings: {props.coupling_constants}")


# Global registry instance
_global_registry = ParticleRegistry()


def register_particle(properties: ParticleProperties):
    """
    Register a particle in the global registry.

    Args:
        properties: Particle properties to register
    """
    _global_registry.register(properties)


def get_particle(particle_type: ParticleType) -> Optional[ParticleProperties]:
    """
    Get particle properties from global registry.

    Args:
        particle_type: Type of particle

    Returns:
        ParticleProperties if registered, None otherwise
    """
    return _global_registry.get(particle_type)


def get_registry() -> ParticleRegistry:
    """Get the global particle registry instance."""
    return _global_registry

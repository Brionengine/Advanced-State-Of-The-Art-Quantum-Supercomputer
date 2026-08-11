"""
Quantum Echoes Settings
=======================

Configuration parameters for the Quantum Echoes system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class EchoSettings:
    """
    Configuration settings for Quantum Echoes algorithm.

    Centralizes all tunable parameters for the system.
    """

    # Ion Emitter Settings
    ion_type: str = "Ytterbium-171"
    num_ions: int = 100
    trap_frequency: float = 1e6  # Hz
    ion_temperature: float = 1e-6  # Kelvin

    # Topological Qubit Settings
    num_anyons_per_qubit: int = 4
    anyon_type: str = "fibonacci"
    lattice_spacing: float = 1e-6  # meters

    # Echo Propagation Settings
    propagation_mode: str = "cavity"
    refractive_index: float = 1.0
    absorption_length: float = 1e10  # meters
    cavity_size: float = 1e-3  # meters

    # Error Correction Settings
    enable_error_correction: bool = True
    error_correction_code: str = "surface"
    code_distance: int = 5
    syndrome_measurement_interval: float = 1e-6  # seconds

    # Algorithm Settings
    default_num_qubits: int = 4
    coherence_time: float = 1e-3  # seconds
    gate_fidelity_target: float = 0.999

    # Performance Settings
    max_circuit_depth: int = 1000
    simulation_timestep: float = 1e-9  # seconds

    # Logging Settings
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'ion_emitter': {
                'ion_type': self.ion_type,
                'num_ions': self.num_ions,
                'trap_frequency': self.trap_frequency,
                'temperature': self.ion_temperature
            },
            'topological_qubits': {
                'num_anyons_per_qubit': self.num_anyons_per_qubit,
                'anyon_type': self.anyon_type,
                'lattice_spacing': self.lattice_spacing
            },
            'echo_propagation': {
                'propagation_mode': self.propagation_mode,
                'refractive_index': self.refractive_index,
                'absorption_length': self.absorption_length,
                'cavity_size': self.cavity_size
            },
            'error_correction': {
                'enabled': self.enable_error_correction,
                'code': self.error_correction_code,
                'distance': self.code_distance,
                'measurement_interval': self.syndrome_measurement_interval
            },
            'algorithm': {
                'default_num_qubits': self.default_num_qubits,
                'coherence_time': self.coherence_time,
                'gate_fidelity_target': self.gate_fidelity_target
            },
            'performance': {
                'max_circuit_depth': self.max_circuit_depth,
                'simulation_timestep': self.simulation_timestep
            },
            'logging': {
                'log_level': self.log_level,
                'detailed': self.enable_detailed_logging
            }
        }

    def configure_logging(self):
        """Configure logging based on settings."""
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.enable_detailed_logging:
            logging.getLogger('quantum_echoes').setLevel(logging.DEBUG)

    def print_settings(self):
        """Print all settings."""
        print("\n" + "=" * 60)
        print("Quantum Echoes Configuration")
        print("=" * 60)

        settings_dict = self.to_dict()

        for category, params in settings_dict.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for key, value in params.items():
                print(f"  {key}: {value}")

        print("=" * 60 + "\n")


def get_default_settings() -> EchoSettings:
    """Get default configuration settings."""
    return EchoSettings()


def get_high_fidelity_settings() -> EchoSettings:
    """Get settings optimized for high fidelity."""
    return EchoSettings(
        num_ions=500,
        ion_temperature=1e-7,
        code_distance=7,
        gate_fidelity_target=0.9999,
        coherence_time=1e-2
    )


def get_fast_simulation_settings() -> EchoSettings:
    """Get settings optimized for fast simulation."""
    return EchoSettings(
        num_ions=50,
        num_anyons_per_qubit=4,
        code_distance=3,
        enable_error_correction=False,
        simulation_timestep=1e-8
    )

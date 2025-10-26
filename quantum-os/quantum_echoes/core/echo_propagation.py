"""
Quantum Echo Propagation
=========================

Implements propagation of quantum echoes through the system.
Echoes represent quantum information encoded in particle emission patterns
that propagate and interfere constructively or destructively.

This module enables signal amplification, error detection, and quantum memory.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import logging

from .particle_emitter import EmissionPattern, ParticleType

logger = logging.getLogger(__name__)


class PropagationMode(Enum):
    """Modes of echo propagation through the quantum system."""
    FREE_SPACE = "free_space"  # Propagation in free space
    WAVEGUIDE = "waveguide"    # Guided propagation
    CAVITY = "cavity"          # Resonant cavity
    LATTICE = "lattice"        # Propagation through optical/ion lattice
    BRAGG = "bragg"            # Bragg scattering regime


@dataclass
class EchoState:
    """
    Represents the state of a propagating quantum echo.

    Contains both classical (position, momentum) and quantum (wavefunction)
    information about the echo.
    """
    position: np.ndarray  # 3D position
    momentum: np.ndarray  # 3D momentum
    wavefunction: np.ndarray  # Quantum state
    creation_time: float
    parent_pattern: Optional[EmissionPattern] = None

    # Propagation properties
    group_velocity: float = 3e8  # m/s (speed of light default)
    dispersion: float = 0.0  # Dispersion parameter
    damping: float = 1e-6  # Damping coefficient

    def __post_init__(self):
        """Initialize derived properties."""
        if len(self.position) != 3:
            self.position = np.pad(self.position, (0, 3 - len(self.position)))
        if len(self.momentum) != 3:
            self.momentum = np.pad(self.momentum, (0, 3 - len(self.momentum)))


class EchoPropagator:
    """
    Propagates quantum echoes through various media and geometries.

    Implements time evolution, interference, and boundary conditions
    for quantum echo dynamics.
    """

    def __init__(self,
                 mode: PropagationMode = PropagationMode.FREE_SPACE,
                 geometry: Optional[Dict] = None,
                 medium_properties: Optional[Dict] = None):
        """
        Initialize echo propagator.

        Args:
            mode: Propagation mode
            geometry: Dict describing system geometry
            medium_properties: Dict with refractive index, absorption, etc.
        """
        self.mode = mode
        self.geometry = geometry or {}
        self.medium_properties = medium_properties or {
            'refractive_index': 1.0,
            'absorption_length': 1e10,  # meters
            'nonlinearity': 0.0
        }

        # Active echoes in the system
        self.active_echoes: List[EchoState] = []

        # Interference patterns
        self.interference_map = np.zeros((100, 100, 100))  # 3D grid

        logger.info(f"Initialized EchoPropagator in {mode.value} mode")

    def create_echo(self,
                    emission_pattern: EmissionPattern,
                    initial_position: Optional[np.ndarray] = None,
                    initial_momentum: Optional[np.ndarray] = None) -> EchoState:
        """
        Create a propagating echo from an emission pattern.

        Args:
            emission_pattern: Source emission pattern
            initial_position: Initial 3D position
            initial_momentum: Initial 3D momentum

        Returns:
            Created echo state
        """
        # Default position and momentum
        if initial_position is None:
            initial_position = np.zeros(3)

        if initial_momentum is None:
            # Use mean spatial mode from emission
            mean_direction = np.mean(emission_pattern.spatial_modes, axis=0)
            if len(mean_direction) < 3:
                mean_direction = np.pad(mean_direction, (0, 3 - len(mean_direction)))

            # Momentum from wavelength (assume photon-like)
            hbar = 1.054571817e-34
            wavelength = 500e-9  # 500 nm default
            k = 2 * np.pi / wavelength
            initial_momentum = hbar * k * mean_direction / np.linalg.norm(mean_direction)

        # Create wavefunction from emission correlations
        wavefunction = self._emission_to_wavefunction(emission_pattern)

        # Calculate group velocity
        n = self.medium_properties['refractive_index']
        c = 3e8
        group_velocity = c / n

        # Calculate dispersion
        dispersion = self._calculate_dispersion(emission_pattern)

        echo = EchoState(
            position=initial_position.copy(),
            momentum=initial_momentum.copy(),
            wavefunction=wavefunction,
            creation_time=0.0,
            parent_pattern=emission_pattern,
            group_velocity=group_velocity,
            dispersion=dispersion,
            damping=1.0 / self.medium_properties['absorption_length']
        )

        self.active_echoes.append(echo)

        logger.debug(f"Created echo at position {initial_position}")

        return echo

    def _emission_to_wavefunction(self, pattern: EmissionPattern) -> np.ndarray:
        """
        Convert emission pattern to quantum wavefunction.

        The wavefunction encodes the quantum information from particle emissions.
        """
        # Dimension based on number of particles
        dim = len(pattern.particle_types)

        # Create superposition state from correlations
        wavefunction = np.zeros(dim, dtype=complex)

        for i in range(dim):
            # Amplitude from correlation strength
            correlations = pattern.quantum_correlations[i, :]
            amplitude = np.mean(correlations)

            # Phase from emission timing
            phase = 2 * np.pi * pattern.emission_times[i] / np.max(pattern.emission_times)

            wavefunction[i] = amplitude * np.exp(1j * phase)

        # Normalize
        wavefunction /= np.linalg.norm(wavefunction)

        return wavefunction

    def _calculate_dispersion(self, pattern: EmissionPattern) -> float:
        """
        Calculate dispersion parameter for echo propagation.

        Dispersion causes different frequency components to travel at different speeds.
        """
        # Estimate from emission time variance
        if len(pattern.emission_times) > 1:
            time_variance = np.var(pattern.emission_times)
            # Dispersion ~ variance / coherence_time^2
            dispersion = time_variance / (pattern.coherence_time ** 2)
        else:
            dispersion = 0.0

        return dispersion

    def propagate(self, echo: EchoState, time_step: float) -> EchoState:
        """
        Propagate echo for one time step.

        Updates position, momentum, and wavefunction according to
        quantum dynamics.

        Args:
            echo: Echo state to propagate
            time_step: Time step in seconds

        Returns:
            Updated echo state
        """
        # Update position (classical motion)
        velocity = echo.momentum / np.linalg.norm(echo.momentum) * echo.group_velocity
        echo.position += velocity * time_step

        # Apply boundary conditions based on geometry
        self._apply_boundary_conditions(echo)

        # Update wavefunction (quantum evolution)
        echo.wavefunction = self._evolve_wavefunction(
            echo.wavefunction,
            time_step,
            echo.dispersion
        )

        # Apply damping/absorption
        damping_factor = np.exp(-echo.damping * echo.group_velocity * time_step)
        echo.wavefunction *= np.sqrt(damping_factor)

        # Renormalize if not too small
        norm = np.linalg.norm(echo.wavefunction)
        if norm > 1e-6:
            echo.wavefunction /= norm

        return echo

    def _evolve_wavefunction(self,
                            wavefunction: np.ndarray,
                            time_step: float,
                            dispersion: float) -> np.ndarray:
        """
        Evolve wavefunction under free propagation Hamiltonian.

        Uses split-operator method for dispersion.
        """
        # Fourier transform to momentum space
        wf_k = np.fft.fft(wavefunction)

        # Apply kinetic energy operator in momentum space
        k_values = np.fft.fftfreq(len(wavefunction), d=1.0)
        hbar = 1.054571817e-34
        m_eff = 1.0  # Effective mass (normalized)

        kinetic = hbar ** 2 * k_values ** 2 / (2 * m_eff)
        dispersive = dispersion * k_values ** 4

        phase = -1j * (kinetic + dispersive) * time_step / hbar

        wf_k *= np.exp(phase)

        # Transform back to position space
        wavefunction_new = np.fft.ifft(wf_k)

        return wavefunction_new

    def _apply_boundary_conditions(self, echo: EchoState):
        """
        Apply boundary conditions based on propagation mode and geometry.

        Modifies echo position and momentum for reflections, periodic boundaries, etc.
        """
        if self.mode == PropagationMode.CAVITY:
            # Reflective boundaries for cavity
            cavity_size = self.geometry.get('size', 1e-3)  # 1 mm default

            for i in range(3):
                if abs(echo.position[i]) > cavity_size / 2:
                    # Reflect at boundary
                    echo.position[i] = np.sign(echo.position[i]) * cavity_size / 2
                    echo.momentum[i] *= -1

        elif self.mode == PropagationMode.WAVEGUIDE:
            # Confine to waveguide (1D propagation)
            waveguide_axis = self.geometry.get('axis', 0)  # x-axis default
            for i in range(3):
                if i != waveguide_axis:
                    echo.position[i] = 0
                    echo.momentum[i] = 0

        elif self.mode == PropagationMode.LATTICE:
            # Periodic boundaries for lattice
            lattice_constant = self.geometry.get('lattice_constant', 1e-6)
            echo.position = np.mod(echo.position, lattice_constant)

    def compute_interference(self,
                           echo1: EchoState,
                           echo2: EchoState) -> complex:
        """
        Compute quantum interference between two echoes.

        Returns complex interference amplitude.

        Args:
            echo1: First echo
            echo2: Second echo

        Returns:
            Complex interference amplitude
        """
        # Spatial overlap
        separation = np.linalg.norm(echo1.position - echo2.position)

        # Gaussian spatial envelope
        sigma = 1e-6  # 1 micron spatial extent
        spatial_overlap = np.exp(-separation ** 2 / (2 * sigma ** 2))

        # Quantum state overlap
        min_dim = min(len(echo1.wavefunction), len(echo2.wavefunction))
        wf1 = echo1.wavefunction[:min_dim]
        wf2 = echo2.wavefunction[:min_dim]

        quantum_overlap = np.vdot(wf1, wf2)

        # Total interference amplitude
        interference = spatial_overlap * quantum_overlap

        return interference

    def compute_total_interference_pattern(self) -> np.ndarray:
        """
        Compute total interference pattern from all active echoes.

        Returns 3D array of interference intensity.
        """
        # Grid dimensions
        grid_size = self.interference_map.shape
        x = np.linspace(-1e-3, 1e-3, grid_size[0])
        y = np.linspace(-1e-3, 1e-3, grid_size[1])
        z = np.linspace(-1e-3, 1e-3, grid_size[2])

        # Reset interference map
        self.interference_map = np.zeros(grid_size, dtype=complex)

        # Add contribution from each echo
        for echo in self.active_echoes:
            # Find closest grid point
            ix = np.argmin(np.abs(x - echo.position[0]))
            iy = np.argmin(np.abs(y - echo.position[1]))
            iz = np.argmin(np.abs(z - echo.position[2]))

            # Add wavefunction amplitude
            amplitude = np.sum(echo.wavefunction)
            self.interference_map[ix, iy, iz] += amplitude

        # Compute intensity (|amplitude|^2)
        intensity = np.abs(self.interference_map) ** 2

        return intensity

    def amplify_echo(self, echo: EchoState, gain: float = 2.0) -> EchoState:
        """
        Amplify echo using constructive interference.

        Creates a coherent copy and adds it to the original, increasing amplitude.

        Args:
            echo: Echo to amplify
            gain: Amplification factor

        Returns:
            Amplified echo
        """
        # Increase wavefunction amplitude
        echo.wavefunction *= np.sqrt(gain)

        # Add quantum noise (fundamental limit)
        noise_level = np.sqrt(gain - 1) * 0.1
        noise = (np.random.randn(len(echo.wavefunction)) +
                1j * np.random.randn(len(echo.wavefunction))) * noise_level

        echo.wavefunction += noise

        # Renormalize
        echo.wavefunction /= np.linalg.norm(echo.wavefunction)

        logger.debug(f"Amplified echo by factor {gain}")

        return echo

    def focus_echo(self,
                   echo: EchoState,
                   target_position: np.ndarray,
                   focusing_time: float = 1e-6) -> EchoState:
        """
        Focus echo to a target position.

        Adjusts momentum to converge at target.

        Args:
            echo: Echo to focus
            target_position: Target 3D position
            focusing_time: Time to reach target

        Returns:
            Echo with adjusted momentum
        """
        # Calculate required momentum
        displacement = target_position - echo.position
        required_velocity = displacement / focusing_time

        # Update momentum
        momentum_magnitude = np.linalg.norm(echo.momentum)
        echo.momentum = required_velocity * momentum_magnitude / echo.group_velocity

        logger.debug(f"Focused echo to position {target_position}")

        return echo

    def measure_echo_intensity(self, echo: EchoState) -> float:
        """
        Measure intensity of echo (probability density).

        Returns:
            Intensity (normalized to 1 for initial state)
        """
        intensity = np.sum(np.abs(echo.wavefunction) ** 2)
        return float(intensity)

    def create_standing_wave(self,
                            echo_forward: EchoState,
                            echo_backward: EchoState) -> np.ndarray:
        """
        Create standing wave from forward and backward propagating echoes.

        Useful for creating stable quantum memory.

        Args:
            echo_forward: Forward propagating echo
            echo_backward: Backward propagating echo

        Returns:
            Standing wave pattern (1D array)
        """
        # Ensure momenta are opposite
        if np.dot(echo_forward.momentum, echo_backward.momentum) > 0:
            echo_backward.momentum *= -1

        # Create standing wave from interference
        min_dim = min(len(echo_forward.wavefunction), len(echo_backward.wavefunction))

        wf_forward = echo_forward.wavefunction[:min_dim]
        wf_backward = echo_backward.wavefunction[:min_dim]

        # Standing wave = sum of counter-propagating waves
        standing_wave = (wf_forward + wf_backward) / np.sqrt(2)

        logger.info("Created standing wave pattern")

        return standing_wave

    def detect_echo_arrival(self,
                          detector_position: np.ndarray,
                          detection_radius: float = 1e-6) -> List[EchoState]:
        """
        Detect echoes arriving at a specific position.

        Args:
            detector_position: 3D detector position
            detection_radius: Detection radius

        Returns:
            List of detected echoes
        """
        detected = []

        for echo in self.active_echoes:
            separation = np.linalg.norm(echo.position - detector_position)

            if separation <= detection_radius:
                detected.append(echo)

        logger.debug(f"Detected {len(detected)} echoes at position {detector_position}")

        return detected

    def remove_weak_echoes(self, threshold: float = 1e-3):
        """
        Remove echoes that have decayed below threshold.

        Cleans up the active echo list.

        Args:
            threshold: Minimum intensity to keep echo active
        """
        initial_count = len(self.active_echoes)

        self.active_echoes = [
            echo for echo in self.active_echoes
            if self.measure_echo_intensity(echo) > threshold
        ]

        removed = initial_count - len(self.active_echoes)

        if removed > 0:
            logger.debug(f"Removed {removed} weak echoes")

    def simulate_propagation(self,
                            duration: float,
                            time_step: float = 1e-9,
                            callback: Optional[Callable] = None) -> List[EchoState]:
        """
        Simulate echo propagation for a specified duration.

        Args:
            duration: Total simulation time (seconds)
            time_step: Time step for integration (seconds)
            callback: Optional callback function called each step

        Returns:
            Final list of active echoes
        """
        num_steps = int(duration / time_step)

        logger.info(f"Starting propagation simulation for {duration*1e6:.2f} µs "
                   f"with {len(self.active_echoes)} echoes")

        for step in range(num_steps):
            # Propagate each echo
            for echo in self.active_echoes:
                self.propagate(echo, time_step)

            # Remove weak echoes
            if step % 100 == 0:
                self.remove_weak_echoes()

            # Call user callback
            if callback is not None:
                callback(step, self.active_echoes)

        logger.info(f"Propagation complete. {len(self.active_echoes)} echoes remaining")

        return self.active_echoes

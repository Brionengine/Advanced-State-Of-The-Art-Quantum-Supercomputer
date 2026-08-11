"""
Quantum Sensing Application
===========================

Ultra-precise measurements using quantum echoes for enhanced sensitivity.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.particle_emitter import SpecializedIonEmitter
from core.echo_propagation import EchoPropagator, PropagationMode
from algorithms.quantum_echoes import QuantumEchoesAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class PrecisionMeasurement:
    """Result of quantum sensing measurement."""
    measured_value: float
    uncertainty: float
    signal_to_noise: float
    measurement_time: float


class QuantumSensor:
    """
    Quantum sensor using echo-enhanced interferometry.

    Achieves beyond-classical sensitivity for physical measurements.
    """

    def __init__(self,
                 sensor_type: str = "magnetometer",
                 sensitivity_target: float = 1e-15):
        """
        Initialize quantum sensor.

        Args:
            sensor_type: Type of sensor ('magnetometer', 'gravimeter', 'clock')
            sensitivity_target: Target sensitivity
        """
        self.sensor_type = sensor_type
        self.sensitivity_target = sensitivity_target

        # Initialize quantum system
        self.emitter = SpecializedIonEmitter(num_ions=200)
        self.propagator = EchoPropagator(mode=PropagationMode.CAVITY)

        # Measurement history
        self.measurement_history = []

        logger.info(f"Initialized {sensor_type} with target sensitivity "
                   f"{sensitivity_target:.2e}")

    def measure(self,
                integration_time: float = 1e-3,
                num_echoes: int = 10) -> PrecisionMeasurement:
        """
        Perform quantum-enhanced measurement.

        Args:
            integration_time: Measurement integration time (seconds)
            num_echoes: Number of echo sequences for enhancement

        Returns:
            PrecisionMeasurement result
        """
        logger.info(f"Starting {self.sensor_type} measurement")

        # Create initial emission pattern
        pattern = self.emitter.emit_particle_cascade(num_particles=20)

        # Create and propagate echoes
        echoes = []
        for i in range(num_echoes):
            echo = self.propagator.create_echo(pattern)
            echoes.append(echo)

        # Propagate all echoes
        self.propagator.simulate_propagation(
            duration=integration_time,
            time_step=1e-9
        )

        # Compute interference pattern for signal extraction
        interference = self.propagator.compute_total_interference_pattern()

        # Extract measurement value from interference
        measured_value = self._extract_signal(interference)

        # Compute uncertainty (Heisenberg limited)
        uncertainty = self._compute_quantum_uncertainty(num_echoes, integration_time)

        # Signal-to-noise ratio
        snr = measured_value / uncertainty if uncertainty > 0 else float('inf')

        result = PrecisionMeasurement(
            measured_value=measured_value,
            uncertainty=uncertainty,
            signal_to_noise=snr,
            measurement_time=integration_time
        )

        self.measurement_history.append(result)

        logger.info(f"Measurement complete: {measured_value:.3e} ± {uncertainty:.3e}")

        return result

    def _extract_signal(self, interference_pattern: np.ndarray) -> float:
        """
        Extract measured signal from interference pattern.

        Uses Fourier analysis of fringe pattern.
        """
        # Flatten 3D interference to 1D projection
        projection = np.sum(interference_pattern, axis=(1, 2))

        # FFT to extract dominant frequency
        spectrum = np.fft.fft(projection)
        frequencies = np.fft.fftfreq(len(projection))

        # Find peak frequency (encoded signal)
        peak_idx = np.argmax(np.abs(spectrum[1:])) + 1
        peak_freq = frequencies[peak_idx]

        # Convert frequency to physical quantity based on sensor type
        if self.sensor_type == "magnetometer":
            # Larmor frequency → magnetic field
            gamma = 2.8e10  # Hz/Tesla (gyromagnetic ratio)
            measured_value = abs(peak_freq) / gamma

        elif self.sensor_type == "gravimeter":
            # Acceleration from phase shift
            measured_value = abs(peak_freq) * 9.8  # m/s^2

        elif self.sensor_type == "clock":
            # Frequency standard
            measured_value = abs(peak_freq)

        else:
            measured_value = abs(peak_freq)

        return float(measured_value)

    def _compute_quantum_uncertainty(self,
                                    num_echoes: int,
                                    integration_time: float) -> float:
        """
        Compute quantum uncertainty (Heisenberg limit).

        Args:
            num_echoes: Number of echoes used
            integration_time: Integration time

        Returns:
            Quantum-limited uncertainty
        """
        # Heisenberg uncertainty: Δφ ~ 1/N where N = number of particles
        N_eff = num_echoes * 20  # particles per echo

        # Shot noise limit
        shot_noise = 1.0 / np.sqrt(N_eff)

        # Time averaging improvement
        time_factor = 1.0 / np.sqrt(integration_time * 1e6)

        # Total uncertainty
        uncertainty = shot_noise * time_factor * self.sensitivity_target

        return float(uncertainty)

    def calibrate(self, known_value: float, num_measurements: int = 10) -> float:
        """
        Calibrate sensor using known reference.

        Args:
            known_value: Known reference value
            num_measurements: Number of calibration measurements

        Returns:
            Calibration error
        """
        measurements = []

        for i in range(num_measurements):
            result = self.measure()
            measurements.append(result.measured_value)

        mean_measured = np.mean(measurements)
        calibration_error = abs(mean_measured - known_value)

        logger.info(f"Calibration complete: error = {calibration_error:.3e}")

        return float(calibration_error)

    def get_sensitivity(self) -> float:
        """
        Estimate current sensitivity from measurement history.

        Returns:
            Estimated sensitivity
        """
        if not self.measurement_history:
            return self.sensitivity_target

        uncertainties = [m.uncertainty for m in self.measurement_history]
        return float(np.mean(uncertainties))

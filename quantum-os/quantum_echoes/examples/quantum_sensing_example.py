#!/usr/bin/env python3
"""
Quantum Sensing Example
========================

Demonstrates ultra-precise magnetometry using quantum echoes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from applications.quantum_sensing import QuantumSensor
import numpy as np


def main():
    print("\n" + "=" * 60)
    print("QUANTUM SENSING EXAMPLE - Magnetometry")
    print("=" * 60 + "\n")

    # Initialize quantum magnetometer
    print("1. Initializing quantum magnetometer...")
    print("   Target sensitivity: 1 femtotesla (10⁻¹⁵ T)")

    sensor = QuantumSensor(
        sensor_type="magnetometer",
        sensitivity_target=1e-15  # 1 femtotesla
    )

    # Perform measurements
    print("\n2. Performing quantum-enhanced measurements...")
    print("   Integration time: 1 millisecond per measurement")
    print("   Number of quantum echoes: 10\n")

    measurements = []

    for i in range(5):
        print(f"   Measurement {i+1}/5...")
        result = sensor.measure(
            integration_time=1e-3,
            num_echoes=10
        )

        measurements.append(result)

        print(f"      Magnetic field: {result.measured_value:.6e} T")
        print(f"      Uncertainty: ±{result.uncertainty:.6e} T")
        print(f"      SNR: {result.signal_to_noise:.2f}")
        print(f"      Measurement time: {result.measurement_time*1e3:.2f} ms\n")

    # Calculate statistics
    print("3. Measurement Statistics:")
    values = [m.measured_value for m in measurements]
    uncertainties = [m.uncertainty for m in measurements]

    mean_field = np.mean(values)
    std_field = np.std(values)
    mean_uncertainty = np.mean(uncertainties)

    print(f"   Mean field: {mean_field:.6e} T")
    print(f"   Std deviation: {std_field:.6e} T")
    print(f"   Average uncertainty: {mean_uncertainty:.6e} T")

    # Sensor performance
    achieved_sensitivity = sensor.get_sensitivity()
    print(f"\n4. Sensor Performance:")
    print(f"   Achieved sensitivity: {achieved_sensitivity:.6e} T")
    print(f"   Target sensitivity: {sensor.sensitivity_target:.6e} T")

    improvement = sensor.sensitivity_target / achieved_sensitivity
    print(f"   Performance vs target: {improvement:.2f}x")

    # Calibration example
    print("\n5. Calibrating with known reference field...")
    known_field = 1e-12  # 1 picotesla

    calibration_error = sensor.calibrate(
        known_value=known_field,
        num_measurements=5
    )

    print(f"   Reference field: {known_field:.6e} T")
    print(f"   Calibration error: {calibration_error:.6e} T")
    print(f"   Calibration accuracy: {(1 - calibration_error/known_field)*100:.2f}%")

    print("\n" + "=" * 60)
    print("Quantum sensing example complete!")
    print("\nApplications:")
    print("  - Medical imaging (MEG, MRI)")
    print("  - Geophysical surveying")
    print("  - Space exploration")
    print("  - Materials characterization")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quantum Echoes Demonstration Program
====================================

Comprehensive demonstration of the Quantum Echoes framework for
topological quantum computation using specialized ion emissions.

This demo showcases:
1. Particle emission and echo creation
2. Topological qubit initialization
3. Quantum circuit execution
4. Error correction
5. Practical quantum sensing application

Author: Quantum Echoes Development Team
Version: 1.0.0
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import Quantum Echoes components
from core.particle_emitter import SpecializedIonEmitter, ParticleType
from core.topological_qubit import TopologicalQubit, AnyonType
from core.echo_propagation import EchoPropagator, PropagationMode
from core.particle_registry import get_registry

from algorithms.quantum_echoes import QuantumEchoesAlgorithm
from circuits.echo_circuit_builder import build_circuit, create_bell_circuit, create_ghz_circuit
from error_correction.topological_protection import TopologicalErrorCorrection
from applications.quantum_sensing import QuantumSensor

from config.echo_settings import get_default_settings


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_particle_emission():
    """Demonstrate specialized ion particle emission."""
    print_header("DEMO 1: Specialized Ion Particle Emission")

    print("Initializing Ytterbium-171 ion emitter with 100 ions...")
    emitter = SpecializedIonEmitter(
        ion_type="Ytterbium-171",
        num_ions=100,
        trap_frequency=1e6,
        temperature=1e-6
    )

    print("\nEmitting particle cascade...")
    pattern = emitter.emit_particle_cascade(
        num_particles=15,
        encoding_data=np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1])
    )

    print(f"\nEmission Pattern Statistics:")
    stats = emitter.measure_particle_statistics(pattern)
    print(f"  - Particle types emitted: {len(set(pattern.particle_types))}")
    print(f"  - Mean emission interval: {stats['mean_interval']*1e9:.2f} ns")
    print(f"  - Spatial extent: {stats['spatial_extent']:.2e} m")
    print(f"  - Mean correlation: {stats['mean_correlation']:.3f}")
    print(f"  - Entanglement degree: {stats['entanglement_degree']:.3f}")
    print(f"  - Coherence time: {stats['coherence_time']*1e6:.2f} µs")

    print("\nCreating quantum echo from emission pattern...")
    echo_pattern = emitter.create_echo_from_pattern(pattern, delay_time=1e-6)
    echo_stats = emitter.measure_particle_statistics(echo_pattern)
    print(f"  - Echo entanglement: {echo_stats['entanglement_degree']:.3f}")
    print(f"  - Echo coherence time: {echo_stats['coherence_time']*1e6:.2f} µs")

    print("\n✓ Particle emission demonstration complete!")
    return emitter, pattern


def demo_topological_qubits(emission_pattern):
    """Demonstrate topological qubit creation and manipulation."""
    print_header("DEMO 2: Topological Qubit Operations")

    print("Creating topological qubit from emission pattern...")
    qubit = TopologicalQubit(
        num_anyons=4,
        anyon_type=AnyonType.FIBONACCI,
        lattice_spacing=1e-6
    )

    qubit.create_from_emission(emission_pattern)
    print(f"  - Number of anyons: {qubit.num_anyons}")
    print(f"  - Anyon type: {qubit.anyon_type.value}")
    print(f"  - Topological charge: {qubit.measure_topological_charge()}")

    print("\nApplying topological gates via anyonic braiding...")

    print("  1. Hadamard gate (H)")
    qubit.apply_topological_gate('H')

    print("  2. Phase gate (T)")
    qubit.apply_topological_gate('T')

    print("  3. Pauli-X gate (X)")
    qubit.apply_topological_gate('X')

    print(f"\n  Total braiding operations: {len(qubit.braiding_history)}")

    # Get logical qubit state
    logical_state = qubit.get_logical_state()
    print(f"\nLogical qubit state:")
    print(f"  |0⟩ amplitude: {logical_state[0]:.3f}")
    print(f"  |1⟩ amplitude: {logical_state[1]:.3f}")
    print(f"  Probabilities: |0⟩={abs(logical_state[0])**2:.3f}, |1⟩={abs(logical_state[1])**2:.3f}")

    # Visualize configuration
    vis_data = qubit.visualize_anyon_configuration()
    print(f"\nAnyon configuration summary:")
    print(f"  - Number of braiding operations: {vis_data['braiding_count']}")
    print(f"  - Topological charge: {vis_data['topological_charge']}")

    print("\n✓ Topological qubit demonstration complete!")
    return qubit


def demo_echo_propagation(emission_pattern):
    """Demonstrate quantum echo propagation."""
    print_header("DEMO 3: Quantum Echo Propagation")

    print("Initializing echo propagator in cavity mode...")
    propagator = EchoPropagator(
        mode=PropagationMode.CAVITY,
        geometry={'size': 1e-3},
        medium_properties={
            'refractive_index': 1.0,
            'absorption_length': 1e10
        }
    )

    print("\nCreating propagating echoes...")
    echo1 = propagator.create_echo(emission_pattern)
    echo2 = propagator.create_echo(emission_pattern,
                                   initial_momentum=np.array([0, 0, -1e-24]))

    print(f"  - Active echoes: {len(propagator.active_echoes)}")
    print(f"  - Echo 1 intensity: {propagator.measure_echo_intensity(echo1):.3f}")
    print(f"  - Echo 2 intensity: {propagator.measure_echo_intensity(echo2):.3f}")

    print("\nComputing echo interference...")
    interference = propagator.compute_interference(echo1, echo2)
    print(f"  - Interference amplitude: {abs(interference):.3f}")
    print(f"  - Interference phase: {np.angle(interference):.3f} rad")

    print("\nSimulating echo propagation (10 µs)...")
    propagator.simulate_propagation(duration=10e-6, time_step=1e-9)

    print(f"  - Remaining echoes: {len(propagator.active_echoes)}")

    print("\n✓ Echo propagation demonstration complete!")
    return propagator


def demo_quantum_algorithm():
    """Demonstrate quantum circuit execution."""
    print_header("DEMO 4: Quantum Echoes Algorithm")

    print("Initializing Quantum Echoes Algorithm with 3 qubits...")
    qe = QuantumEchoesAlgorithm(
        num_qubits=3,
        ion_trap_size=100,
        error_correction=True
    )

    print("\nInitializing qubits to |000⟩ state...")
    qe.initialize_qubits('zero')

    print("\nBuilding GHZ state preparation circuit...")
    print("  Circuit: H(0) → CNOT(0,1) → CNOT(0,2) → Measure all")

    circuit = build_circuit(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(0, 2)
    circuit.measure_all()

    print(f"\n  Circuit depth: {circuit.build().depth()}")
    print(f"  Number of gates: {len(circuit.build().gates)}")

    print("\nExecuting circuit...")
    results = qe.run_circuit(circuit.build())

    print(f"\nCircuit Execution Results:")
    print(f"  - Final state vector dimension: {len(results['final_state'])}")
    print(f"  - Measurement outcomes: {results['measurements']}")
    print(f"\nExecution Metrics:")
    print(f"  - Gates applied: {results['metrics']['gate_count']}")
    print(f"  - Particle emissions: {results['metrics']['emission_count']}")
    print(f"  - Echoes created: {results['metrics']['echo_count']}")

    # Compute fidelity with ideal GHZ state
    ideal_ghz = np.zeros(8, dtype=complex)
    ideal_ghz[0] = 1/np.sqrt(2)
    ideal_ghz[-1] = 1/np.sqrt(2)

    fidelity = qe.get_fidelity(ideal_ghz)
    print(f"\n  State fidelity with |GHZ⟩: {fidelity:.4f}")

    print("\n✓ Quantum algorithm demonstration complete!")
    return qe


def demo_error_correction():
    """Demonstrate topological error correction."""
    print_header("DEMO 5: Topological Error Correction")

    print("Initializing Surface Code with distance 5...")
    error_correction = TopologicalErrorCorrection(
        code_distance=5,
        code_type='surface'
    )

    print(f"\n  Code parameters:")
    print(f"  - Distance: {error_correction.code_distance}")
    print(f"  - Data qubits: {error_correction.code.num_data_qubits}")
    print(f"  - Ancilla qubits: {error_correction.code.num_ancilla_qubits}")
    print(f"  - X stabilizers: {len(error_correction.code.x_stabilizers)}")
    print(f"  - Z stabilizers: {len(error_correction.code.z_stabilizers)}")

    print("\nSimulating error correction cycles...")

    # Simulate qubit states with errors
    qubit_states = np.random.randn(error_correction.code.num_data_qubits)

    for cycle in range(5):
        print(f"\n  Cycle {cycle + 1}:")
        corrected_states = error_correction.run_error_correction_cycle(qubit_states)
        print(f"    - Errors detected: {error_correction.stats['errors_detected']}")
        print(f"    - Errors corrected: {error_correction.stats['errors_corrected']}")

    print(f"\n  Logical error rate: {error_correction.get_logical_error_rate():.2e}")

    print("\n✓ Error correction demonstration complete!")
    return error_correction


def demo_quantum_sensing():
    """Demonstrate quantum sensing application."""
    print_header("DEMO 6: Quantum Sensing Application")

    print("Initializing quantum magnetometer...")
    sensor = QuantumSensor(
        sensor_type="magnetometer",
        sensitivity_target=1e-15  # Tesla
    )

    print(f"\n  Sensor type: {sensor.sensor_type}")
    print(f"  Target sensitivity: {sensor.sensitivity_target:.2e} T")

    print("\nPerforming quantum-enhanced measurements...")

    for i in range(3):
        print(f"\n  Measurement {i + 1}:")
        result = sensor.measure(
            integration_time=1e-3,
            num_echoes=10
        )

        print(f"    - Measured field: {result.measured_value:.3e} T")
        print(f"    - Uncertainty: {result.uncertainty:.3e} T")
        print(f"    - Signal-to-noise: {result.signal_to_noise:.2f}")
        print(f"    - Measurement time: {result.measurement_time*1e3:.2f} ms")

    print(f"\n  Current sensitivity: {sensor.get_sensitivity():.3e} T")

    print("\n✓ Quantum sensing demonstration complete!")
    return sensor


def demo_particle_registry():
    """Demonstrate particle registry."""
    print_header("DEMO 7: Particle Registry")

    print("Accessing global particle registry...")
    registry = get_registry()

    print("\nRegistered particles:")
    all_particles = registry.get_all_particles()
    print(f"  Total particles: {len(all_particles)}\n")

    for ptype in all_particles[:5]:  # Show first 5
        props = registry.get(ptype)
        if props:
            print(f"  {props.name} ({ptype.value}):")
            print(f"    - Mass: {props.mass:.3e} kg")
            print(f"    - Charge: {props.charge:+.1f} e")
            print(f"    - Spin: {props.spin}")
            print(f"    - Statistics: {props.statistics}")
            if props.topological_charge:
                print(f"    - Topological charge: {props.topological_charge:.3f}")
            print()

    print("  (... and more)")

    # Show anyonic particles
    anyons = registry.get_all_anyons()
    print(f"\nAnyonic particles: {len(anyons)}")
    for anyon in anyons:
        props = registry.get(anyon)
        print(f"  - {props.name}: charge={props.topological_charge:.3f}")

    print("\n✓ Particle registry demonstration complete!")


def main():
    """Run full Quantum Echoes demonstration."""

    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║        QUANTUM ECHOES DEMONSTRATION PROGRAM                      ║")
    print("║                                                                   ║")
    print("║    Topological Quantum Computing with Ion Emissions              ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    print("\nThis demonstration showcases the complete Quantum Echoes framework")
    print("for practical topological quantum computation.\n")

    try:
        # Run demonstrations
        emitter, pattern = demo_particle_emission()
        qubit = demo_topological_qubits(pattern)
        propagator = demo_echo_propagation(pattern)
        algorithm = demo_quantum_algorithm()
        error_correction = demo_error_correction()
        sensor = demo_quantum_sensing()
        demo_particle_registry()

        # Summary
        print_header("DEMONSTRATION SUMMARY")

        print("✓ All demonstrations completed successfully!\n")

        print("Components tested:")
        print("  1. Specialized ion particle emission ✓")
        print("  2. Topological qubit operations ✓")
        print("  3. Quantum echo propagation ✓")
        print("  4. Quantum circuit execution ✓")
        print("  5. Topological error correction ✓")
        print("  6. Quantum sensing application ✓")
        print("  7. Particle registry ✓")

        print("\nKey Results:")
        print(f"  - Emission entanglement: {emitter.measure_particle_statistics(pattern)['entanglement_degree']:.3f}")
        print(f"  - Topological qubits created: 3")
        print(f"  - Quantum gates executed: {algorithm.metrics['gate_count']}")
        print(f"  - Errors corrected: {error_correction.stats['errors_corrected']}")
        print(f"  - Sensing precision: {sensor.get_sensitivity():.2e} T")

        print("\n" + "=" * 70)
        print("  Thank you for exploring Quantum Echoes!")
        print("  For more information, see the documentation.")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        print(f"\n✗ Demonstration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

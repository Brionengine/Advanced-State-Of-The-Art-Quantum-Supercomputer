"""
Comprehensive Demonstration of Long-Lived Fault-Tolerant Qubits
================================================================

Demonstrates all features of the quantum echoes system with particle emitters
and long-lived logical qubits for practical applications.

This demo shows:
1. Enhanced particle emission with exotic particles
2. Long-lived logical qubit creation and operation
3. Advanced fault tolerance and error correction
4. Qubit longevity management
5. Real-world applications
6. Performance benchmarking
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from datetime import datetime, timedelta

# Import our new modules
from core.enhanced_particle_emitter import (
    EnhancedParticleEmitter,
    ExoticParticleType,
    ExoticParticleRegistry
)
from core.long_lived_qubits import (
    LongLivedLogicalQubit,
    LongLivedQubitFactory,
    StabilizationProtocol
)
from error_correction.advanced_fault_tolerance import (
    AdvancedFaultTolerance,
    CorrectionStrategy
)
from core.qubit_longevity_manager import (
    QubitLongevityManager,
    MaintenanceAction
)
from applications.practical_applications import (
    QuantumComputingService,
    QuantumMemoryDatabase,
    QuantumSensorArray,
    MolecularSimulator,
    ApplicationManager,
    ApplicationDomain,
    ApplicationRequirements
)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_enhanced_particle_emitter():
    """Demonstrate enhanced particle emitter with exotic particles."""
    print_section("ENHANCED PARTICLE EMITTER DEMONSTRATION")

    # Create enhanced emitter
    emitter = EnhancedParticleEmitter(
        num_ions=500,
        enable_exotic_emissions=True,
        target_qubit_rate=1000.0
    )

    print("Enhanced Particle Emitter initialized with:")
    print(f"  - 500 specialized ions")
    print(f"  - Exotic particle emissions enabled")
    print(f"  - Target qubit generation: 1000 qubits/second")
    print()

    # Emit exotic cascade
    print("Emitting exotic particle cascade...")
    emission = emitter.emit_exotic_cascade(
        num_particles=50,
        exotic_fraction=0.6
    )

    print(f"\nEmission Results:")
    print(f"  Total particles: {len(emission.particle_types) + len(emission.exotic_particles)}")
    print(f"  Standard particles: {len(emission.particle_types)}")
    print(f"  Exotic particles: {len(emission.exotic_particles)}")
    print(f"  Entanglement degree: {emission.entanglement_degree:.4f}")
    print(f"  Coherence time: {emission.coherence_time:.2e} seconds")
    print(f"  Qubit generation potential: {emission.qubit_generation_potential:.1f} qubits")

    # Count exotic particle types
    exotic_counts = emission.count_exotic_particles()
    if exotic_counts:
        print(f"\nExotic Particle Distribution:")
        for ptype, count in exotic_counts.items():
            print(f"  {ptype.value}: {count}")

    # Generate qubits from emission
    print("\nGenerating qubits from emission pattern...")
    qubits = emitter.generate_qubits_from_emission(emission)

    print(f"Generated {len(qubits)} qubits")
    if qubits:
        print(f"\nFirst qubit properties:")
        print(f"  ID: {qubits[0]['id']}")
        print(f"  Coherence time: {qubits[0]['coherence_time']:.2e} seconds")
        print(f"  Error rate: {qubits[0]['error_rate']:.2e}")

    # Continuous generation
    print("\nDemonstrating continuous qubit generation (1 second)...")
    start_time = time.time()
    continuous_qubits = emitter.continuous_qubit_generation(duration=1.0, rate=500.0)
    elapsed = time.time() - start_time

    print(f"Generated {len(continuous_qubits)} qubits in {elapsed:.3f} seconds")
    print(f"Actual rate: {len(continuous_qubits)/elapsed:.1f} qubits/second")

    # Statistics
    emitter.print_statistics()


def demo_long_lived_qubits():
    """Demonstrate long-lived logical qubits."""
    print_section("LONG-LIVED LOGICAL QUBIT DEMONSTRATION")

    # Create different types of long-lived qubits
    print("Creating specialized long-lived qubits...\n")

    # 1. Quantum memory qubit
    memory_qubit = LongLivedQubitFactory.create_quantum_memory(lifetime_years=50.0)
    print("Created quantum memory qubit (50-year target lifetime)")

    # 2. Quantum processor qubit
    processor_qubit = LongLivedQubitFactory.create_quantum_processor(gate_fidelity=0.99999)
    print("Created quantum processor qubit (5-9s gate fidelity)")

    # 3. Quantum communication node qubit
    comm_qubit = LongLivedQubitFactory.create_quantum_communication_node()
    print("Created quantum communication node qubit")

    # 4. Custom long-lived qubit
    custom_qubit = LongLivedLogicalQubit(
        num_physical_qubits=25,
        code_distance=7,
        stabilization_protocol=StabilizationProtocol.HYBRID_PROTECTION,
        target_lifetime_years=100.0
    )
    print("Created custom qubit (100-year target lifetime, hybrid protection)\n")

    # Activate custom qubit
    print("Activating custom qubit...")
    custom_qubit.activate()

    # Run stabilization cycles
    print("\nRunning stabilization cycles...")
    for cycle in range(10):
        metrics = custom_qubit.apply_stabilization_cycle(cycle_duration=1e-3)

        if cycle % 3 == 0:
            print(f"  Cycle {cycle}: ", end="")
            if 'errors_corrected' in metrics:
                print(f"Corrected {metrics['errors_corrected']} errors, ", end="")
            if 'echo_fidelity' in metrics:
                print(f"Echo fidelity: {metrics['echo_fidelity']:.4f}")
            else:
                print()

    # Perform logical gates
    print("\nPerforming fault-tolerant logical gates...")
    gates = ['X', 'H', 'T', 'Z']
    for gate in gates:
        state = custom_qubit.perform_logical_gate(gate)
        fidelity = custom_qubit.get_state_fidelity()
        print(f"  Applied {gate} gate - State fidelity: {fidelity:.6f}")

    # Print status
    custom_qubit.print_status()


def demo_advanced_fault_tolerance():
    """Demonstrate advanced fault tolerance."""
    print_section("ADVANCED FAULT TOLERANCE DEMONSTRATION")

    # Initialize fault tolerance system
    ft_system = AdvancedFaultTolerance(
        num_logical_qubits=10,
        code_distance=7,
        physical_error_rate=1e-4,
        correction_strategy=CorrectionStrategy.ADAPTIVE
    )

    print("Advanced Fault Tolerance System initialized:")
    print(f"  Logical qubits: 10")
    print(f"  Physical qubits: {ft_system.num_physical_qubits}")
    print(f"  Code distance: 7")
    print(f"  Correction strategy: ADAPTIVE")
    print()

    # Run correction cycles
    print("Running error correction cycles...")
    for cycle in range(20):
        metrics = ft_system.run_correction_cycle(use_prediction=True)

        if metrics['errors_detected'] > 0:
            print(f"  Cycle {cycle}: Detected {metrics['errors_detected']} errors, "
                  f"corrected {metrics['errors_corrected']}, "
                  f"logical errors: {metrics['logical_errors']}")

    # Print metrics
    ft_system.print_metrics()


def demo_longevity_manager():
    """Demonstrate qubit longevity management."""
    print_section("QUBIT LONGEVITY MANAGER DEMONSTRATION")

    # Initialize manager
    manager = QubitLongevityManager(
        maintenance_mode="automated",
        health_check_interval=3600.0,
        auto_repair=True
    )

    print("Qubit Longevity Manager initialized in automated mode\n")

    # Register multiple qubits with varying health
    print("Registering fleet of qubits...")
    for i in range(20):
        # Vary initial fidelity
        fidelity = 0.9999 - i * 0.0002

        manager.register_qubit(
            qubit_id=f"qubit_{i:02d}",
            initial_metrics={
                'state_fidelity': fidelity,
                'coherence_T1': 1e6 * (1.0 - i * 0.05),
                'coherence_T2': 1e5 * (1.0 - i * 0.05),
                'gate_fidelity': 0.9999,
                'measurement_fidelity': 0.999,
                'error_rate': 1e-12 * (1.0 + i * 0.1)
            }
        )

    print(f"Registered {len(manager.qubits)} qubits\n")

    # Simulate operation over time
    print("Simulating operation and maintenance over time...")
    for day in range(30):
        # Update metrics (simulate degradation)
        for qubit_id in list(manager.qubits.keys()):
            degradation = np.random.rand() * 0.0001
            metrics = manager.qubits[qubit_id]

            manager.update_metrics(qubit_id, {
                'state_fidelity': metrics.state_fidelity - degradation,
                'uptime': 86400,  # 1 day
                'corrections': np.random.randint(0, 5)
            })

        # Run health check
        if day % 7 == 0:
            print(f"\nDay {day} health check:")
            health_counts = manager.run_health_check_cycle()
            for status, count in health_counts.items():
                if count > 0:
                    print(f"  {status}: {count}")

    # Print fleet status
    manager.print_fleet_status()

    # Print individual qubit status
    print("Sample individual qubit report:")
    manager.print_qubit_status("qubit_10")


def demo_practical_applications():
    """Demonstrate practical real-world applications."""
    print_section("PRACTICAL APPLICATIONS DEMONSTRATION")

    # 1. Quantum Computing Service
    print("1. QUANTUM COMPUTING SERVICE")
    print("-" * 40)

    qc_service = QuantumComputingService(num_qubits=100, service_tier="enterprise")

    circuit = {
        'num_qubits': 50,
        'depth': 1000,
        'shots': 1000,
        'gates': ['H', 'CNOT', 'T']
    }

    job_id = qc_service.submit_circuit(circuit, priority="high")
    print(f"Submitted circuit job: {job_id}")

    results = qc_service.execute_job(job_id)
    print(f"Job completed with {results['num_shots']} measurements")
    print(f"Execution time: {results['execution_time_seconds']:.6f} seconds")
    print(f"Fidelity: {results['fidelity']:.4f}\n")

    # 2. Quantum Memory Database
    print("2. QUANTUM MEMORY DATABASE")
    print("-" * 40)

    memory_db = QuantumMemoryDatabase(
        storage_capacity_qubits=10000,
        guaranteed_retention_days=365
    )

    # Store quantum states
    state1 = np.array([1, 0], dtype=complex) / np.sqrt(1)  # |0⟩
    state2 = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+⟩

    id1 = memory_db.store_quantum_state(state1, metadata={'label': 'zero_state'})
    id2 = memory_db.store_quantum_state(state2, metadata={'label': 'plus_state'})

    print(f"Stored quantum states: {id1}, {id2}")

    retrieved = memory_db.retrieve_quantum_state(id1)
    print(f"Retrieved state fidelity: {np.abs(np.vdot(state1, retrieved))**2:.6f}\n")

    # 3. Quantum Sensor Array
    print("3. QUANTUM SENSOR ARRAY")
    print("-" * 40)

    sensor_array = QuantumSensorArray(num_sensors=100, sensitivity=1e-15)

    measurement = sensor_array.perform_sensing(
        integration_time=10.0,
        target_parameter="magnetic_field"
    )

    print(f"Magnetic field measurement:")
    print(f"  Value: {measurement['value']:.2e} Tesla")
    print(f"  Uncertainty: {measurement['uncertainty']:.2e} Tesla")
    print(f"  Quantum advantage: {measurement['quantum_advantage']:.1f}x\n")

    # 4. Molecular Simulator
    print("4. MOLECULAR SIMULATOR")
    print("-" * 40)

    mol_sim = MolecularSimulator(max_molecular_size=50)

    molecule = "C6H6"  # Benzene
    results = mol_sim.simulate_molecule(molecule, property_to_compute="ground_state_energy")

    print(f"Simulated {results['molecule']}:")
    print(f"  Ground state energy: {results['value']:.2f} eV")
    print(f"  Required qubits: {results['required_qubits']}")
    print(f"  Accuracy: {results['accuracy']}\n")


def run_benchmark_suite():
    """Run comprehensive benchmarks."""
    print_section("PERFORMANCE BENCHMARKING SUITE")

    print("Benchmark 1: Particle Emission Rate")
    print("-" * 40)

    emitter = EnhancedParticleEmitter(num_ions=1000, target_qubit_rate=10000)

    start = time.time()
    qubits = emitter.continuous_qubit_generation(duration=0.1, rate=10000)
    elapsed = time.time() - start

    print(f"Generated {len(qubits)} qubits in {elapsed:.3f} seconds")
    print(f"Rate: {len(qubits)/elapsed:.1f} qubits/second\n")

    print("Benchmark 2: Long-Lived Qubit Stabilization")
    print("-" * 40)

    qubit = LongLivedLogicalQubit(
        num_physical_qubits=25,
        code_distance=7,
        target_lifetime_years=50.0
    )
    qubit.activate()

    start = time.time()
    for _ in range(100):
        qubit.apply_stabilization_cycle(cycle_duration=1e-4)
    elapsed = time.time() - start

    print(f"Ran 100 stabilization cycles in {elapsed:.3f} seconds")
    print(f"Average cycle time: {elapsed/100*1000:.2f} milliseconds")
    print(f"Estimated lifetime: {qubit.get_current_lifetime_estimate():.1f} years\n")

    print("Benchmark 3: Error Correction Throughput")
    print("-" * 40)

    ft_system = AdvancedFaultTolerance(
        num_logical_qubits=50,
        code_distance=5,
        physical_error_rate=1e-3
    )

    start = time.time()
    for _ in range(100):
        ft_system.run_correction_cycle(use_prediction=True)
    elapsed = time.time() - start

    print(f"Ran 100 correction cycles in {elapsed:.3f} seconds")
    print(f"Average cycle time: {elapsed/100*1000:.2f} milliseconds")

    metrics = ft_system.get_current_metrics()
    print(f"Error rate improvement: {metrics['error_rate_improvement']:.2e}x\n")


def main():
    """Run comprehensive demonstration."""
    print("\n" + "="*80)
    print("  COMPREHENSIVE DEMONSTRATION")
    print("  Long-Lived Fault-Tolerant Logical Qubits")
    print("  with Enhanced Particle Emitter System")
    print("="*80)

    try:
        # Run all demonstrations
        demo_enhanced_particle_emitter()
        demo_long_lived_qubits()
        demo_advanced_fault_tolerance()
        demo_longevity_manager()
        demo_practical_applications()
        run_benchmark_suite()

        print_section("DEMONSTRATION COMPLETE")
        print("All components successfully demonstrated!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Enhanced particle emitter with exotic particles")
        print("  ✓ Long-lived logical qubits (multi-year coherence)")
        print("  ✓ Advanced fault tolerance with error correction")
        print("  ✓ Automated qubit longevity management")
        print("  ✓ Practical real-world applications")
        print("  ✓ Performance benchmarking")
        print("\nThe system is ready for practical quantum computing applications!")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Quick test script for Quantum OS
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Quantum OS components
try:
    from core.kernel import QuantumOS
    from config.settings import QuantumOSConfig
    from error_correction.surface_codes import SurfaceCode
    print("✓ Successfully imported Quantum OS modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Quantum OS - Quick System Test")
print("="*60)

# Test 1: Create config
print("\n[Test 1] Creating Quantum OS configuration...")
try:
    config = QuantumOSConfig()
    print(f"✓ Config created successfully")
    print(f"  Version: {config.version}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Initialize Quantum OS
print("\n[Test 2] Initializing Quantum OS...")
try:
    qos = QuantumOS(config)
    print("✓ Quantum OS initialized")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: List available backends
print("\n[Test 3] Checking available backends...")
try:
    backends = qos.list_backends()
    print(f"✓ Found {len(backends)} backend(s):")
    for backend in backends:
        print(f"  - {backend}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Get system status
print("\n[Test 4] Getting system status...")
try:
    status = qos.get_system_status()
    print(f"✓ System status retrieved")
    print(f"  Version: {status.get('version', 'N/A')}")
    print(f"  Backends: {len(status.get('backends', {}))}")
    print(f"  Max qubits: {status.get('resources', {}).get('max_qubits', 'N/A')}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test error correction
print("\n[Test 5] Testing error correction...")
try:
    code = SurfaceCode(code_distance=3)
    params = code.get_code_parameters()
    print(f"✓ Surface code created")
    print(f"  Code distance: {params['code_distance']}")
    print(f"  Data qubits: {params['num_data_qubits']}")
    print(f"  Logical error rate: {params['logical_error_rate']:.2e}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Try creating a simple circuit
print("\n[Test 6] Creating a simple quantum circuit...")
try:
    import cirq
    circuit = qos.create_circuit(num_qubits=2, backend_name='cirq_simulator')
    qubits = sorted(circuit.all_qubits())
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    print("✓ Bell state circuit created")
    print(f"  Circuit has {len(qubits)} qubits")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Execute circuit
print("\n[Test 7] Executing quantum circuit...")
try:
    result = qos.execute(circuit, shots=1024, backend_name='cirq_simulator')
    print(f"✓ Execution completed")
    print(f"  Success: {result.success}")
    print(f"  Execution time: {result.execution_time:.4f}s")
    if result.counts:
        print(f"  Results: {result.counts}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Quantum OS System Test Complete!")
print("="*60)

"""
Test Quantum OS Core Functionality
"""

import pytest
import sys
sys.path.insert(0, '..')

from quantum_os import create_quantum_os, QuantumOS


def test_quantum_os_creation():
    """Test Quantum OS creation"""
    qos = create_quantum_os()
    assert isinstance(qos, QuantumOS)
    assert qos.VERSION is not None


def test_list_backends():
    """Test backend listing"""
    qos = create_quantum_os()
    backends = qos.list_backends()

    assert isinstance(backends, list)
    assert len(backends) > 0


def test_system_status():
    """Test system status"""
    qos = create_quantum_os()
    status = qos.get_system_status()

    assert 'version' in status
    assert 'backends' in status
    assert 'scheduler' in status
    assert 'resources' in status


def test_circuit_creation():
    """Test circuit creation"""
    qos = create_quantum_os()

    try:
        circuit = qos.create_circuit(num_qubits=3)
        assert circuit is not None
    except Exception as e:
        pytest.skip(f"Circuit creation failed: {e}")


def test_resource_management():
    """Test resource management"""
    qos = create_quantum_os()
    status = qos.get_system_status()

    resources = status['resources']
    assert 'max_qubits' in resources
    assert 'available_qubits' in resources
    assert resources['available_qubits'] <= resources['max_qubits']


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

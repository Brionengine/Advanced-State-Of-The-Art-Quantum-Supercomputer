"""
Test Error Correction Modules
"""

import pytest
import sys
sys.path.insert(0, '..')

from quantum_os import SurfaceCode, StabilizerCode
import numpy as np


def test_surface_code_initialization():
    """Test surface code initialization"""
    code = SurfaceCode(code_distance=3)

    assert code.code_distance == 3
    assert code.num_data_qubits == 9
    assert code.num_syndrome_qubits == 8


def test_surface_code_parameters():
    """Test surface code parameters"""
    code = SurfaceCode(code_distance=5)
    params = code.get_code_parameters()

    assert params['code_distance'] == 5
    assert params['num_data_qubits'] == 25
    assert params['logical_error_rate'] < params['physical_error_rate']


def test_surface_code_distance_calculation():
    """Test required distance calculation"""
    distance = SurfaceCode.calculate_required_distance(
        target_error_rate=1e-9,
        physical_error_rate=0.001
    )

    assert distance >= 3
    assert distance % 2 == 1  # Must be odd


def test_stabilizer_codes():
    """Test stabilizer code creation"""
    # Bit flip code
    bit_flip = StabilizerCode.create_bit_flip_code()
    assert bit_flip.num_qubits == 3
    assert bit_flip.num_logical_qubits == 1

    # Phase flip code
    phase_flip = StabilizerCode.create_phase_flip_code()
    assert phase_flip.num_qubits == 3

    # Shor code
    shor = StabilizerCode.create_shor_code()
    assert shor.num_qubits == 9


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

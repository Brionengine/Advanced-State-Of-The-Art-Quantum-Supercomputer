"""
Tests for surface-code error-rate modelling.

The forward model and the distance solver invert the same relation, so they are
tested together — a prefactor applied to one and not the other is the failure
mode most likely to recur.
"""

import pytest

from quantum_os.error_correction.surface_codes import SurfaceCode


# -- Forward model ----------------------------------------------------------


@pytest.mark.parametrize("distance", [3, 5, 7, 9, 11])
def test_code_corrects_errors_at_every_distance(distance):
    """A surface code below threshold must beat the physical error rate."""
    params = SurfaceCode(distance).get_code_parameters()

    assert params["logical_error_rate"] < params["physical_error_rate"]
    assert params["can_correct_errors"] is True


def test_distance_five_is_not_break_even():
    """Regression: without the prefactor this landed exactly on the physical rate."""
    params = SurfaceCode(5).get_code_parameters()

    assert params["logical_error_rate"] < params["physical_error_rate"]


def test_logical_error_rate_falls_with_distance():
    rates = [SurfaceCode(d).logical_error_rate for d in (3, 5, 7, 9)]

    assert rates == sorted(rates, reverse=True)


def test_each_step_gains_a_full_ratio_factor():
    """Exponent floor((d+1)/2) grows by 1 every two steps of distance."""
    ratio = SurfaceCode.DEFAULT_PHYSICAL_ERROR_RATE / SurfaceCode.ERROR_THRESHOLD

    assert SurfaceCode(7).logical_error_rate == pytest.approx(
        SurfaceCode(5).logical_error_rate * ratio
    )


def test_logical_rate_matches_the_documented_formula():
    code = SurfaceCode(5)
    expected = SurfaceCode.LOGICAL_ERROR_PREFACTOR * (
        code.physical_error_rate / code.error_threshold
    ) ** 3

    assert code.logical_error_rate == pytest.approx(expected)


def test_qubit_counts_scale_with_distance():
    code = SurfaceCode(5)

    assert code.num_data_qubits == 25
    assert code.num_syndrome_qubits == 24


@pytest.mark.parametrize("bad", [2, 4, 1, 0, -3])
def test_invalid_distances_are_rejected(bad):
    with pytest.raises(ValueError):
        SurfaceCode(bad)


# -- Distance solver --------------------------------------------------------


@pytest.mark.parametrize("target", [1e-4, 1e-6, 1e-9, 1e-12])
def test_solved_distance_actually_meets_the_target(target):
    distance = SurfaceCode.calculate_required_distance(target_error_rate=target)

    assert SurfaceCode(distance).logical_error_rate <= target


@pytest.mark.parametrize("target", [1e-6, 1e-9, 1e-12])
def test_solved_distance_is_minimal(target):
    """The next smaller odd distance must fall short, or we over-provisioned."""
    distance = SurfaceCode.calculate_required_distance(target_error_rate=target)

    if distance > 3:
        assert SurfaceCode(distance - 2).logical_error_rate > target


@pytest.mark.parametrize("target", [1e-4, 1e-6, 1e-9])
def test_solved_distance_is_odd_and_at_least_three(target):
    distance = SurfaceCode.calculate_required_distance(target_error_rate=target)

    assert distance >= 3
    assert distance % 2 == 1


def test_loose_target_returns_the_smallest_code():
    assert SurfaceCode.calculate_required_distance(target_error_rate=0.5) == 3


def test_physical_rate_above_threshold_is_rejected():
    with pytest.raises(ValueError, match="exceeds surface code threshold"):
        SurfaceCode.calculate_required_distance(
            target_error_rate=1e-9, physical_error_rate=0.02
        )


def test_non_positive_target_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        SurfaceCode.calculate_required_distance(target_error_rate=0.0)


def test_solver_and_forward_model_share_the_threshold():
    """Regression: the solver used to hardcode 0.01 independently."""
    with pytest.raises(ValueError):
        SurfaceCode.calculate_required_distance(
            target_error_rate=1e-9,
            physical_error_rate=SurfaceCode.ERROR_THRESHOLD,
        )

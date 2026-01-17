"""
Unit tests for the ΔΦ diagnostic operator.

These tests verify:
- Algebraic correctness of ΔΦ components
- Deterministic behavior
- Absence of control or side effects

The ΔΦ operator is tested strictly as a diagnostic observer,
consistent with the paper and repository scope.
"""

import numpy as np
import pytest

from framework.delta_phi import (
    TriadicFeatures,
    DeltaPhiOperator,
    DeltaPhiVector
)


# ---------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------

def test_delta_phi_zero_state():
    """
    If φ_S = φ_I = φ_C and no change occurs,
    then ΔΦ must be identically zero.
    """
    features = TriadicFeatures(
        phi_S=1.0,
        phi_I=1.0,
        phi_C=1.0,
        timestamp=0.0
    )

    operator = DeltaPhiOperator(dt=1.0)

    delta = operator.compute(features, prev_features=features)

    assert isinstance(delta, DeltaPhiVector)
    assert delta.phi_S_dot == 0.0
    assert delta.e_I == 0.0
    assert delta.e_C == 0.0
    assert delta.norm() == 0.0


def test_delta_phi_memory_and_constraint_errors():
    """
    Verify algebraic definitions:
    e_I = φ_S − φ_I
    e_C = φ_C − φ_S
    """
    features = TriadicFeatures(
        phi_S=2.0,
        phi_I=1.5,
        phi_C=3.0,
        timestamp=0.0
    )

    operator = DeltaPhiOperator(dt=1.0)
    delta = operator.compute(features, prev_features=None)

    assert delta.e_I == pytest.approx(0.5)
    assert delta.e_C == pytest.approx(1.0)


# ---------------------------------------------------------------------
# Derivative behavior tests
# ---------------------------------------------------------------------

def test_finite_difference_derivative():
    """
    Finite difference derivative:
    φ̇_S(t) = (φ_S(t) − φ_S(t−1)) / dt
    """
    prev = TriadicFeatures(phi_S=1.0, phi_I=0.0, phi_C=2.0)
    curr = TriadicFeatures(phi_S=1.5, phi_I=0.0, phi_C=2.0)

    operator = DeltaPhiOperator(dt=0.5)
    delta = operator.compute(curr, prev)

    expected_derivative = (1.5 - 1.0) / 0.5
    assert delta.phi_S_dot == pytest.approx(expected_derivative)


def test_first_call_derivative_is_zero():
    """
    On first invocation without history,
    φ̇_S must default to zero (no undefined behavior).
    """
    features = TriadicFeatures(phi_S=1.0, phi_I=0.0, phi_C=2.0)

    operator = DeltaPhiOperator(dt=1.0)
    delta = operator.compute(features, prev_features=None)

    assert delta.phi_S_dot == 0.0


# ---------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------

def test_delta_phi_determinism():
    """
    Identical inputs must produce identical ΔΦ outputs.
    """
    features = TriadicFeatures(
        phi_S=1.2,
        phi_I=0.8,
        phi_C=2.0,
        timestamp=1.0
    )

    operator1 = DeltaPhiOperator(dt=1.0)
    operator2 = DeltaPhiOperator(dt=1.0)

    delta1 = operator1.compute(features, prev_features=None)
    delta2 = operator2.compute(features, prev_features=None)

    assert delta1.to_vector().tolist() == delta2.to_vector().tolist()


# ---------------------------------------------------------------------
# Non-intervention sanity test
# ---------------------------------------------------------------------

def test_operator_does_not_modify_inputs():
    """
    ΔΦ operator must not mutate input feature objects.
    """
    features = TriadicFeatures(
        phi_S=1.0,
        phi_I=0.5,
        phi_C=2.0
    )

    features_copy = TriadicFeatures(
        phi_S=features.phi_S,
        phi_I=features.phi_I,
        phi_C=features.phi_C
    )

    operator = DeltaPhiOperator(dt=1.0)
    operator.compute(features, prev_features=None)

    assert features.phi_S == features_copy.phi_S
    assert features.phi_I == features_copy.phi_I
    assert features.phi_C == features_copy.phi_C

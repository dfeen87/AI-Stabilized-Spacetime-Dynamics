"""
Unit tests for stability scoring J(t) and regime classification.

These tests verify:
- Correct computation of weighted stability score J(t)
- Proper regime classification across thresholds
- Deterministic behavior
- Absence of control or intervention logic

The stability monitor is tested strictly as a diagnostic evaluator,
consistent with the ΔΦ framework definition.
"""

import numpy as np
import pytest

from framework.delta_phi import DeltaPhiVector
from framework.stability import (
    StabilityMonitor,
    StabilityWeights,
    RegimeThresholds,
    RegimeState
)


# ---------------------------------------------------------------------
# Basic score correctness
# ---------------------------------------------------------------------

def test_stability_score_components():
    """
    Verify correct decomposition:
    J = w_S·φ̇_S² + w_I·e_I² + w_C·e_C²
    """
    delta_phi = DeltaPhiVector(
        phi_S_dot=2.0,
        e_I=1.0,
        e_C=0.5,
        timestamp=0.0
    )

    weights = StabilityWeights(w_S=1.0, w_I=2.0, w_C=4.0)
    monitor = StabilityMonitor(weights=weights)

    score = monitor.compute_score(delta_phi)

    expected_J_S = 1.0 * (2.0 ** 2)
    expected_J_I = 2.0 * (1.0 ** 2)
    expected_J_C = 4.0 * (0.5 ** 2)

    assert score.J_S == pytest.approx(expected_J_S)
    assert score.J_I == pytest.approx(expected_J_I)
    assert score.J_C == pytest.approx(expected_J_C)
    assert score.J == pytest.approx(expected_J_S + expected_J_I + expected_J_C)


# ---------------------------------------------------------------------
# Regime classification tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "J_value, expected_regime",
    [
        (0.001, RegimeState.STABLE),
        (0.05,  RegimeState.WARNING),
        (0.30,  RegimeState.UNSTABLE),
        (2.00,  RegimeState.CRITICAL),
    ]
)
def test_regime_classification(J_value, expected_regime):
    """
    Regime classification based on sqrt(J) thresholds.
    """
    thresholds = RegimeThresholds(
        epsilon_stable=0.1,
        epsilon_warning=0.5,
        epsilon_critical=1.0
    )

    monitor = StabilityMonitor(thresholds=thresholds)

    # Construct ΔΦ with desired J = φ̇_S²
    delta_phi = DeltaPhiVector(
        phi_S_dot=np.sqrt(J_value),
        e_I=0.0,
        e_C=0.0,
        timestamp=0.0
    )

    score = monitor.compute_score(delta_phi)
    assert score.regime == expected_regime


# ---------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------

def test_stability_score_determinism():
    """
    Identical ΔΦ inputs must produce identical J(t) and regime.
    """
    delta_phi = DeltaPhiVector(
        phi_S_dot=0.3,
        e_I=0.2,
        e_C=0.1,
        timestamp=1.0
    )

    weights = StabilityWeights(w_S=1.0, w_I=1.0, w_C=1.0)
    thresholds = RegimeThresholds(
        epsilon_stable=0.2,
        epsilon_warning=0.5,
        epsilon_critical=1.0
    )

    monitor1 = StabilityMonitor(weights=weights, thresholds=thresholds)
    monitor2 = StabilityMonitor(weights=weights, thresholds=thresholds)

    score1 = monitor1.compute_score(delta_phi)
    score2 = monitor2.compute_score(delta_phi)

    assert score1.J == pytest.approx(score2.J)
    assert score1.regime == score2.regime


# ---------------------------------------------------------------------
# Coherent stability test
# ---------------------------------------------------------------------

def test_coherent_stability_detection():
    """
    Coherent stability requires all recent J(t) < epsilon_stable².
    """
    thresholds = RegimeThresholds(
        epsilon_stable=0.2,
        epsilon_warning=0.5,
        epsilon_critical=1.0
    )

    monitor = StabilityMonitor(thresholds=thresholds)

    # Generate small ΔΦ values
    for t in range(10):
        delta_phi = DeltaPhiVector(
            phi_S_dot=0.05,
            e_I=0.02,
            e_C=0.01,
            timestamp=t
        )
        monitor.compute_score(delta_phi)

    assert monitor.is_coherently_stable(horizon=5) is True


def test_not_coherently_stable():
    """
    A single large instability breaks coherent stability.
    """
    thresholds = RegimeThresholds(
        epsilon_stable=0.2,
        epsilon_warning=0.5,
        epsilon_critical=1.0
    )

    monitor = StabilityMonitor(thresholds=thresholds)

    # Mostly stable
    for t in range(4):
        monitor.compute_score(
            DeltaPhiVector(phi_S_dot=0.05, e_I=0.01, e_C=0.01, timestamp=t)
        )

    # One instability
    monitor.compute_score(
        DeltaPhiVector(phi_S_dot=1.0, e_I=0.0, e_C=0.0, timestamp=5)
    )

    assert monitor.is_coherently_stable(horizon=5) is False


# ---------------------------------------------------------------------
# Transition detection tests
# ---------------------------------------------------------------------

def test_transition_detection_by_component():
    """
    Transition detection identifies dominant ΔΦ component.
    """
    monitor = StabilityMonitor()

    # Stable region
    for t in range(5):
        monitor.compute_score(
            DeltaPhiVector(phi_S_dot=0.01, e_I=0.01, e_C=0.01, timestamp=t)
        )

    # Constraint-dominated transition
    delta_phi = DeltaPhiVector(
        phi_S_dot=0.02,
        e_I=0.02,
        e_C=1.0,
        timestamp=6
    )
    monitor.compute_score(delta_phi)

    is_transition, transition_type = monitor.detect_transition(lookback=5)

    assert is_transition is True
    assert transition_type == "constraint_approach"


# ---------------------------------------------------------------------
# History management sanity check
# ---------------------------------------------------------------------

def test_history_length_respected():
    """
    StabilityMonitor must respect history_length cap.
    """
    monitor = StabilityMonitor(history_length=5)

    for t in range(10):
        monitor.compute_score(
            DeltaPhiVector(phi_S_dot=0.1, e_I=0.1, e_C=0.1, timestamp=t)
        )

    assert len(monitor.score_history) == 5
    assert len(monitor.delta_phi_history) == 5

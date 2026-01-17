"""
Reproducibility & Determinism Tests

These tests enforce that the ΔΦ diagnostic pipeline is:
- Deterministic given fixed configuration and fixed inputs
- Replayable (same input sequence -> identical outputs)
- Consistent across processing modes (stepwise vs batch)
- Configurable in an auditable way (save/load + hash consistency)

This suite intentionally avoids stochastic feature maps or hidden randomness.
"""

import json
import numpy as np
import pytest

from pipeline.observer import DiagnosticObserver, ObserverConfig
from pipeline.config import ObserverConfiguration, create_default_config


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_deterministic_signal(n: int = 200) -> np.ndarray:
    """
    Create a deterministic, nontrivial signal sequence.
    Avoids randomness to ensure tests are stable and replayable.
    """
    t = np.linspace(0.0, 4.0 * np.pi, n)
    y = np.sin(t) + 0.25 * np.sin(3.0 * t)  # deterministic multi-frequency
    return y


def _extract_output_signature(outputs):
    """
    Convert ObserverOutput list into a deterministic signature structure.

    We only include numerically relevant fields:
    - phi_S, phi_I, phi_C
    - ΔΦ components (phi_S_dot, e_I, e_C)
    - J components and regime label
    """
    sig = []
    for o in outputs:
        sig.append({
            "t": float(o.timestamp),
            "phi_S": float(o.phi_S),
            "phi_I": float(o.phi_I),
            "phi_C": float(o.phi_C),
            "dphi": {
                "phi_S_dot": float(o.delta_phi.phi_S_dot),
                "e_I": float(o.delta_phi.e_I),
                "e_C": float(o.delta_phi.e_C),
            },
            "score": {
                "J": float(o.score.J),
                "J_S": float(o.score.J_S),
                "J_I": float(o.score.J_I),
                "J_C": float(o.score.J_C),
                "regime": str(o.score.regime.value),
            }
        })
    return sig


def _assert_signatures_equal(sig_a, sig_b, atol=1e-12, rtol=1e-12):
    """
    Deep compare two signatures with strict numeric tolerances.
    """
    assert len(sig_a) == len(sig_b)

    for i, (a, b) in enumerate(zip(sig_a, sig_b)):
        assert a["t"] == pytest.approx(b["t"], abs=atol)

        assert a["phi_S"] == pytest.approx(b["phi_S"], abs=atol, rel=rtol)
        assert a["phi_I"] == pytest.approx(b["phi_I"], abs=atol, rel=rtol)
        assert a["phi_C"] == pytest.approx(b["phi_C"], abs=atol, rel=rtol)

        assert a["dphi"]["phi_S_dot"] == pytest.approx(b["dphi"]["phi_S_dot"], abs=atol, rel=rtol)
        assert a["dphi"]["e_I"] == pytest.approx(b["dphi"]["e_I"], abs=atol, rel=rtol)
        assert a["dphi"]["e_C"] == pytest.approx(b["dphi"]["e_C"], abs=atol, rel=rtol)

        assert a["score"]["J"] == pytest.approx(b["score"]["J"], abs=atol, rel=rtol)
        assert a["score"]["J_S"] == pytest.approx(b["score"]["J_S"], abs=atol, rel=rtol)
        assert a["score"]["J_I"] == pytest.approx(b["score"]["J_I"], abs=atol, rel=rtol)
        assert a["score"]["J_C"] == pytest.approx(b["score"]["J_C"], abs=atol, rel=rtol)

        # Regime should match exactly
        assert a["score"]["regime"] == b["score"]["regime"], f"Mismatch at index {i}"


# ---------------------------------------------------------------------
# Core determinism: same config + same inputs -> identical outputs
# ---------------------------------------------------------------------

def test_observer_replay_determinism_same_inputs_same_outputs():
    """
    Running the observer twice with the same configuration and the same
    input sequence must produce identical outputs.
    """
    y = _make_deterministic_signal(250)
    t = np.arange(len(y)) * 0.1

    config = ObserverConfig(
        # Feature extraction: deterministic
        feature_map="identity",
        feature_params={},

        # Memory: deterministic EWMA
        memory_model="ewma",
        memory_params={"alpha": 0.2, "initial_value": 0.0},

        # Constraint: percentile, deterministic given history
        constraint_model="percentile",
        constraint_params={"percentile": 95.0, "min_history": 10, "default_value": 1.0},

        # ΔΦ
        dt=0.1,
        derivative_method="finite_difference",

        # Stability
        w_S=1.0,
        w_I=0.5,
        w_C=2.0,
        epsilon_stable=0.1,
        epsilon_warning=0.5,
        epsilon_critical=1.0,

        history_length=300,
    )

    obs1 = DiagnosticObserver(config)
    out1 = obs1.process_batch(y, timestamps=t)
    sig1 = _extract_output_signature(out1)

    obs2 = DiagnosticObserver(config)
    out2 = obs2.process_batch(y, timestamps=t)
    sig2 = _extract_output_signature(out2)

    _assert_signatures_equal(sig1, sig2)


# ---------------------------------------------------------------------
# Processing mode equivalence: batch vs stepwise
# ---------------------------------------------------------------------

def test_batch_vs_stepwise_equivalence():
    """
    Processing the same sequence via process_batch vs repeated process()
    must yield identical outputs.
    """
    y = _make_deterministic_signal(120)
    t = np.arange(len(y)) * 0.2

    config = ObserverConfig(
        feature_map="identity",
        memory_model="ewma",
        memory_params={"alpha": 0.3, "initial_value": 0.0},
        constraint_model="percentile",
        constraint_params={"percentile": 90.0, "min_history": 5, "default_value": 1.0},
        dt=0.2,
        derivative_method="finite_difference",
        w_S=1.0,
        w_I=1.0,
        w_C=1.0,
        epsilon_stable=0.2,
        epsilon_warning=0.6,
        epsilon_critical=1.2,
        history_length=200,
    )

    # Batch
    obs_batch = DiagnosticObserver(config)
    out_batch = obs_batch.process_batch(y, timestamps=t)
    sig_batch = _extract_output_signature(out_batch)

    # Stepwise
    obs_step = DiagnosticObserver(config)
    out_step = []
    for i in range(len(y)):
        out_step.append(obs_step.process(y[i], timestamp=t[i]))
    sig_step = _extract_output_signature(out_step)

    _assert_signatures_equal(sig_batch, sig_step)


# ---------------------------------------------------------------------
# Reset determinism: same observer after reset behaves identically
# ---------------------------------------------------------------------

def test_reset_reproducibility():
    """
    After observer.reset(), running the same inputs again must reproduce
    the same outputs exactly.
    """
    y = _make_deterministic_signal(180)
    t = np.arange(len(y)) * 0.05

    config = ObserverConfig(
        feature_map="identity",
        memory_model="ewma",
        memory_params={"alpha": 0.15, "initial_value": 0.0},
        constraint_model="rolling_max",
        constraint_params={"window": 30, "margin": 0.1, "default_value": 1.0},
        dt=0.05,
        derivative_method="finite_difference",
        w_S=1.0,
        w_I=0.7,
        w_C=1.8,
        epsilon_stable=0.15,
        epsilon_warning=0.5,
        epsilon_critical=1.0,
        history_length=300,
    )

    obs = DiagnosticObserver(config)

    out1 = obs.process_batch(y, timestamps=t)
    sig1 = _extract_output_signature(out1)

    obs.reset()

    out2 = obs.process_batch(y, timestamps=t)
    sig2 = _extract_output_signature(out2)

    _assert_signatures_equal(sig1, sig2)


# ---------------------------------------------------------------------
# Config reproducibility: save/load + hash consistency
# ---------------------------------------------------------------------

def test_configuration_save_load_hash_stability(tmp_path):
    """
    Saving and re-loading an ObserverConfiguration must preserve the config hash.
    This is a reproducibility primitive for audit trails.
    """
    config = create_default_config()

    # Validate before proceeding
    is_valid, errors = config.validate()
    assert is_valid, f"Default config invalid: {errors}"

    # Lock it (freezes hash)
    config.lock()
    original_hash = config.compute_hash()

    # Save JSON
    json_path = tmp_path / "config.json"
    config.save(str(json_path), format="json")

    # Load JSON
    loaded = ObserverConfiguration.load(str(json_path))
    loaded_hash = loaded.compute_hash()

    assert original_hash == loaded_hash

    # Ensure canonical content equivalence (excluding runtime-only fields)
    # We compare dicts after removing created_at + locked flags, which can differ.
    d1 = config.to_dict()
    d2 = loaded.to_dict()

    for k in ["_created_at", "_locked", "_config_hash"]:
        d1.pop(k, None)
        d2.pop(k, None)

    assert d1 == d2


def test_locked_configuration_prevents_modification():
    """
    Locked configurations must reject modifications to preserve reproducibility.
    """
    config = create_default_config()
    config.lock()

    with pytest.raises(RuntimeError):
        config.stability.w_S = 999.0  # should be blocked

"""
Reproducibility & Deterministic Replay Tests

This suite enforces that:
- DiagnosticObserver is deterministic for fixed inputs + fixed config
- Reset reproduces identical outputs
- ReplayRecording hashes are stable and verifiable
- Recordings survive save/load without mutation (JSON + Pickle)
- ReplayValidator compares recordings frame-by-frame within tolerance
"""

import numpy as np
import pytest

from pipeline.observer import DiagnosticObserver, ObserverConfig
from pipeline.config import ObserverConfiguration, create_default_config
from pipeline.replay import ReplayRecording, ReplayFrame, ReplayValidator


# ---------------------------------------------------------------------
# Deterministic signal helper (NO randomness)
# ---------------------------------------------------------------------

def _make_signal(n: int = 250) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return np.sin(t) + 0.25 * np.sin(3.0 * t)


def _make_observer_config() -> ObserverConfig:
    """
    Use only deterministic components:
    - feature_map: identity
    - memory: ewma
    - constraint: percentile (deterministic given history)
    """
    return ObserverConfig(
        feature_map="identity",
        feature_params={},

        memory_model="ewma",
        memory_params={"alpha": 0.2, "initial_value": 0.0},

        constraint_model="percentile",
        constraint_params={"percentile": 95.0, "min_history": 10, "default_value": 1.0},

        dt=0.1,
        derivative_method="finite_difference",

        w_S=1.0,
        w_I=0.5,
        w_C=2.0,

        epsilon_stable=0.1,
        epsilon_warning=0.5,
        epsilon_critical=1.0,

        history_length=400,
    )


def _record_session(observer: DiagnosticObserver, y: np.ndarray, timestamps: np.ndarray) -> ReplayRecording:
    """
    Run observer on signal and build a ReplayRecording from ObserverOutput frames.
    """
    rec = ReplayRecording(
        config_dict=observer.get_config(),
        metadata={"test": "reproducibility"}
    )

    for step, (val, t) in enumerate(zip(y, timestamps)):
        out = observer.process(val, timestamp=float(t))
        rec.add_frame(ReplayFrame.from_observer_output(out, step=step))

    rec.finalize()
    return rec


# ---------------------------------------------------------------------
# Core determinism tests
# ---------------------------------------------------------------------

def test_observer_is_deterministic_across_fresh_instances():
    """
    Two fresh observers with identical config + identical inputs must yield
    identical recordings (frame-by-frame) and identical recording hashes.
    """
    y = _make_signal(300)
    t = np.arange(len(y)) * 0.1

    cfg = _make_observer_config()

    obs1 = DiagnosticObserver(cfg)
    rec1 = _record_session(obs1, y, t)

    obs2 = DiagnosticObserver(cfg)
    rec2 = _record_session(obs2, y, t)

    # Hash match is the strongest signal, but still compare frames too
    assert rec1.recording_hash == rec2.recording_hash

    comparison = ReplayValidator.compare_recordings(rec1, rec2, tolerance=1e-12)
    assert comparison["identical"], f"Differences: {comparison['differences'][:5]}"


def test_observer_reset_reproduces_identical_recording():
    """
    Same observer: run -> reset -> run again must match exactly.
    """
    y = _make_signal(220)
    t = np.arange(len(y)) * 0.1

    cfg = _make_observer_config()
    obs = DiagnosticObserver(cfg)

    rec1 = _record_session(obs, y, t)
    obs.reset()
    rec2 = _record_session(obs, y, t)

    assert rec1.recording_hash == rec2.recording_hash

    comparison = ReplayValidator.compare_recordings(rec1, rec2, tolerance=1e-12)
    assert comparison["identical"], f"Differences: {comparison['differences'][:5]}"


# ---------------------------------------------------------------------
# ReplayRecording integrity: verify_hash + save/load round-trip
# ---------------------------------------------------------------------

def test_recording_verify_hash_passes():
    """
    After finalize(), verify_hash() must succeed.
    """
    y = _make_signal(150)
    t = np.arange(len(y)) * 0.1

    cfg = _make_observer_config()
    obs = DiagnosticObserver(cfg)
    rec = _record_session(obs, y, t)

    assert rec.verify_hash() is True


def test_recording_save_load_json_roundtrip(tmp_path):
    """
    JSON save/load must preserve:
    - frames count
    - content (within strict tolerance)
    - recording hash
    - verify_hash() should pass after load
    """
    y = _make_signal(120)
    t = np.arange(len(y)) * 0.1

    cfg = _make_observer_config()
    obs = DiagnosticObserver(cfg)
    rec = _record_session(obs, y, t)

    path = tmp_path / "recording.json"
    rec.save(str(path), format="json")

    loaded = ReplayRecording.load(str(path))

    # Hash should match (the file should not mutate content)
    assert loaded.recording_hash == rec.recording_hash
    assert len(loaded.frames) == len(rec.frames)

    assert loaded.verify_hash() is True

    comparison = ReplayValidator.compare_recordings(rec, loaded, tolerance=1e-12)
    assert comparison["identical"], f"Differences: {comparison['differences'][:5]}"


def test_recording_save_load_pickle_roundtrip(tmp_path):
    """
    Pickle save/load must preserve content and hash.
    """
    y = _make_signal(120)
    t = np.arange(len(y)) * 0.1

    cfg = _make_observer_config()
    obs = DiagnosticObserver(cfg)
    rec = _record_session(obs, y, t)

    path = tmp_path / "recording.pkl"
    rec.save(str(path), format="pickle", compress=False)

    loaded = ReplayRecording.load(str(path))

    assert loaded.recording_hash == rec.recording_hash
    assert len(loaded.frames) == len(rec.frames)
    assert loaded.verify_hash() is True

    comparison = ReplayValidator.compare_recordings(rec, loaded, tolerance=1e-12)
    assert comparison["identical"], f"Differences: {comparison['differences'][:5]}"


# ---------------------------------------------------------------------
# Config auditability: save/load hash stability
# ---------------------------------------------------------------------

def test_configuration_save_load_hash_stability(tmp_path):
    """
    Saving and re-loading ObserverConfiguration must preserve compute_hash().
    """
    cfg = create_default_config()
    is_valid, errors = cfg.validate()
    assert is_valid, f"Default config invalid: {errors}"

    cfg.lock()
    h1 = cfg.compute_hash()

    json_path = tmp_path / "config.json"
    cfg.save(str(json_path), format="json")

    loaded = ObserverConfiguration.load(str(json_path))
    h2 = loaded.compute_hash()

    assert h1 == h2


def test_locked_configuration_prevents_modification():
    cfg = create_default_config()
    cfg.lock()
    with pytest.raises(RuntimeError):
        cfg.stability.w_S = 999.0


# ---------------------------------------------------------------------
# High-level determinism validator
# ---------------------------------------------------------------------

def test_replayvalidator_validate_determinism_reports_true():
    """
    ReplayValidator.validate_determinism() should report deterministic=True
    for deterministic config + deterministic signal.
    """
    y = _make_signal(120)

    cfg = _make_observer_config()
    obs = DiagnosticObserver(cfg)

    result = ReplayValidator.validate_determinism(obs, y_signal=y, num_runs=3)

    assert result["deterministic"] is True
    assert result["all_hashes_match"] is True
    assert len(result["hashes"]) == 3
    assert len(set(result["hashes"])) == 1

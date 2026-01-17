"""
Feature Maps: φ_S(t) - Instantaneous State Features

This module provides various feature extraction methods for φ_S(t).
All implementations are deterministic and reproducible when parameters are fixed.

Feature maps extract structural characteristics from signals y(t), such as:
- Synchrony / coherence measures
- Order parameters
- Phase estimates
- Curvature proxies
"""

import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod


class FeatureMap(ABC):
    """
    Abstract base class for feature extraction: y(t) → φ_S(t)
    
    All feature maps must be:
    - Deterministic (same input → same output)
    - Explicitly parameterized (no hidden state)
    - Reproducible (parameters fully specified)
    """
    
    @abstractmethod
    def extract(self, signal: np.ndarray) -> float:
        """
        Extract feature φ_S from signal y(t)
        
        Args:
            signal: Input signal y(t) (may be scalar or vector)
        
        Returns:
            Scalar feature value φ_S(t)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Return all parameters for reproducibility"""
        pass
    
    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class IdentityFeature(FeatureMap):
    """Simplest feature: φ_S(t) = y(t) (for scalar signals)"""
    
    def extract(self, signal: np.ndarray) -> float:
        if np.isscalar(signal):
            return float(signal)
        return float(signal[0]) if len(signal) > 0 else 0.0
    
    def get_params(self) -> dict:
        return {}


class NormFeature(FeatureMap):
    """
    L2 norm feature: φ_S(t) = ‖y(t)‖
    
    Useful for vector-valued signals where magnitude is meaningful.
    """
    
    def __init__(self, ord: Optional[int] = 2):
        """
        Args:
            ord: Order of norm (1, 2, np.inf, etc.)
        """
        self.ord = ord
    
    def extract(self, signal: np.ndarray) -> float:
        if np.isscalar(signal):
            return abs(float(signal))
        return float(np.linalg.norm(signal, ord=self.ord))
    
    def get_params(self) -> dict:
        return {'ord': self.ord}


class SynchronyFeature(FeatureMap):
    """
    Synchrony measure for multi-dimensional signals
    
    φ_S(t) = 1 - std(y(t)) / (mean(|y(t)|) + ε)
    
    High synchrony → φ_S ≈ 1
    Low synchrony → φ_S ≈ 0
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.epsilon = epsilon
    
    def extract(self, signal: np.ndarray) -> float:
        if np.isscalar(signal) or len(signal) == 1:
            return 1.0  # Single value is perfectly synchronous
        
        mean_abs = np.mean(np.abs(signal))
        std_val = np.std(signal)
        
        synchrony = 1.0 - std_val / (mean_abs + self.epsilon)
        return float(np.clip(synchrony, 0.0, 1.0))
    
    def get_params(self) -> dict:
        return {'epsilon': self.epsilon}


class PhaseCoherenceFeature(FeatureMap):
    """
    Phase coherence measure using Hilbert transform
    
    φ_S(t) = |⟨exp(iθ_j(t))⟩| where θ_j are instantaneous phases
    
    Perfect coherence → φ_S = 1
    Random phases → φ_S ≈ 0
    """
    
    def extract(self, signal: np.ndarray) -> float:
        if np.isscalar(signal) or len(signal) == 1:
            return 1.0
        
        # Compute instantaneous phases via Hilbert transform
        from scipy.signal import hilbert
        analytic_signal = hilbert(signal)
        phases = np.angle(analytic_signal)
        
        # Phase coherence order parameter
        mean_phase_vector = np.mean(np.exp(1j * phases))
        coherence = np.abs(mean_phase_vector)
        
        return float(coherence)
    
    def get_params(self) -> dict:
        return {}


class MovingAverageFeature(FeatureMap):
    """
    Moving average feature (smoothed signal value)
    
    φ_S(t) = (1/window) Σ y(t-k) for k in [0, window-1]
    """
    
    def __init__(self, window: int = 5):
        """
        Args:
            window: Window size for moving average
        """
        self.window = window
        self.buffer = []
    
    def extract(self, signal: np.ndarray) -> float:
        value = float(signal) if np.isscalar(signal) else float(signal[0])
        
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        return float(np.mean(self.buffer))
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def get_params(self) -> dict:
        return {'window': self.window}


class VarianceFeature(FeatureMap):
    """
    Variance-based feature (local volatility measure)
    
    φ_S(t) = var(y[t-window:t])
    """
    
    def __init__(self, window: int = 10):
        """
        Args:
            window: Window size for variance computation
        """
        self.window = window
        self.buffer = []
    
    def extract(self, signal: np.ndarray) -> float:
        value = float(signal) if np.isscalar(signal) else float(signal[0])
        
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        if len(self.buffer) < 2:
            return 0.0
        
        return float(np.var(self.buffer))
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def get_params(self) -> dict:
        return {'window': self.window}


class CustomFeature(FeatureMap):
    """
    Custom feature map defined by user function
    
    Allows arbitrary feature extraction while maintaining interface
    """
    
    def __init__(self, func: Callable[[np.ndarray], float], 
                 name: str = "custom",
                 params: Optional[dict] = None):
        """
        Args:
            func: Function mapping signal → feature value
            name: Descriptive name
            params: Dictionary of parameters (for reproducibility)
        """
        self.func = func
        self.name = name
        self.params = params or {}
    
    def extract(self, signal: np.ndarray) -> float:
        return float(self.func(signal))
    
    def get_params(self) -> dict:
        return {'name': self.name, **self.params}


# Registry of available feature maps
FEATURE_MAP_REGISTRY = {
    'identity': IdentityFeature,
    'norm': NormFeature,
    'synchrony': SynchronyFeature,
    'phase_coherence': PhaseCoherenceFeature,
    'moving_average': MovingAverageFeature,
    'variance': VarianceFeature,
}


def create_feature_map(name: str, **kwargs) -> FeatureMap:
    """
    Factory function to create feature map by name
    
    Args:
        name: Name of feature map (from registry)
        **kwargs: Parameters for feature map constructor
    
    Returns:
        Instantiated FeatureMap
    """
    if name not in FEATURE_MAP_REGISTRY:
        raise ValueError(
            f"Unknown feature map '{name}'. "
            f"Available: {list(FEATURE_MAP_REGISTRY.keys())}"
        )
    
    return FEATURE_MAP_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Feature Map Examples")
    print("=" * 50)
    
    # Generate synthetic signal
    t = np.linspace(0, 2*np.pi, 100)
    y_scalar = np.sin(t)
    y_vector = np.column_stack([np.sin(t), np.cos(t), 0.5*np.sin(2*t)])
    
    # Test various feature maps
    features = [
        ('identity', IdentityFeature()),
        ('norm', NormFeature()),
        ('synchrony', SynchronyFeature()),
        ('moving_average', MovingAverageFeature(window=3)),
        ('variance', VarianceFeature(window=5)),
    ]
    
    print("\nScalar signal test (single time point):")
    for name, feat_map in features:
        if name == 'synchrony':
            continue  # Skip synchrony for scalar
        phi_S = feat_map.extract(y_scalar[50])
        print(f"  {name:20s}: φ_S = {phi_S:.4f}")
        print(f"    params: {feat_map.get_params()}")
    
    print("\nVector signal test (t=50):")
    for name, feat_map in features:
        phi_S = feat_map.extract(y_vector[50])
        print(f"  {name:20s}: φ_S = {phi_S:.4f}")
    
    print("\nFactory creation test:")
    feat = create_feature_map('norm', ord=1)
    print(f"  Created: {feat}")
    print(f"  φ_S = {feat.extract([3.0, 4.0]):.4f}")  # Should be 7.0

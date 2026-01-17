"""
Memory Models: φ_I(t) - Memory / History Features

This module provides memory operators for tracking system history.
Memory features capture "where the system came from" through:
- Exponential weighted moving averages (EWMA)
- Integral states
- Hysteresis models
- Adaptive filters

All implementations are deterministic and stateful (explicit memory).
"""

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class MemoryModel(ABC):
    """
    Abstract base class for memory/history tracking: φ_S(t) → φ_I(t)
    
    Memory models maintain internal state and update based on new features.
    They must be deterministic and reproducible.
    """
    
    @abstractmethod
    def update(self, phi_S: float, dt: float = 1.0) -> float:
        """
        Update memory state with new feature value
        
        Args:
            phi_S: Current state feature φ_S(t)
            dt: Time step
        
        Returns:
            Updated memory feature φ_I(t)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal memory state"""
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Return all parameters for reproducibility"""
        pass
    
    @abstractmethod
    def get_state(self) -> dict:
        """Return current internal state"""
        pass
    
    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class EWMAMemory(MemoryModel):
    """
    Exponential Weighted Moving Average (EWMA)
    
    φ_I(t+1) = α·φ_S(t) + (1-α)·φ_I(t)
    
    Also known as exponential smoothing. Parameter α controls memory decay:
    - α ≈ 0: long memory (slow adaptation)
    - α ≈ 1: short memory (fast adaptation)
    """
    
    def __init__(self, alpha: float = 0.1, initial_value: float = 0.0):
        """
        Args:
            alpha: Smoothing parameter in [0, 1]
            initial_value: Initial memory state
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        
        self.alpha = alpha
        self.initial_value = initial_value
        self.phi_I = initial_value
    
    def update(self, phi_S: float, dt: float = 1.0) -> float:
        """Update EWMA"""
        self.phi_I = self.alpha * phi_S + (1 - self.alpha) * self.phi_I
        return self.phi_I
    
    def reset(self):
        """Reset to initial value"""
        self.phi_I = self.initial_value
    
    def get_params(self) -> dict:
        return {'alpha': self.alpha, 'initial_value': self.initial_value}
    
    def get_state(self) -> dict:
        return {'phi_I': self.phi_I}


class IntegralMemory(MemoryModel):
    """
    Integral (cumulative) memory
    
    φ_I(t) = ∫₀ᵗ k(τ) φ_S(τ) dτ
    
    Discrete approximation: φ_I(t+dt) = φ_I(t) + k·φ_S(t)·dt
    
    With decay: φ_I(t+dt) = λ·φ_I(t) + k·φ_S(t)·dt
    """
    
    def __init__(self, gain: float = 1.0, decay: float = 1.0, 
                 initial_value: float = 0.0):
        """
        Args:
            gain: Integration gain k
            decay: Decay factor λ ∈ (0, 1] (1 = no decay)
            initial_value: Initial integral state
        """
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        
        self.gain = gain
        self.decay = decay
        self.initial_value = initial_value
        self.phi_I = initial_value
    
    def update(self, phi_S: float, dt: float = 1.0) -> float:
        """Update integral with optional decay"""
        self.phi_I = self.decay * self.phi_I + self.gain * phi_S * dt
        return self.phi_I
    
    def reset(self):
        """Reset to initial value"""
        self.phi_I = self.initial_value
    
    def get_params(self) -> dict:
        return {
            'gain': self.gain,
            'decay': self.decay,
            'initial_value': self.initial_value
        }
    
    def get_state(self) -> dict:
        return {'phi_I': self.phi_I}


class HysteresisMemory(MemoryModel):
    """
    Hysteresis-based memory with switching thresholds
    
    Memory "sticks" to current state until threshold is crossed:
    - If φ_S > φ_I + threshold_up: φ_I increases
    - If φ_S < φ_I - threshold_down: φ_I decreases
    - Otherwise: φ_I remains constant
    """
    
    def __init__(self, threshold_up: float = 0.1, threshold_down: float = 0.1,
                 adaptation_rate: float = 0.5, initial_value: float = 0.0):
        """
        Args:
            threshold_up: Threshold for upward switching
            threshold_down: Threshold for downward switching
            adaptation_rate: Rate of adaptation when threshold crossed
            initial_value: Initial memory state
        """
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        self.adaptation_rate = adaptation_rate
        self.initial_value = initial_value
        self.phi_I = initial_value
    
    def update(self, phi_S: float, dt: float = 1.0) -> float:
        """Update hysteresis state"""
        if phi_S > self.phi_I + self.threshold_up:
            # Upward adaptation
            self.phi_I += self.adaptation_rate * (phi_S - self.phi_I) * dt
        elif phi_S < self.phi_I - self.threshold_down:
            # Downward adaptation
            self.phi_I += self.adaptation_rate * (phi_S - self.phi_I) * dt
        # else: no change (hysteresis)
        
        return self.phi_I
    
    def reset(self):
        """Reset to initial value"""
        self.phi_I = self.initial_value
    
    def get_params(self) -> dict:
        return {
            'threshold_up': self.threshold_up,
            'threshold_down': self.threshold_down,
            'adaptation_rate': self.adaptation_rate,
            'initial_value': self.initial_value
        }
    
    def get_state(self) -> dict:
        return {'phi_I': self.phi_I}


class WindowedAverageMemory(MemoryModel):
    """
    Windowed moving average (finite impulse response)
    
    φ_I(t) = (1/window) Σ_{k=0}^{window-1} φ_S(t-k)
    """
    
    def __init__(self, window: int = 10):
        """
        Args:
            window: Window size for averaging
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        
        self.window = window
        self.buffer = []
    
    def update(self, phi_S: float, dt: float = 1.0) -> float:
        """Update windowed average"""
        self.buffer.append(phi_S)
        
        # Maintain window size
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        return float(np.mean(self.buffer))
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def get_params(self) -> dict:
        return {'window': self.window}
    
    def get_state(self) -> dict:
        return {
            'buffer': self.buffer.copy(),
            'current_phi_I': np.mean(self.buffer) if self.buffer else 0.0
        }


class AdaptiveEWMAMemory(MemoryModel):
    """
    Adaptive EWMA with time-varying α based on volatility
    
    When φ_S is volatile, α increases (faster adaptation)
    When φ_S is stable, α decreases (longer memory)
    """
    
    def __init__(self, alpha_min: float = 0.01, alpha_max: float = 0.5,
                 volatility_window: int = 10, initial_value: float = 0.0):
        """
        Args:
            alpha_min: Minimum smoothing parameter
            alpha_max: Maximum smoothing parameter
            volatility_window: Window for volatility estimation
            initial_value: Initial memory state
        """
        if not 0 < alpha_min < alpha_max <= 1:
            raise ValueError("Must have 0 < alpha_min < alpha_max <= 1")
        
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.volatility_window = volatility_window
        self.initial_value = initial_value
        
        self.phi_I = initial_value
        self.history = []
        self.current_alpha = alpha_min
    
    def update(self, phi_S: float, dt: float = 1.0) -> float:
        """Update adaptive EWMA"""
        # Update history for volatility estimation
        self.history.append(phi_S)
        if len(self.history) > self.volatility_window:
            self.history.pop(0)
        
        # Estimate volatility
        if len(self.history) > 1:
            volatility = np.std(self.history)
            # Normalize to [0, 1] (heuristic)
            normalized_vol = np.clip(volatility, 0, 1)
            # Map volatility to alpha
            self.current_alpha = (self.alpha_min + 
                                 (self.alpha_max - self.alpha_min) * normalized_vol)
        
        # Update EWMA with adaptive alpha
        self.phi_I = self.current_alpha * phi_S + (1 - self.current_alpha) * self.phi_I
        return self.phi_I
    
    def reset(self):
        """Reset state and history"""
        self.phi_I = self.initial_value
        self.history.clear()
        self.current_alpha = self.alpha_min
    
    def get_params(self) -> dict:
        return {
            'alpha_min': self.alpha_min,
            'alpha_max': self.alpha_max,
            'volatility_window': self.volatility_window,
            'initial_value': self.initial_value
        }
    
    def get_state(self) -> dict:
        return {
            'phi_I': self.phi_I,
            'current_alpha': self.current_alpha,
            'history_length': len(self.history)
        }


# Registry of available memory models
MEMORY_MODEL_REGISTRY = {
    'ewma': EWMAMemory,
    'integral': IntegralMemory,
    'hysteresis': HysteresisMemory,
    'windowed_average': WindowedAverageMemory,
    'adaptive_ewma': AdaptiveEWMAMemory,
}


def create_memory_model(name: str, **kwargs) -> MemoryModel:
    """
    Factory function to create memory model by name
    
    Args:
        name: Name of memory model (from registry)
        **kwargs: Parameters for memory model constructor
    
    Returns:
        Instantiated MemoryModel
    """
    if name not in MEMORY_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown memory model '{name}'. "
            f"Available: {list(MEMORY_MODEL_REGISTRY.keys())}"
        )
    
    return MEMORY_MODEL_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Memory Model Examples")
    print("=" * 50)
    
    # Generate test signal
    t = np.linspace(0, 10, 100)
    phi_S_signal = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Test different memory models
    models = [
        ('EWMA (α=0.1)', EWMAMemory(alpha=0.1)),
        ('EWMA (α=0.5)', EWMAMemory(alpha=0.5)),
        ('Integral (decay=0.95)', IntegralMemory(gain=0.1, decay=0.95)),
        ('Hysteresis', HysteresisMemory(threshold_up=0.2, threshold_down=0.2)),
        ('Windowed (w=10)', WindowedAverageMemory(window=10)),
        ('Adaptive EWMA', AdaptiveEWMAMemory(alpha_min=0.05, alpha_max=0.4)),
    ]
    
    print("\nProcessing signal (showing every 10th step):")
    for i, phi_S in enumerate(phi_S_signal):
        if i % 10 == 0:
            print(f"\nt = {i:3d}, φ_S = {phi_S:6.3f}")
            for name, model in models:
                phi_I = model.update(phi_S, dt=0.1)
                print(f"  {name:25s}: φ_I = {phi_I:6.3f}")
    
    print("\n" + "=" * 50)
    print("Final states and parameters:")
    for name, model in models:
        print(f"\n{name}:")
        print(f"  Params: {model.get_params()}")
        print(f"  State:  {model.get_state()}")

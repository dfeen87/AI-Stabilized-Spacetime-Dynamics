"""
Constraint Models: φ_C(t) - Admissible Regime Boundaries

This module provides constraint estimation methods for φ_C(t).
Constraints define the "safe set" or admissible operating regime:
- Fixed bounds
- Percentile envelopes
- Control barrier functions
- Adaptive safety margins

All implementations are deterministic and their parameters must be disclosed.
"""

import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod


class ConstraintModel(ABC):
    """
    Abstract base class for constraint estimation
    
    Constraint models define admissible boundaries φ_C(t) that the
    system should not approach or violate.
    """
    
    @abstractmethod
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """
        Estimate constraint boundary φ_C(t)
        
        Args:
            phi_S: Current state feature (optional, for adaptive constraints)
            history: Historical φ_S values (optional, for data-driven estimates)
        
        Returns:
            Constraint value φ_C(t)
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


class FixedConstraint(ConstraintModel):
    """
    Fixed constant constraint: φ_C(t) = C (constant)
    
    Simplest constraint model. Defines a hard boundary that does not adapt.
    """
    
    def __init__(self, value: float):
        """
        Args:
            value: Fixed constraint value
        """
        self.value = value
    
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """Return fixed constraint value"""
        return self.value
    
    def get_params(self) -> dict:
        return {'value': self.value}


class PercentileConstraint(ConstraintModel):
    """
    Data-driven percentile envelope
    
    φ_C(t) = percentile(history, p)
    
    Adapts to historical distribution. Requires sufficient history.
    """
    
    def __init__(self, percentile: float = 95.0, min_history: int = 20,
                 default_value: float = 1.0):
        """
        Args:
            percentile: Percentile level (e.g., 95 for 95th percentile)
            min_history: Minimum history length before computing percentile
            default_value: Default constraint when history insufficient
        """
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in (0, 100)")
        
        self.percentile = percentile
        self.min_history = min_history
        self.default_value = default_value
    
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """Compute percentile constraint from history"""
        if history is None or len(history) < self.min_history:
            return self.default_value
        
        return float(np.percentile(history, self.percentile))
    
    def get_params(self) -> dict:
        return {
            'percentile': self.percentile,
            'min_history': self.min_history,
            'default_value': self.default_value
        }


class RollingMaxConstraint(ConstraintModel):
    """
    Rolling maximum over window
    
    φ_C(t) = max(φ_S[t-window:t]) + margin
    
    Tracks recent peak with safety margin.
    """
    
    def __init__(self, window: int = 50, margin: float = 0.1,
                 default_value: float = 1.0):
        """
        Args:
            window: Window size for rolling maximum
            margin: Safety margin above maximum
            default_value: Default when history insufficient
        """
        self.window = window
        self.margin = margin
        self.default_value = default_value
    
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """Compute rolling max + margin"""
        if history is None or len(history) == 0:
            return self.default_value
        
        recent = history[-self.window:] if len(history) > self.window else history
        return float(np.max(recent)) + self.margin
    
    def get_params(self) -> dict:
        return {
            'window': self.window,
            'margin': self.margin,
            'default_value': self.default_value
        }


class EWMAConstraint(ConstraintModel):
    """
    EWMA-based adaptive constraint
    
    φ_C(t) = ewma(|φ_S|) + k·std(φ_S)
    
    Adapts to mean + multiple standard deviations.
    """
    
    def __init__(self, alpha: float = 0.1, k_std: float = 2.0,
                 min_history: int = 10, default_value: float = 1.0):
        """
        Args:
            alpha: EWMA smoothing parameter
            k_std: Number of standard deviations for margin
            min_history: Minimum history for std computation
            default_value: Default constraint when history insufficient
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        
        self.alpha = alpha
        self.k_std = k_std
        self.min_history = min_history
        self.default_value = default_value
        
        self.ewma_value = None
    
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """Compute EWMA + k·std constraint"""
        if history is None or len(history) < self.min_history:
            return self.default_value
        
        # Update EWMA
        if self.ewma_value is None:
            self.ewma_value = np.mean(np.abs(history))
        elif phi_S is not None:
            self.ewma_value = (self.alpha * abs(phi_S) + 
                              (1 - self.alpha) * self.ewma_value)
        
        # Compute std from recent history
        std_val = np.std(history)
        
        return self.ewma_value + self.k_std * std_val
    
    def reset(self):
        """Reset EWMA state"""
        self.ewma_value = None
    
    def get_params(self) -> dict:
        return {
            'alpha': self.alpha,
            'k_std': self.k_std,
            'min_history': self.min_history,
            'default_value': self.default_value
        }


class ControlBarrierConstraint(ConstraintModel):
    """
    Control Barrier Function (CBF) style constraint
    
    φ_C(t) = h(φ_S) where h defines a safe set
    
    For simple case: h(φ_S) = φ_max - φ_S
    Safety margin defined by φ_S < φ_max
    """
    
    def __init__(self, phi_max: float = 1.0, safety_margin: float = 0.1):
        """
        Args:
            phi_max: Maximum allowable φ_S value
            safety_margin: Additional safety buffer
        """
        self.phi_max = phi_max
        self.safety_margin = safety_margin
    
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """Compute CBF-style constraint"""
        # Constraint is distance to boundary
        return self.phi_max - self.safety_margin
    
    def get_params(self) -> dict:
        return {
            'phi_max': self.phi_max,
            'safety_margin': self.safety_margin
        }


class HybridConstraint(ConstraintModel):
    """
    Hybrid constraint combining multiple models
    
    φ_C(t) = op(C1(t), C2(t), ..., Cn(t))
    
    where op can be min, max, mean, etc.
    """
    
    def __init__(self, models: List[ConstraintModel], 
                 operation: str = 'min'):
        """
        Args:
            models: List of constraint models to combine
            operation: How to combine ('min', 'max', 'mean')
        """
        if not models:
            raise ValueError("Must provide at least one constraint model")
        
        valid_ops = ['min', 'max', 'mean']
        if operation not in valid_ops:
            raise ValueError(f"operation must be one of {valid_ops}")
        
        self.models = models
        self.operation = operation
    
    def estimate(self, phi_S: Optional[float] = None, 
                history: Optional[List[float]] = None) -> float:
        """Combine multiple constraint estimates"""
        estimates = [model.estimate(phi_S, history) for model in self.models]
        
        if self.operation == 'min':
            return min(estimates)
        elif self.operation == 'max':
            return max(estimates)
        elif self.operation == 'mean':
            return float(np.mean(estimates))
        
        return estimates[0]  # Fallback
    
    def get_params(self) -> dict:
        return {
            'operation': self.operation,
            'num_models': len(self.models),
            'models': [m.get_params() for m in self.models]
        }


# Registry of available constraint models
CONSTRAINT_MODEL_REGISTRY = {
    'fixed': FixedConstraint,
    'percentile': PercentileConstraint,
    'rolling_max': RollingMaxConstraint,
    'ewma': EWMAConstraint,
    'control_barrier': ControlBarrierConstraint,
}


def create_constraint_model(name: str, **kwargs) -> ConstraintModel:
    """
    Factory function to create constraint model by name
    
    Args:
        name: Name of constraint model (from registry)
        **kwargs: Parameters for constraint model constructor
    
    Returns:
        Instantiated ConstraintModel
    """
    if name not in CONSTRAINT_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown constraint model '{name}'. "
            f"Available: {list(CONSTRAINT_MODEL_REGISTRY.keys())}"
        )
    
    return CONSTRAINT_MODEL_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Constraint Model Examples")
    print("=" * 50)
    
    # Generate test history
    np.random.seed(42)
    history = [0.5 + 0.3 * np.sin(i/10) + 0.1 * np.random.randn() 
              for i in range(100)]
    
    # Test different constraint models
    models = [
        ('Fixed (C=1.0)', FixedConstraint(value=1.0)),
        ('Percentile (95%)', PercentileConstraint(percentile=95)),
        ('Rolling Max (w=20)', RollingMaxConstraint(window=20, margin=0.1)),
        ('EWMA + 2σ', EWMAConstraint(alpha=0.1, k_std=2.0)),
        ('Control Barrier', ControlBarrierConstraint(phi_max=1.2, safety_margin=0.15)),
    ]
    
    print("\nConstraint estimates (using full history):")
    phi_S_current = history[-1]
    
    for name, model in models:
        phi_C = model.estimate(phi_S=phi_S_current, history=history)
        print(f"\n{name}:")
        print(f"  Params: {model.get_params()}")
        print(f"  φ_C = {phi_C:.4f}")
        print(f"  Margin (φ_C - φ_S) = {phi_C - phi_S_current:.4f}")
    
    # Test hybrid model
    print("\n" + "=" * 50)
    print("Hybrid Constraint (min of Fixed and Percentile):")
    hybrid = HybridConstraint(
        models=[
            FixedConstraint(1.0),
            PercentileConstraint(percentile=90)
        ],
        operation='min'
    )
    phi_C_hybrid = hybrid.estimate(phi_S=phi_S_current, history=history)
    print(f"  φ_C = {phi_C_hybrid:.4f}")
    print(f"  Params: {hybrid.get_params()}")

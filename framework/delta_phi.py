"""
Core ΔΦ Operator Implementation

This module implements the triadic phase operator ΔΦ as a deterministic diagnostic tool.
It is explicitly NOT a controller, optimizer, or predictor.

The ΔΦ operator evaluates system evolution relative to history and constraints without
issuing actions or modifying system inputs.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TriadicFeatures:
    """
    Triadic feature state: Φ(t) = {φ_S(t), φ_I(t), φ_C(t)}
    
    Attributes:
        phi_S: State feature (instantaneous structural configuration)
        phi_I: Integral/memory feature (accumulated history)
        phi_C: Constraint feature (admissible regime boundary)
        timestamp: Time index for this feature state
    """
    phi_S: float
    phi_I: float
    phi_C: float
    timestamp: Optional[float] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector [φ_S, φ_I, φ_C]"""
        return np.array([self.phi_S, self.phi_I, self.phi_C])
    
    def __repr__(self) -> str:
        t_str = f", t={self.timestamp:.3f}" if self.timestamp is not None else ""
        return f"Φ(φ_S={self.phi_S:.4f}, φ_I={self.phi_I:.4f}, φ_C={self.phi_C:.4f}{t_str})"


@dataclass
class DeltaPhiVector:
    """
    The ΔΦ instability vector
    
    ΔΦ(t) = [φ̇_S(t), e_I(t), e_C(t)]ᵀ
          = [φ̇_S(t), φ_S(t) - φ_I(t), φ_C(t) - φ_S(t)]ᵀ
    
    Attributes:
        phi_S_dot: Rate of change of state feature (local variation)
        e_I: Memory error (coherence-to-history mismatch)
        e_C: Constraint margin (boundary proximity)
        timestamp: Time index for this operator evaluation
    """
    phi_S_dot: float
    e_I: float
    e_C: float
    timestamp: Optional[float] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector [φ̇_S, e_I, e_C]"""
        return np.array([self.phi_S_dot, self.e_I, self.e_C])
    
    def norm(self) -> float:
        """Euclidean norm of ΔΦ vector"""
        return np.linalg.norm(self.to_vector())
    
    def weighted_norm_squared(self, w_S: float = 1.0, w_I: float = 1.0, 
                             w_C: float = 1.0) -> float:
        """
        Weighted squared norm: ‖ΔΦ‖²_W
        
        Returns: w_S·φ̇_S² + w_I·e_I² + w_C·e_C²
        """
        return (w_S * self.phi_S_dot**2 + 
                w_I * self.e_I**2 + 
                w_C * self.e_C**2)
    
    def __repr__(self) -> str:
        t_str = f", t={self.timestamp:.3f}" if self.timestamp is not None else ""
        return (f"ΔΦ(φ̇_S={self.phi_S_dot:.4f}, "
                f"e_I={self.e_I:.4f}, e_C={self.e_C:.4f}{t_str})")


class DeltaPhiOperator:
    """
    Triadic Phase Operator ΔΦ - Diagnostic Observer (NOT a controller)
    
    This operator evaluates system stability by computing:
    1. Local variation rate: φ̇_S(t)
    2. Deviation from historical coherence: e_I(t) = φ_S(t) - φ_I(t)
    3. Distance to constraint boundary: e_C(t) = φ_C(t) - φ_S(t)
    
    The operator is purely diagnostic and does NOT:
    - Issue control commands
    - Modify system inputs
    - Enforce constraints autonomously
    - Replace human decision-making
    """
    
    def __init__(self, dt: float = 1.0, derivative_method: str = 'finite_difference'):
        """
        Initialize ΔΦ operator
        
        Args:
            dt: Time step for derivative computation
            derivative_method: Method for computing φ̇_S 
                             ('finite_difference', 'backward', 'central')
        """
        self.dt = dt
        self.derivative_method = derivative_method
        self._prev_phi_S: Optional[float] = None
        self._prev_prev_phi_S: Optional[float] = None
        
    def compute(self, features: TriadicFeatures, 
                prev_features: Optional[TriadicFeatures] = None) -> DeltaPhiVector:
        """
        Compute ΔΦ(t) from current (and optionally previous) triadic features
        
        Args:
            features: Current triadic feature state Φ(t)
            prev_features: Previous triadic feature state Φ(t-dt) for derivative
        
        Returns:
            DeltaPhiVector containing [φ̇_S, e_I, e_C]
        """
        # Compute derivative φ̇_S(t)
        phi_S_dot = self._compute_derivative(features.phi_S, prev_features)
        
        # Compute error terms
        e_I = features.phi_S - features.phi_I  # Memory error
        e_C = features.phi_C - features.phi_S  # Constraint margin
        
        return DeltaPhiVector(
            phi_S_dot=phi_S_dot,
            e_I=e_I,
            e_C=e_C,
            timestamp=features.timestamp
        )
    
    def _compute_derivative(self, phi_S: float, 
                           prev_features: Optional[TriadicFeatures]) -> float:
        """
        Compute φ̇_S(t) using specified derivative method
        
        Args:
            phi_S: Current state feature value
            prev_features: Previous features (if available)
        
        Returns:
            Estimated derivative φ̇_S
        """
        if self.derivative_method == 'finite_difference':
            return self._finite_difference(phi_S, prev_features)
        elif self.derivative_method == 'backward':
            return self._backward_difference(phi_S, prev_features)
        elif self.derivative_method == 'central':
            return self._central_difference(phi_S, prev_features)
        else:
            raise ValueError(f"Unknown derivative method: {self.derivative_method}")
    
    def _finite_difference(self, phi_S: float, 
                          prev_features: Optional[TriadicFeatures]) -> float:
        """Forward difference: (φ_S(t) - φ_S(t-1)) / dt"""
        if prev_features is None:
            # First call - no history yet
            self._prev_phi_S = phi_S
            return 0.0
        
        derivative = (phi_S - prev_features.phi_S) / self.dt
        self._prev_phi_S = phi_S
        return derivative
    
    def _backward_difference(self, phi_S: float, 
                            prev_features: Optional[TriadicFeatures]) -> float:
        """Same as finite difference for this implementation"""
        return self._finite_difference(phi_S, prev_features)
    
    def _central_difference(self, phi_S: float, 
                           prev_features: Optional[TriadicFeatures]) -> float:
        """Central difference: (φ_S(t) - φ_S(t-2)) / (2·dt)"""
        if self._prev_prev_phi_S is None:
            # Not enough history for central difference yet
            deriv = self._finite_difference(phi_S, prev_features)
        else:
            deriv = (phi_S - self._prev_prev_phi_S) / (2 * self.dt)
        
        # Update history
        self._prev_prev_phi_S = self._prev_phi_S
        self._prev_phi_S = phi_S
        
        return deriv
    
    def reset(self):
        """Reset internal state (history for derivatives)"""
        self._prev_phi_S = None
        self._prev_prev_phi_S = None
    
    def __repr__(self) -> str:
        return (f"DeltaPhiOperator(dt={self.dt}, "
                f"method='{self.derivative_method}')")


def compute_delta_phi(phi_S: float, phi_I: float, phi_C: float,
                      phi_S_prev: Optional[float] = None,
                      dt: float = 1.0) -> DeltaPhiVector:
    """
    Convenience function to compute ΔΦ from raw feature values
    
    Args:
        phi_S: Current state feature
        phi_I: Current memory feature
        phi_C: Current constraint feature
        phi_S_prev: Previous state feature (for derivative)
        dt: Time step
    
    Returns:
        DeltaPhiVector
    """
    features = TriadicFeatures(phi_S=phi_S, phi_I=phi_I, phi_C=phi_C)
    prev_features = (TriadicFeatures(phi_S=phi_S_prev, phi_I=0, phi_C=0) 
                    if phi_S_prev is not None else None)
    
    operator = DeltaPhiOperator(dt=dt)
    return operator.compute(features, prev_features)


if __name__ == "__main__":
    # Example usage (diagnostic demonstration only)
    print("ΔΦ Operator Diagnostic Example")
    print("=" * 50)
    
    # Simulate a simple feature trajectory
    time_steps = 10
    dt = 0.1
    
    operator = DeltaPhiOperator(dt=dt)
    
    for t in range(time_steps):
        # Synthetic features (for demonstration only)
        phi_S = np.sin(2 * np.pi * t / time_steps)
        phi_I = 0.8 * phi_S  # Memory lags behind
        phi_C = 1.2  # Fixed constraint boundary
        
        features = TriadicFeatures(phi_S, phi_I, phi_C, timestamp=t*dt)
        prev_features = (TriadicFeatures(
            np.sin(2 * np.pi * (t-1) / time_steps), 0, 0, (t-1)*dt
        ) if t > 0 else None)
        
        delta_phi = operator.compute(features, prev_features)
        
        print(f"\nt = {t*dt:.2f}")
        print(f"  {features}")
        print(f"  {delta_phi}")
        print(f"  ‖ΔΦ‖ = {delta_phi.norm():.4f}")

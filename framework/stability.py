"""
Stability Score J(t) and Regime Detection

This module implements the stability score and regime flagging based on the ΔΦ operator.
All functions are diagnostic only and do NOT issue control actions.

The stability score J(t) = ‖ΔΦ(t)‖²_W provides an interpretable measure of:
- System volatility (via φ̇_S term)
- Deviation from history (via e_I term)  
- Proximity to constraints (via e_C term)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum

from .delta_phi import DeltaPhiVector


class RegimeState(Enum):
    """Regime classification based on stability score"""
    STABLE = "stable"
    WARNING = "warning"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


@dataclass
class StabilityWeights:
    """
    Weights for the stability score J(t) = ‖ΔΦ‖²_W
    
    J(t) = w_S·φ̇_S² + w_I·e_I² + w_C·e_C²
    
    Attributes:
        w_S: Weight for state variation term (volatility)
        w_I: Weight for memory error term (history deviation)
        w_C: Weight for constraint margin term (boundary proximity)
    """
    w_S: float = 1.0
    w_I: float = 1.0
    w_C: float = 1.0
    
    def __post_init__(self):
        if self.w_S < 0 or self.w_I < 0 or self.w_C < 0:
            raise ValueError("All weights must be non-negative")
    
    def normalize(self) -> 'StabilityWeights':
        """Return normalized weights (sum to 1)"""
        total = self.w_S + self.w_I + self.w_C
        if total == 0:
            raise ValueError("Cannot normalize zero weights")
        return StabilityWeights(
            w_S=self.w_S / total,
            w_I=self.w_I / total,
            w_C=self.w_C / total
        )


@dataclass
class RegimeThresholds:
    """
    Thresholds for regime classification based on J(t)
    
    Regime classification:
    - STABLE:   J(t) < epsilon_stable²
    - WARNING:  epsilon_stable² ≤ J(t) < epsilon_warning²
    - UNSTABLE: epsilon_warning² ≤ J(t) < epsilon_critical²
    - CRITICAL: J(t) ≥ epsilon_critical²
    """
    epsilon_stable: float = 0.1
    epsilon_warning: float = 0.5
    epsilon_critical: float = 1.0
    
    def __post_init__(self):
        if not (0 < self.epsilon_stable < self.epsilon_warning < self.epsilon_critical):
            raise ValueError(
                "Thresholds must satisfy: 0 < stable < warning < critical"
            )


@dataclass
class StabilityScore:
    """
    Stability score J(t) and associated diagnostics
    
    Attributes:
        J: Overall stability score (weighted squared norm)
        J_S: State variation component (w_S·φ̇_S²)
        J_I: Memory error component (w_I·e_I²)
        J_C: Constraint margin component (w_C·e_C²)
        regime: Classified regime state
        timestamp: Time index
    """
    J: float
    J_S: float
    J_I: float
    J_C: float
    regime: RegimeState
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging/storage"""
        return {
            'J': self.J,
            'J_S': self.J_S,
            'J_I': self.J_I,
            'J_C': self.J_C,
            'regime': self.regime.value,
            'timestamp': self.timestamp if self.timestamp is not None else -1
        }
    
    def __repr__(self) -> str:
        t_str = f", t={self.timestamp:.3f}" if self.timestamp is not None else ""
        return (f"J(t)={self.J:.4f} [{self.regime.value}]"
                f"(J_S={self.J_S:.4f}, J_I={self.J_I:.4f}, J_C={self.J_C:.4f}{t_str})")


class StabilityMonitor:
    """
    Stability monitoring and regime detection based on ΔΦ operator
    
    This is a DIAGNOSTIC TOOL ONLY. It does not:
    - Issue control commands
    - Modify system behavior
    - Enforce constraints
    - Make autonomous decisions
    
    It only evaluates and reports stability metrics.
    """
    
    def __init__(self, 
                 weights: Optional[StabilityWeights] = None,
                 thresholds: Optional[RegimeThresholds] = None,
                 history_length: int = 100):
        """
        Initialize stability monitor
        
        Args:
            weights: Weights for stability score components
            thresholds: Thresholds for regime classification
            history_length: Number of past scores to retain
        """
        self.weights = weights or StabilityWeights()
        self.thresholds = thresholds or RegimeThresholds()
        self.history_length = history_length
        
        # History buffers
        self.score_history: List[StabilityScore] = []
        self.delta_phi_history: List[DeltaPhiVector] = []
    
    def compute_score(self, delta_phi: DeltaPhiVector) -> StabilityScore:
        """
        Compute stability score J(t) from ΔΦ(t)
        
        Args:
            delta_phi: The ΔΦ vector
        
        Returns:
            StabilityScore with overall score and components
        """
        # Compute weighted components
        J_S = self.weights.w_S * delta_phi.phi_S_dot**2
        J_I = self.weights.w_I * delta_phi.e_I**2
        J_C = self.weights.w_C * delta_phi.e_C**2
        
        # Total score
        J = J_S + J_I + J_C
        
        # Classify regime
        regime = self._classify_regime(J)
        
        score = StabilityScore(
            J=J,
            J_S=J_S,
            J_I=J_I,
            J_C=J_C,
            regime=regime,
            timestamp=delta_phi.timestamp
        )
        
        # Store in history
        self._update_history(score, delta_phi)
        
        return score
    
    def _classify_regime(self, J: float) -> RegimeState:
        """Classify regime based on stability score"""
        J_sqrt = np.sqrt(J)  # Compare against epsilon (not epsilon²)
        
        if J_sqrt < self.thresholds.epsilon_stable:
            return RegimeState.STABLE
        elif J_sqrt < self.thresholds.epsilon_warning:
            return RegimeState.WARNING
        elif J_sqrt < self.thresholds.epsilon_critical:
            return RegimeState.UNSTABLE
        else:
            return RegimeState.CRITICAL
    
    def _update_history(self, score: StabilityScore, delta_phi: DeltaPhiVector):
        """Update history buffers with length limit"""
        self.score_history.append(score)
        self.delta_phi_history.append(delta_phi)
        
        # Trim to history_length
        if len(self.score_history) > self.history_length:
            self.score_history = self.score_history[-self.history_length:]
        if len(self.delta_phi_history) > self.history_length:
            self.delta_phi_history = self.delta_phi_history[-self.history_length:]
    
    def is_coherently_stable(self, horizon: int) -> bool:
        """
        Check if system is coherently stable over recent horizon
        
        A system is coherently stable on [t-horizon, t] if:
            sup_{τ∈[t-horizon,t]} J(τ) < epsilon_stable²
        
        Args:
            horizon: Number of recent time steps to check
        
        Returns:
            True if coherently stable over horizon
        """
        if len(self.score_history) < horizon:
            return False  # Insufficient history
        
        recent_scores = self.score_history[-horizon:]
        max_score = max(score.J for score in recent_scores)
        
        return max_score < self.thresholds.epsilon_stable**2
    
    def get_regime_statistics(self) -> Dict[str, float]:
        """
        Compute statistics over score history
        
        Returns:
            Dictionary with mean, std, max, min of J(t)
        """
        if not self.score_history:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        scores = [s.J for s in self.score_history]
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'max': np.max(scores),
            'min': np.min(scores),
            'current': scores[-1] if scores else 0.0
        }
    
    def detect_transition(self, lookback: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Detect potential regime transitions
        
        Args:
            lookback: Number of steps to analyze for transition
        
        Returns:
            (is_transition, transition_type) where transition_type is:
            - "volatility_spike" if J_S dominates
            - "history_drift" if J_I dominates
            - "constraint_approach" if J_C dominates
            - None if no clear transition
        """
        if len(self.score_history) < lookback:
            return False, None
        
        recent = self.score_history[-lookback:]
        
        # Check for regime change
        regimes = [s.regime for s in recent]
        if len(set(regimes)) == 1:
            return False, None  # No regime change
        
        # Identify dominant component in recent increase
        latest = recent[-1]
        components = {'J_S': latest.J_S, 'J_I': latest.J_I, 'J_C': latest.J_C}
        dominant = max(components, key=components.get)
        
        transition_map = {
            'J_S': 'volatility_spike',
            'J_I': 'history_drift',
            'J_C': 'constraint_approach'
        }
        
        return True, transition_map[dominant]
    
    def reset(self):
        """Clear all history"""
        self.score_history.clear()
        self.delta_phi_history.clear()
    
    def __repr__(self) -> str:
        return (f"StabilityMonitor(weights={self.weights}, "
                f"thresholds={self.thresholds}, "
                f"history={len(self.score_history)} steps)")


if __name__ == "__main__":
    # Example usage
    print("Stability Monitoring Example")
    print("=" * 50)
    
    # Create monitor with custom settings
    weights = StabilityWeights(w_S=1.0, w_I=0.5, w_C=2.0)
    thresholds = RegimeThresholds(
        epsilon_stable=0.1,
        epsilon_warning=0.5,
        epsilon_critical=1.0
    )
    monitor = StabilityMonitor(weights=weights, thresholds=thresholds)
    
    # Simulate evolving ΔΦ signals
    for t in range(20):
        # Create synthetic ΔΦ vector
        phi_S_dot = 0.1 * np.sin(2 * np.pi * t / 10)
        e_I = 0.2 * t / 20  # Slowly drifting from history
        e_C = 1.0 - 0.05 * t  # Approaching constraint
        
        delta_phi = DeltaPhiVector(
            phi_S_dot=phi_S_dot,
            e_I=e_I,
            e_C=e_C,
            timestamp=t * 0.1
        )
        
        score = monitor.compute_score(delta_phi)
        
        if t % 5 == 0:
            print(f"\nt = {t*0.1:.2f}")
            print(f"  {delta_phi}")
            print(f"  {score}")
            
            # Check for transitions
            is_trans, trans_type = monitor.detect_transition(lookback=5)
            if is_trans:
                print(f"  ⚠ Transition detected: {trans_type}")
    
    print("\n" + "=" * 50)
    print("Statistics:")
    stats = monitor.get_regime_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

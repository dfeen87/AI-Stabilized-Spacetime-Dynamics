"""
Diagnostic Observer Pipeline: y(t) → ΔΦ → J(t)

This module implements the complete diagnostic pipeline as shown in the paper.
It is EXPLICITLY NON-INTERVENTIONAL - it observes and diagnoses only.

Pipeline stages:
1. Signal input: y(t)
2. Feature extraction: y(t) → φ_S(t)
3. Memory update: φ_S(t) → φ_I(t)
4. Constraint estimation: history → φ_C(t)
5. ΔΦ computation: {φ_S, φ_I, φ_C} → ΔΦ
6. Stability scoring: ΔΦ → J(t)
7. Regime flagging: J(t) → {stable, warning, unstable, critical}

NO CONTROL ACTIONS ARE ISSUED AT ANY STAGE.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# Import framework components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.delta_phi import (
    DeltaPhiOperator, TriadicFeatures, DeltaPhiVector
)
from framework.stability import (
    StabilityMonitor, StabilityScore, StabilityWeights, RegimeThresholds
)
from features.feature_maps import FeatureMap, create_feature_map
from features.memory_models import MemoryModel, create_memory_model
from features.constraint_models import ConstraintModel, create_constraint_model


@dataclass
class ObserverConfig:
    """
    Configuration for diagnostic observer
    
    All parameters must be explicitly specified for reproducibility.
    """
    # Feature extraction
    feature_map: str = 'identity'
    feature_params: Dict = None
    
    # Memory model
    memory_model: str = 'ewma'
    memory_params: Dict = None
    
    # Constraint model
    constraint_model: str = 'fixed'
    constraint_params: Dict = None
    
    # ΔΦ operator
    dt: float = 1.0
    derivative_method: str = 'finite_difference'
    
    # Stability weights and thresholds
    w_S: float = 1.0
    w_I: float = 1.0
    w_C: float = 1.0
    epsilon_stable: float = 0.1
    epsilon_warning: float = 0.5
    epsilon_critical: float = 1.0
    
    # History tracking
    history_length: int = 100
    
    def __post_init__(self):
        if self.feature_params is None:
            self.feature_params = {}
        if self.memory_params is None:
            self.memory_params = {}
        if self.constraint_params is None:
            self.constraint_params = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'feature_map': self.feature_map,
            'feature_params': self.feature_params,
            'memory_model': self.memory_model,
            'memory_params': self.memory_params,
            'constraint_model': self.constraint_model,
            'constraint_params': self.constraint_params,
            'dt': self.dt,
            'derivative_method': self.derivative_method,
            'w_S': self.w_S,
            'w_I': self.w_I,
            'w_C': self.w_C,
            'epsilon_stable': self.epsilon_stable,
            'epsilon_warning': self.epsilon_warning,
            'epsilon_critical': self.epsilon_critical,
            'history_length': self.history_length
        }


@dataclass
class ObserverOutput:
    """
    Complete diagnostic output at time t
    
    Contains all intermediate values for transparency and auditability.
    """
    timestamp: float
    y: np.ndarray  # Input signal
    phi_S: float  # State feature
    phi_I: float  # Memory feature
    phi_C: float  # Constraint feature
    delta_phi: DeltaPhiVector  # ΔΦ operator output
    score: StabilityScore  # Stability score J(t)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'timestamp': self.timestamp,
            'y': self.y.tolist() if isinstance(self.y, np.ndarray) else self.y,
            'phi_S': self.phi_S,
            'phi_I': self.phi_I,
            'phi_C': self.phi_C,
            'delta_phi': {
                'phi_S_dot': self.delta_phi.phi_S_dot,
                'e_I': self.delta_phi.e_I,
                'e_C': self.delta_phi.e_C
            },
            'score': self.score.to_dict()
        }
    
    def __repr__(self) -> str:
        return (f"ObserverOutput(t={self.timestamp:.3f}, "
                f"regime={self.score.regime.value}, J={self.score.J:.4f})")


class DiagnosticObserver:
    """
    Complete diagnostic observer pipeline
    
    This is a DIAGNOSTIC TOOL ONLY. It:
    - Observes system behavior
    - Extracts features
    - Computes stability metrics
    - Flags potential regime transitions
    
    It does NOT:
    - Issue control commands
    - Modify system inputs u(t)
    - Enforce constraints autonomously
    - Replace human decision-making
    
    All AI/learning components (if any) are confined to feature extraction
    and constraint estimation. The diagnostic logic remains explicit and
    deterministic.
    """
    
    def __init__(self, config: ObserverConfig):
        """
        Initialize diagnostic observer with explicit configuration
        
        Args:
            config: Complete observer configuration (all parameters disclosed)
        """
        self.config = config
        
        # Initialize components
        self.feature_map: FeatureMap = create_feature_map(
            config.feature_map, **config.feature_params
        )
        
        self.memory_model: MemoryModel = create_memory_model(
            config.memory_model, **config.memory_params
        )
        
        self.constraint_model: ConstraintModel = create_constraint_model(
            config.constraint_model, **config.constraint_params
        )
        
        self.delta_phi_op = DeltaPhiOperator(
            dt=config.dt,
            derivative_method=config.derivative_method
        )
        
        self.stability_monitor = StabilityMonitor(
            weights=StabilityWeights(
                w_S=config.w_S,
                w_I=config.w_I,
                w_C=config.w_C
            ),
            thresholds=RegimeThresholds(
                epsilon_stable=config.epsilon_stable,
                epsilon_warning=config.epsilon_warning,
                epsilon_critical=config.epsilon_critical
            ),
            history_length=config.history_length
        )
        
        # History tracking for constraint estimation
        self.phi_S_history: List[float] = []
        self.prev_features: Optional[TriadicFeatures] = None
        
        # Output history
        self.output_history: List[ObserverOutput] = []
        
        # Time tracking
        self.current_time = 0.0
    
    def process(self, y: np.ndarray, 
                timestamp: Optional[float] = None) -> ObserverOutput:
        """
        Process one time step through diagnostic pipeline
        
        Args:
            y: Input signal y(t) (scalar or vector)
            timestamp: Optional explicit timestamp (auto-incremented if None)
        
        Returns:
            ObserverOutput containing all diagnostic information
        """
        # Update timestamp
        if timestamp is None:
            timestamp = self.current_time
            self.current_time += self.config.dt
        else:
            self.current_time = timestamp
        
        # Stage 1: Feature extraction
        phi_S = self.feature_map.extract(y)
        
        # Stage 2: Memory update
        phi_I = self.memory_model.update(phi_S, dt=self.config.dt)
        
        # Stage 3: Constraint estimation
        self.phi_S_history.append(phi_S)
        if len(self.phi_S_history) > self.config.history_length:
            self.phi_S_history.pop(0)
        
        phi_C = self.constraint_model.estimate(
            phi_S=phi_S,
            history=self.phi_S_history
        )
        
        # Stage 4: Construct triadic features
        features = TriadicFeatures(
            phi_S=phi_S,
            phi_I=phi_I,
            phi_C=phi_C,
            timestamp=timestamp
        )
        
        # Stage 5: Compute ΔΦ
        delta_phi = self.delta_phi_op.compute(features, self.prev_features)
        delta_phi.timestamp = timestamp
        
        # Stage 6: Compute stability score
        score = self.stability_monitor.compute_score(delta_phi)
        
        # Create output
        output = ObserverOutput(
            timestamp=timestamp,
            y=y,
            phi_S=phi_S,
            phi_I=phi_I,
            phi_C=phi_C,
            delta_phi=delta_phi,
            score=score
        )
        
        # Store for next iteration
        self.prev_features = features
        self.output_history.append(output)
        
        # Trim output history
        if len(self.output_history) > self.config.history_length:
            self.output_history.pop(0)
        
        return output
    
    def process_batch(self, y_sequence: np.ndarray,
                     timestamps: Optional[np.ndarray] = None) -> List[ObserverOutput]:
        """
        Process a batch of signals
        
        Args:
            y_sequence: Sequence of signals [y(0), y(1), ..., y(T)]
            timestamps: Optional timestamp array
        
        Returns:
            List of ObserverOutput for each time step
        """
        outputs = []
        
        for i, y in enumerate(y_sequence):
            t = timestamps[i] if timestamps is not None else None
            output = self.process(y, timestamp=t)
            outputs.append(output)
        
        return outputs
    
    def get_diagnostics_summary(self) -> Dict:
        """
        Get summary statistics of recent diagnostics
        
        Returns:
            Dictionary with diagnostic metrics
        """
        if not self.output_history:
            return {}
        
        regime_counts = {}
        for output in self.output_history:
            regime = output.score.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        scores = [o.score.J for o in self.output_history]
        
        return {
            'num_observations': len(self.output_history),
            'regime_distribution': regime_counts,
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'min': np.min(scores)
            },
            'current_regime': self.output_history[-1].score.regime.value,
            'stability_monitor': self.stability_monitor.get_regime_statistics()
        }
    
    def detect_transitions(self, lookback: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Detect regime transitions in recent history
        
        Args:
            lookback: Number of steps to analyze
        
        Returns:
            (is_transition, transition_type)
        """
        return self.stability_monitor.detect_transition(lookback=lookback)
    
    def is_stable(self, horizon: int = 10) -> bool:
        """
        Check if system is coherently stable over horizon
        
        Args:
            horizon: Number of recent steps to check
        
        Returns:
            True if coherently stable
        """
        return self.stability_monitor.is_coherently_stable(horizon)
    
    def reset(self):
        """Reset all components to initial state"""
        self.memory_model.reset()
        self.delta_phi_op.reset()
        self.stability_monitor.reset()
        
        self.phi_S_history.clear()
        self.prev_features = None
        self.output_history.clear()
        self.current_time = 0.0
    
    def get_config(self) -> Dict:
        """Get complete configuration for reproducibility"""
        return self.config.to_dict()
    
    def __repr__(self) -> str:
        return (f"DiagnosticObserver(\n"
                f"  feature_map={self.feature_map},\n"
                f"  memory_model={self.memory_model},\n"
                f"  constraint_model={self.constraint_model},\n"
                f"  observations={len(self.output_history)}\n"
                f")")


if __name__ == "__main__":
    # Example usage
    print("Diagnostic Observer Pipeline Example")
    print("=" * 60)
    
    # Create configuration
    config = ObserverConfig(
        feature_map='identity',
        memory_model='ewma',
        memory_params={'alpha': 0.2},
        constraint_model='percentile',
        constraint_params={'percentile': 95},
        w_S=1.0,
        w_I=0.5,
        w_C=2.0,
        dt=0.1
    )
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Create observer
    observer = DiagnosticObserver(config)
    print(f"\n{observer}")
    
    # Generate synthetic signal
    print("\nProcessing synthetic signal...")
    t = np.linspace(0, 10, 100)
    y_signal = np.sin(2 * np.pi * t) + 0.2 * np.random.randn(100)
    
    # Process signal
    outputs = observer.process_batch(y_signal, timestamps=t)
    
    # Display results (every 10th step)
    print("\nDiagnostic Results (every 10th step):")
    for i in range(0, len(outputs), 10):
        output = outputs[i]
        print(f"\nt = {output.timestamp:.2f}")
        print(f"  y = {output.y:.4f}")
        print(f"  Φ(t) = (φ_S={output.phi_S:.4f}, "
              f"φ_I={output.phi_I:.4f}, φ_C={output.phi_C:.4f})")
        print(f"  {output.delta_phi}")
        print(f"  {output.score}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Diagnostics Summary:")
    summary = observer.get_diagnostics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Check for transitions
    is_trans, trans_type = observer.detect_transitions(lookback=20)
    print(f"\nTransition detected: {is_trans}")
    if is_trans:
        print(f"  Type: {trans_type}")
    
    # Stability check
    is_stable = observer.is_stable(horizon=20)
    print(f"Coherently stable (20 steps): {is_stable}")

"""
Core ΔΦ Framework Components

This package implements the fundamental operators and data structures
for the ΔΦ (Delta-Phi) diagnostic framework.

Contents:
- Triadic feature representation (φ_S, φ_I, φ_C)
- ΔΦ operator and vector
- Stability scoring and regime classification

This package is diagnostic-only and issues NO control actions.
"""

from .delta_phi import (
    TriadicFeatures,
    DeltaPhiVector,
    DeltaPhiOperator,
)

from .stability import (
    StabilityMonitor,
    StabilityScore,
    StabilityWeights,
    RegimeThresholds,
    RegimeState,
)

__all__ = [
    # ΔΦ core
    "TriadicFeatures",
    "DeltaPhiVector",
    "DeltaPhiOperator",
    # Stability
    "StabilityMonitor",
    "StabilityScore",
    "StabilityWeights",
    "RegimeThresholds",
    "RegimeState",
]

"""
Diagnostic Observer Pipeline

This package implements the end-to-end diagnostic observer:
y(t) → φ → ΔΦ → J(t) → regime classification

The pipeline is STRICTLY NON-INTERVENTIONAL and issues no control actions.
"""

from .observer import (
    DiagnosticObserver,
    ObserverConfig,
    ObserverOutput,
)

from .config import (
    ObserverConfiguration,
    FeatureConfig,
    MemoryConfig,
    ConstraintConfig,
    StabilityConfig,
    PipelineConfig,
    create_default_config,
)

__all__ = [
    # Observer
    "DiagnosticObserver",
    "ObserverConfig",
    "ObserverOutput",
    # Configuration
    "ObserverConfiguration",
    "FeatureConfig",
    "MemoryConfig",
    "ConstraintConfig",
    "StabilityConfig",
    "PipelineConfig",
    "create_default_config",
]

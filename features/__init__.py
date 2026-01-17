"""
Feature, Memory, and Constraint Models

This package contains all modular components used to construct the
triadic diagnostic features:

- φ_S(t): Instantaneous state features
- φ_I(t): Memory / history features
- φ_C(t): Constraint / admissible regime features

All components are deterministic and explicitly parameterized.
"""

from .feature_maps import (
    FeatureMap,
    create_feature_map,
)

from .memory_models import (
    MemoryModel,
    create_memory_model,
)

from .constraint_models import (
    ConstraintModel,
    create_constraint_model,
)

__all__ = [
    # Feature maps
    "FeatureMap",
    "create_feature_map",
    # Memory models
    "MemoryModel",
    "create_memory_model",
    # Constraint models
    "ConstraintModel",
    "create_constraint_model",
]

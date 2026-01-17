"""
Configuration Management for Diagnostic Observer

This module provides configuration loading, saving, validation, and locking
for reproducible experiments. All parameters must be explicitly disclosed
for falsifiability.

Configuration includes:
- Feature extraction settings
- Memory model parameters
- Constraint estimation parameters
- Stability weights and thresholds
- Pipeline settings (dt, history length, etc.)
"""

import json
import yaml
import hashlib
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class FeatureConfig:
    """Configuration for feature extraction (φ_S)"""
    map_type: str = 'identity'
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    def to_dict(self) -> Dict:
        return {'map_type': self.map_type, 'params': self.params}


@dataclass
class MemoryConfig:
    """Configuration for memory model (φ_I)"""
    model_type: str = 'ewma'
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    def to_dict(self) -> Dict:
        return {'model_type': self.model_type, 'params': self.params}


@dataclass
class ConstraintConfig:
    """Configuration for constraint estimation (φ_C)"""
    model_type: str = 'fixed'
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
    
    def to_dict(self) -> Dict:
        return {'model_type': self.model_type, 'params': self.params}


@dataclass
class StabilityConfig:
    """Configuration for stability monitoring"""
    w_S: float = 1.0  # Weight for state variation
    w_I: float = 1.0  # Weight for memory error
    w_C: float = 1.0  # Weight for constraint margin
    epsilon_stable: float = 0.1
    epsilon_warning: float = 0.5
    epsilon_critical: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'w_S': self.w_S,
            'w_I': self.w_I,
            'w_C': self.w_C,
            'epsilon_stable': self.epsilon_stable,
            'epsilon_warning': self.epsilon_warning,
            'epsilon_critical': self.epsilon_critical
        }


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    dt: float = 1.0
    derivative_method: str = 'finite_difference'
    history_length: int = 100
    
    def to_dict(self) -> Dict:
        return {
            'dt': self.dt,
            'derivative_method': self.derivative_method,
            'history_length': self.history_length
        }


class ObserverConfiguration:
    """
    Complete configuration for diagnostic observer
    
    This class manages all parameters needed for reproducible experiments.
    Configurations can be:
    - Loaded from files (JSON/YAML)
    - Saved for reproducibility
    - Locked to prevent modification
    - Validated for correctness
    - Hashed for versioning
    """
    
    def __init__(self,
                 feature: Optional[FeatureConfig] = None,
                 memory: Optional[MemoryConfig] = None,
                 constraint: Optional[ConstraintConfig] = None,
                 stability: Optional[StabilityConfig] = None,
                 pipeline: Optional[PipelineConfig] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize configuration
        
        Args:
            feature: Feature extraction configuration
            memory: Memory model configuration
            constraint: Constraint estimation configuration
            stability: Stability monitoring configuration
            pipeline: Pipeline execution configuration
            metadata: Additional metadata (description, author, etc.)
        """
        self.feature = feature or FeatureConfig()
        self.memory = memory or MemoryConfig()
        self.constraint = constraint or ConstraintConfig()
        self.stability = stability or StabilityConfig()
        self.pipeline = pipeline or PipelineConfig()
        self.metadata = metadata or {}
        
        # Configuration state
        self._locked = False
        self._created_at = datetime.now().isoformat()
        self._config_hash = None
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        config_dict = {
            'feature': self.feature.to_dict(),
            'memory': self.memory.to_dict(),
            'constraint': self.constraint.to_dict(),
            'stability': self.stability.to_dict(),
            'pipeline': self.pipeline.to_dict(),
            'metadata': self.metadata,
            '_created_at': self._created_at,
            '_locked': self._locked
        }
        
        if self._config_hash:
            config_dict['_config_hash'] = self._config_hash
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ObserverConfiguration':
        """Create configuration from dictionary"""
        config = cls(
            feature=FeatureConfig(**config_dict.get('feature', {})),
            memory=MemoryConfig(**config_dict.get('memory', {})),
            constraint=ConstraintConfig(**config_dict.get('constraint', {})),
            stability=StabilityConfig(**config_dict.get('stability', {})),
            pipeline=PipelineConfig(**config_dict.get('pipeline', {})),
            metadata=config_dict.get('metadata', {})
        )
        
        config._locked = config_dict.get('_locked', False)
        config._created_at = config_dict.get('_created_at', datetime.now().isoformat())
        config._config_hash = config_dict.get('_config_hash')
        
        return config
    
    def save(self, filepath: str, format: str = 'json'):
        """
        Save configuration to file
        
        Args:
            filepath: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        config_dict = self.to_dict()
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ObserverConfiguration':
        """
        Load configuration from file
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            ObserverConfiguration instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Determine format from extension
        if filepath.suffix.lower() in ['.json']:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        config = cls.from_dict(config_dict)
        print(f"Configuration loaded from: {filepath}")
        
        return config
    
    def lock(self):
        """
        Lock configuration to prevent modifications
        
        Locked configurations cannot be modified and are suitable for
        production runs or published experiments.
        """
        if self._locked:
            print("Configuration is already locked")
            return
        
        self._locked = True
        self._config_hash = self.compute_hash()
        print(f"Configuration locked with hash: {self._config_hash[:16]}...")
    
    def unlock(self):
        """Unlock configuration to allow modifications"""
        self._locked = False
        self._config_hash = None
        print("Configuration unlocked")
    
    def is_locked(self) -> bool:
        """Check if configuration is locked"""
        return self._locked
    
    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of configuration
        
        This hash serves as a version identifier for reproducibility.
        
        Returns:
            Hexadecimal hash string
        """
        # Create a canonical representation (sorted keys)
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration parameters
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Validate stability weights
        if self.stability.w_S < 0 or self.stability.w_I < 0 or self.stability.w_C < 0:
            errors.append("Stability weights must be non-negative")
        
        # Validate thresholds
        if not (0 < self.stability.epsilon_stable < 
                self.stability.epsilon_warning < 
                self.stability.epsilon_critical):
            errors.append("Thresholds must satisfy: 0 < stable < warning < critical")
        
        # Validate pipeline parameters
        if self.pipeline.dt <= 0:
            errors.append("Time step dt must be positive")
        
        if self.pipeline.history_length < 1:
            errors.append("History length must be at least 1")
        
        valid_methods = ['finite_difference', 'backward', 'central']
        if self.pipeline.derivative_method not in valid_methods:
            errors.append(f"Derivative method must be one of {valid_methods}")
        
        # Validate feature map
        valid_features = ['identity', 'norm', 'synchrony', 'phase_coherence', 
                         'moving_average', 'variance']
        if self.feature.map_type not in valid_features:
            errors.append(f"Unknown feature map: {self.feature.map_type}")
        
        # Validate memory model
        valid_memory = ['ewma', 'integral', 'hysteresis', 
                       'windowed_average', 'adaptive_ewma']
        if self.memory.model_type not in valid_memory:
            errors.append(f"Unknown memory model: {self.memory.model_type}")
        
        # Validate constraint model
        valid_constraints = ['fixed', 'percentile', 'rolling_max', 
                           'ewma', 'control_barrier']
        if self.constraint.model_type not in valid_constraints:
            errors.append(f"Unknown constraint model: {self.constraint.model_type}")
        
        return len(errors) == 0, errors
    
    def __setattr__(self, name, value):
        """Override to prevent modification of locked configurations"""
        if hasattr(self, '_locked') and self._locked:
            if name not in ['_locked', '_config_hash']:
                raise RuntimeError(
                    f"Cannot modify locked configuration. "
                    f"Call unlock() first to allow modifications."
                )
        super().__setattr__(name, value)
    
    def __repr__(self) -> str:
        status = "LOCKED" if self._locked else "UNLOCKED"
        hash_str = f", hash={self._config_hash[:16]}..." if self._config_hash else ""
        return (f"ObserverConfiguration(status={status}{hash_str}, "
                f"created={self._created_at})")


def create_default_config() -> ObserverConfiguration:
    """Create a default configuration with reasonable parameters"""
    return ObserverConfiguration(
        feature=FeatureConfig(
            map_type='identity',
            params={}
        ),
        memory=MemoryConfig(
            model_type='ewma',
            params={'alpha': 0.2, 'initial_value': 0.0}
        ),
        constraint=ConstraintConfig(
            model_type='percentile',
            params={'percentile': 95, 'min_history': 20, 'default_value': 1.0}
        ),
        stability=StabilityConfig(
            w_S=1.0,
            w_I=0.5,
            w_C=2.0,
            epsilon_stable=0.1,
            epsilon_warning=0.5,
            epsilon_critical=1.0
        ),
        pipeline=PipelineConfig(
            dt=0.1,
            derivative_method='finite_difference',
            history_length=100
        ),
        metadata={
            'description': 'Default ΔΦ diagnostic configuration',
            'author': 'ΔΦ Framework',
            'version': '1.0'
        }
    )


if __name__ == "__main__":
    # Example usage
    print("Configuration Management Example")
    print("=" * 60)
    
    # Create default configuration
    config = create_default_config()
    print(f"\nCreated: {config}")
    
    # Validate
    is_valid, errors = config.validate()
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Compute hash
    hash_value = config.compute_hash()
    print(f"\nConfiguration hash: {hash_value[:32]}...")
    
    # Lock configuration
    print("\n" + "-" * 60)
    config.lock()
    print(f"Configuration state: {config}")
    
    # Try to modify (should fail)
    print("\nAttempting to modify locked configuration...")
    try:
        config.stability.w_S = 2.0
        print("  ERROR: Modification succeeded (should have failed!)")
    except RuntimeError as e:
        print(f"  ✓ Modification prevented: {str(e)[:50]}...")
    
    # Unlock and modify
    print("\n" + "-" * 60)
    config.unlock()
    config.stability.w_S = 2.0
    print(f"Modified w_S to: {config.stability.w_S}")
    
    # Save configuration
    print("\n" + "-" * 60)
    config.save('example_config.json', format='json')
    config.save('example_config.yaml', format='yaml')
    
    # Load configuration
    print("\n" + "-" * 60)
    loaded_config = ObserverConfiguration.load('example_config.json')
    print(f"Loaded: {loaded_config}")
    print(f"w_S value: {loaded_config.stability.w_S}")
    
    # Compare hashes
    print("\n" + "-" * 60)
    print("Hash comparison:")
    print(f"  Original: {config.compute_hash()[:32]}...")
    print(f"  Loaded:   {loaded_config.compute_hash()[:32]}...")
    print(f"  Match: {config.compute_hash() == loaded_config.compute_hash()}")
    
    # Display full configuration
    print("\n" + "=" * 60)
    print("Full Configuration Dictionary:")
    print(json.dumps(config.to_dict(), indent=2))

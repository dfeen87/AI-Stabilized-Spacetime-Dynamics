"""
Deterministic Replay Tools

This module provides tools for recording, saving, and replaying diagnostic
sessions with bit-perfect reproducibility. This is essential for:
- Verifying deterministic behavior
- Auditing diagnostic decisions
- Comparing algorithm changes
- Publishing reproducible results

All recorded data includes:
- Input signals y(t)
- Configuration parameters
- Intermediate feature values
- ΔΦ outputs
- Stability scores
- Regime flags
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class ReplayFrame:
    """
    Single frame in a replay recording
    
    Contains all information needed to verify a single time step.
    """
    timestamp: float
    step: int
    
    # Input
    y: Any  # Input signal (scalar or array)
    
    # Features
    phi_S: float
    phi_I: float
    phi_C: float
    
    # ΔΦ operator
    phi_S_dot: float
    e_I: float
    e_C: float
    
    # Stability
    J: float
    J_S: float
    J_I: float
    J_C: float
    regime: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'step': self.step,
            'y': self.y.tolist() if isinstance(self.y, np.ndarray) else self.y,
            'phi_S': self.phi_S,
            'phi_I': self.phi_I,
            'phi_C': self.phi_C,
            'phi_S_dot': self.phi_S_dot,
            'e_I': self.e_I,
            'e_C': self.e_C,
            'J': self.J,
            'J_S': self.J_S,
            'J_I': self.J_I,
            'J_C': self.J_C,
            'regime': self.regime
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReplayFrame':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_observer_output(cls, output, step: int) -> 'ReplayFrame':
        """Create from ObserverOutput object"""
        return cls(
            timestamp=output.timestamp,
            step=step,
            y=output.y,
            phi_S=output.phi_S,
            phi_I=output.phi_I,
            phi_C=output.phi_C,
            phi_S_dot=output.delta_phi.phi_S_dot,
            e_I=output.delta_phi.e_I,
            e_C=output.delta_phi.e_C,
            J=output.score.J,
            J_S=output.score.J_S,
            J_I=output.score.J_I,
            J_C=output.score.J_C,
            regime=output.score.regime.value
        )


class ReplayRecording:
    """
    Complete recording of a diagnostic session
    
    Contains:
    - Configuration used
    - All frames (input/output at each time step)
    - Metadata (timestamps, hashes, etc.)
    """
    
    def __init__(self, config_dict: Dict, metadata: Optional[Dict] = None):
        """
        Initialize recording
        
        Args:
            config_dict: Configuration dictionary
            metadata: Additional metadata (description, author, etc.)
        """
        self.config = config_dict
        self.metadata = metadata or {}
        self.frames: List[ReplayFrame] = []
        
        self.created_at = datetime.now().isoformat()
        self.recording_hash = None
    
    def add_frame(self, frame: ReplayFrame):
        """Add a frame to the recording"""
        self.frames.append(frame)
    
    def finalize(self):
        """
        Finalize recording and compute hash
        
        Call this after all frames are added to seal the recording.
        """
        self.recording_hash = self._compute_hash()
        print(f"Recording finalized: {len(self.frames)} frames")
        print(f"Recording hash: {self.recording_hash[:32]}...")
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of recording"""
        # Hash configuration + all frame data
        data_str = json.dumps({
            'config': self.config,
            'frames': [f.to_dict() for f in self.frames]
        }, sort_keys=True)
        
        hash_obj = hashlib.sha256(data_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def save(self, filepath: str, format: str = 'json', compress: bool = False):
        """
        Save recording to file
        
        Args:
            filepath: Path to save recording
            format: File format ('json' or 'pickle')
            compress: Whether to compress (for pickle only)
        """
        filepath = Path(filepath)
        
        recording_dict = {
            'config': self.config,
            'metadata': self.metadata,
            'frames': [f.to_dict() for f in self.frames],
            'created_at': self.created_at,
            'recording_hash': self.recording_hash,
            'num_frames': len(self.frames)
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(recording_dict, f, indent=2)
        elif format.lower() == 'pickle':
            import gzip
            if compress:
                with gzip.open(str(filepath) + '.gz', 'wb') as f:
                    pickle.dump(recording_dict, f)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(recording_dict, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Recording saved to: {filepath}")
        print(f"  Frames: {len(self.frames)}")
        print(f"  Hash: {self.recording_hash[:32]}..." if self.recording_hash else "  (not finalized)")
    
    @classmethod
    def load(cls, filepath: str) -> 'ReplayRecording':
        """
        Load recording from file
        
        Args:
            filepath: Path to recording file
        
        Returns:
            ReplayRecording instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            # Try with .gz extension
            if Path(str(filepath) + '.gz').exists():
                filepath = Path(str(filepath) + '.gz')
            else:
                raise FileNotFoundError(f"Recording file not found: {filepath}")
        
        # Determine format
        if str(filepath).endswith('.gz'):
            import gzip
            with gzip.open(filepath, 'rb') as f:
                recording_dict = pickle.load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                recording_dict = json.load(f)
        else:
            # Assume pickle
            with open(filepath, 'rb') as f:
                recording_dict = pickle.load(f)
        
        # Reconstruct recording
        recording = cls(
            config_dict=recording_dict['config'],
            metadata=recording_dict.get('metadata', {})
        )
        
        recording.frames = [
            ReplayFrame.from_dict(f) for f in recording_dict['frames']
        ]
        recording.created_at = recording_dict.get('created_at', 'unknown')
        recording.recording_hash = recording_dict.get('recording_hash')
        
        print(f"Recording loaded from: {filepath}")
        print(f"  Frames: {len(recording.frames)}")
        print(f"  Created: {recording.created_at}")
        
        return recording
    
    def verify_hash(self) -> bool:
        """
        Verify that current hash matches stored hash
        
        Returns:
            True if hashes match (recording unchanged)
        """
        if self.recording_hash is None:
            print("Warning: Recording not finalized (no hash)")
            return False
        
        current_hash = self._compute_hash()
        matches = current_hash == self.recording_hash
        
        if matches:
            print("✓ Hash verification passed")
        else:
            print("✗ Hash verification FAILED")
            print(f"  Expected: {self.recording_hash[:32]}...")
            print(f"  Got:      {current_hash[:32]}...")
        
        return matches
    
    def get_frame(self, step: int) -> Optional[ReplayFrame]:
        """Get frame by step number"""
        if 0 <= step < len(self.frames):
            return self.frames[step]
        return None
    
    def get_summary(self) -> Dict:
        """Get summary statistics of recording"""
        if not self.frames:
            return {}
        
        J_values = [f.J for f in self.frames]
        regimes = [f.regime for f in self.frames]
        
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'num_frames': len(self.frames),
            'duration': self.frames[-1].timestamp - self.frames[0].timestamp,
            'J_statistics': {
                'mean': np.mean(J_values),
                'std': np.std(J_values),
                'min': np.min(J_values),
                'max': np.max(J_values)
            },
            'regime_distribution': regime_counts,
            'created_at': self.created_at,
            'hash': self.recording_hash[:32] + '...' if self.recording_hash else None
        }


class ReplayValidator:
    """
    Validator for comparing recordings and verifying reproducibility
    """
    
    @staticmethod
    def compare_recordings(recording1: ReplayRecording, 
                          recording2: ReplayRecording,
                          tolerance: float = 1e-10) -> Dict:
        """
        Compare two recordings for reproducibility
        
        Args:
            recording1: First recording
            recording2: Second recording
            tolerance: Numerical tolerance for floating-point comparison
        
        Returns:
            Dictionary with comparison results
        """
        results = {
            'identical': True,
            'differences': [],
            'config_match': False,
            'num_frames_match': False,
            'hash_match': False
        }
        
        # Compare configurations
        if recording1.config == recording2.config:
            results['config_match'] = True
        else:
            results['identical'] = False
            results['differences'].append('Configurations differ')
        
        # Compare number of frames
        if len(recording1.frames) == len(recording2.frames):
            results['num_frames_match'] = True
        else:
            results['identical'] = False
            results['differences'].append(
                f"Frame count differs: {len(recording1.frames)} vs {len(recording2.frames)}"
            )
            return results  # Can't compare frames
        
        # Compare hashes
        if recording1.recording_hash and recording2.recording_hash:
            if recording1.recording_hash == recording2.recording_hash:
                results['hash_match'] = True
            else:
                results['identical'] = False
                results['differences'].append('Recording hashes differ')
        
        # Compare frame-by-frame
        max_diff = 0.0
        for i, (f1, f2) in enumerate(zip(recording1.frames, recording2.frames)):
            # Compare all numeric fields
            fields = ['phi_S', 'phi_I', 'phi_C', 'phi_S_dot', 'e_I', 'e_C',
                     'J', 'J_S', 'J_I', 'J_C']
            
            for field in fields:
                v1 = getattr(f1, field)
                v2 = getattr(f2, field)
                diff = abs(v1 - v2)
                max_diff = max(max_diff, diff)
                
                if diff > tolerance:
                    results['identical'] = False
                    results['differences'].append(
                        f"Frame {i}, {field}: diff={diff:.2e} "
                        f"(v1={v1:.6f}, v2={v2:.6f})"
                    )
            
            # Compare regime
            if f1.regime != f2.regime:
                results['identical'] = False
                results['differences'].append(
                    f"Frame {i}, regime: {f1.regime} vs {f2.regime}"
                )
        
        results['max_numeric_difference'] = max_diff
        results['reproducible'] = max_diff < tolerance
        
        return results
    
    @staticmethod
    def validate_determinism(observer, y_signal: np.ndarray, 
                           num_runs: int = 3) -> Dict:
        """
        Validate that observer produces identical results across runs
        
        Args:
            observer: DiagnosticObserver instance
            y_signal: Input signal to process
            num_runs: Number of runs to compare
        
        Returns:
            Dictionary with validation results
        """
        recordings = []
        
        print(f"Running determinism validation ({num_runs} runs)...")
        
        for run in range(num_runs):
            # Reset observer
            observer.reset()
            
            # Create recording
            recording = ReplayRecording(
                config_dict=observer.get_config(),
                metadata={'run': run, 'validation': 'determinism'}
            )
            
            # Process signal
            for step, y in enumerate(y_signal):
                output = observer.process(y)
                frame = ReplayFrame.from_observer_output(output, step)
                recording.add_frame(frame)
            
            recording.finalize()
            recordings.append(recording)
            
            print(f"  Run {run+1}: hash={recording.recording_hash[:16]}...")
        
        # Compare all pairs
        print("\nComparing recordings...")
        all_match = True
        
        for i in range(len(recordings) - 1):
            comparison = ReplayValidator.compare_recordings(
                recordings[i], recordings[i+1]
            )
            
            if not comparison['identical']:
                all_match = False
                print(f"  ✗ Run {i} vs Run {i+1}: DIFFERENCES FOUND")
                for diff in comparison['differences'][:5]:  # Show first 5
                    print(f"    - {diff}")
            else:
                print(f"  ✓ Run {i} vs Run {i+1}: identical")
        
        return {
            'deterministic': all_match,
            'num_runs': num_runs,
            'hashes': [r.recording_hash for r in recordings],
            'all_hashes_match': len(set(r.recording_hash for r in recordings)) == 1
        }


if __name__ == "__main__":
    # Example usage
    print("Replay and Validation Example")
    print("=" * 60)
    
    # Create a mock recording
    from pipeline.config import create_default_config
    
    config = create_default_config()
    recording = ReplayRecording(
        config_dict=config.to_dict(),
        metadata={'description': 'Example recording', 'author': 'Test'}
    )
    
    # Add some synthetic frames
    print("\nCreating synthetic recording...")
    for i in range(20):
        t = i * 0.1
        y = np.sin(2 * np.pi * t)
        
        frame = ReplayFrame(
            timestamp=t,
            step=i,
            y=y,
            phi_S=y,
            phi_I=0.8 * y,
            phi_C=1.0,
            phi_S_dot=np.cos(2 * np.pi * t) * 2 * np.pi * 0.1,
            e_I=0.2 * y,
            e_C=1.0 - y,
            J=0.1 + 0.05 * abs(y),
            J_S=0.05,
            J_I=0.03,
            J_C=0.02,
            regime='stable' if abs(y) < 0.5 else 'warning'
        )
        recording.add_frame(frame)
    
    recording.finalize()
    
    # Display summary
    print("\nRecording summary:")
    summary = recording.get_summary()
    print(json.dumps(summary, indent=2))
    
    # Save and load
    print("\n" + "-" * 60)
    recording.save('example_recording.json', format='json')
    recording.save('example_recording.pkl', format='pickle')
    
    # Load and verify
    print("\n" + "-" * 60)
    loaded = ReplayRecording.load('example_recording.json')
    loaded.verify_hash()
    
    # Compare recordings
    print("\n" + "-" * 60)
    print("Comparing original and loaded recordings...")
    comparison = ReplayValidator.compare_recordings(recording, loaded)
    print(f"  Identical: {comparison['identical']}")
    print(f"  Config match: {comparison['config_match']}")
    print(f"  Hash match: {comparison['hash_match']}")
    print(f"  Max difference: {comparison['max_numeric_difference']:.2e}")
    
    if comparison['differences']:
        print(f"  Differences found: {len(comparison['differences'])}")

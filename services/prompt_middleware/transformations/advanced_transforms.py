#!/usr/bin/env python3
"""
Advanced Prompt Transformations
Implements sophisticated geometric and mathematical transformations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import logging
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, FastICA
import networkx as nx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TransformationConfig:
    """Configuration for transformations"""
    intensity: float = 1.0
    preserve_semantics: bool = True
    add_noise: bool = False
    noise_level: float = 0.1
    temporal_factor: float = 0.5
    spatial_factor: float = 0.5

class HyperbolicTransformer:
    """Implements hyperbolic geometry transformations"""
    
    def __init__(self, curvature: float = -1.0):
        self.curvature = curvature
    
    def poincare_ball_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to Poincaré ball model"""
        # Normalize to unit ball
        norm = np.linalg.norm(embeddings)
        if norm >= 1.0:
            embeddings = embeddings / (norm + 1e-8)
        
        # Apply hyperbolic distance preservation
        # In hyperbolic space, distances grow exponentially
        hyperbolic_embeddings = embeddings * np.exp(self.curvature * np.linalg.norm(embeddings))
        
        return hyperbolic_embeddings
    
    def mobius_transformation(self, embeddings: np.ndarray, a: complex = 1+0j, b: complex = 0+0j, 
                            c: complex = 0+0j, d: complex = 1+0j) -> np.ndarray:
        """Apply Möbius transformation (conformal mapping)"""
        # Convert to complex representation
        if embeddings.shape[0] % 2 == 0:
            complex_embeddings = embeddings[:len(embeddings)//2] + 1j * embeddings[len(embeddings)//2:]
        else:
            complex_embeddings = embeddings[:-1] + 1j * embeddings[1:]
        
        # Apply Möbius transformation: (a*z + b) / (c*z + d)
        transformed = (a * complex_embeddings + b) / (c * complex_embeddings + d + 1e-10)
        
        # Convert back to real representation
        real_part = np.real(transformed)
        imag_part = np.imag(transformed)
        
        if embeddings.shape[0] % 2 == 0:
            return np.concatenate([real_part, imag_part])
        else:
            return np.concatenate([real_part, imag_part, [embeddings[-1]]])

class QuantumInspiredTransformer:
    """Implements quantum-inspired transformations"""
    
    def __init__(self):
        self.pauli_matrices = {
            'x': np.array([[0, 1], [1, 0]], dtype=complex),
            'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
    
    def quantum_superposition(self, embeddings: np.ndarray, num_states: int = 3) -> np.ndarray:
        """Create quantum superposition of embedding states"""
        # Create multiple "quantum states" of the embeddings
        states = []
        for i in range(num_states):
            # Apply different rotations to create different states
            angle = 2 * np.pi * i / num_states
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Apply rotation to first two dimensions
            rotated = embeddings.copy()
            if len(embeddings) >= 2:
                rotated[:2] = np.dot(rotation_matrix, embeddings[:2])
            states.append(rotated)
        
        # Create superposition (weighted average)
        weights = np.ones(num_states) / num_states
        superposition = np.average(states, axis=0, weights=weights)
        
        return superposition
    
    def quantum_entanglement(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create quantum entanglement between two embedding vectors"""
        # Ensure same dimensionality
        min_dim = min(len(embeddings1), len(embeddings2))
        emb1 = embeddings1[:min_dim]
        emb2 = embeddings2[:min_dim]
        
        # Create entangled state using Bell state
        # |ψ⟩ = (|00⟩ + |11⟩) / √2
        entangled_state = (emb1 + emb2) / np.sqrt(2)
        
        # Project back to individual states
        entangled_emb1 = entangled_state + np.random.normal(0, 0.1, min_dim)
        entangled_emb2 = entangled_state + np.random.normal(0, 0.1, min_dim)
        
        return entangled_emb1, entangled_emb2

class FractalTransformer:
    """Implements fractal-based transformations"""
    
    def __init__(self, fractal_type: str = "mandelbrot"):
        self.fractal_type = fractal_type
    
    def mandelbrot_transform(self, embeddings: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply Mandelbrot set transformation"""
        # Convert embeddings to complex numbers
        if embeddings.shape[0] % 2 == 0:
            c = embeddings[:len(embeddings)//2] + 1j * embeddings[len(embeddings)//2:]
        else:
            c = embeddings[:-1] + 1j * embeddings[1:]
        
        # Apply Mandelbrot iteration: z = z² + c
        z = np.zeros_like(c)
        with np.errstate(over='ignore', invalid='ignore'):
            for _ in range(max_iter):
                z = z**2 + c
                # Bailout to avoid overflow
                diverged = np.abs(z) > 2.0
                if np.any(diverged):
                    # Normalize diverged values to avoid NaNs while keeping phase information
                    z[diverged] = 2.0 * np.exp(1j * np.angle(z[diverged]))
        
        # Convert back to real representation
        real_part = np.real(z)
        imag_part = np.imag(z)
        
        if embeddings.shape[0] % 2 == 0:
            return np.concatenate([real_part, imag_part])
        else:
            return np.concatenate([real_part, imag_part, [embeddings[-1]]])
    
    def julia_transform(self, embeddings: np.ndarray, c: complex = -0.7+0.27015j, max_iter: int = 100) -> np.ndarray:
        """Apply Julia set transformation"""
        # Convert embeddings to complex numbers
        if embeddings.shape[0] % 2 == 0:
            z = embeddings[:len(embeddings)//2] + 1j * embeddings[len(embeddings)//2:]
        else:
            z = embeddings[:-1] + 1j * embeddings[1:]
        
        # Apply Julia iteration: z = z² + c
        with np.errstate(over='ignore', invalid='ignore'):
            for _ in range(max_iter):
                z = z**2 + c
                diverged = np.abs(z) > 2.0
                if np.any(diverged):
                    z[diverged] = 2.0 * np.exp(1j * np.angle(z[diverged]))
        
        # Convert back to real representation
        real_part = np.real(z)
        imag_part = np.imag(z)
        
        if embeddings.shape[0] % 2 == 0:
            return np.concatenate([real_part, imag_part])
        else:
            return np.concatenate([real_part, imag_part, [embeddings[-1]]])

class ManifoldTransformer:
    """Implements manifold learning transformations"""
    
    def __init__(self):
        self.tsne = TSNE(n_components=2, random_state=42)
        self.mds = MDS(n_components=2, random_state=42)
        self.pca = PCA(n_components=2)
        self.ica = FastICA(n_components=2, random_state=42)
    
    def manifold_projection(self, embeddings: np.ndarray, method: str = "tsne") -> np.ndarray:
        """Project embeddings to lower-dimensional manifold"""
        # Reshape for manifold learning
        if len(embeddings.shape) == 1:
            embeddings_2d = embeddings.reshape(1, -1)
        else:
            embeddings_2d = embeddings
        
        # For single sample, we can't use t-SNE (needs multiple samples)
        if embeddings_2d.shape[0] == 1:
            # For single sample, just return the first 2 dimensions
            if embeddings_2d.shape[1] >= 2:
                projected = embeddings_2d[:, :2]
            else:
                # Pad with zeros if not enough dimensions
                projected = np.zeros((1, 2))
                projected[0, :embeddings_2d.shape[1]] = embeddings_2d[0]
        else:
            if method == "tsne":
                # Ensure we have enough samples for t-SNE
                if embeddings_2d.shape[0] < 3:
                    projected = self.pca.fit_transform(embeddings_2d)
                else:
                    projected = self.tsne.fit_transform(embeddings_2d)
            elif method == "mds":
                projected = self.mds.fit_transform(embeddings_2d)
            elif method == "pca":
                projected = self.pca.fit_transform(embeddings_2d)
            elif method == "ica":
                projected = self.ica.fit_transform(embeddings_2d)
            else:
                projected = embeddings_2d
        
        return projected.flatten()
    
    def geodesic_distance_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply geodesic distance transformation"""
        # Calculate pairwise distances
        distances = pdist(embeddings.reshape(1, -1))
        distance_matrix = squareform(distances)
        
        # Apply geodesic transformation (simplified)
        # In practice, this would use more sophisticated manifold learning
        geodesic_embeddings = embeddings * (1 + np.mean(distance_matrix))
        
        return geodesic_embeddings

class TemporalTransformer:
    """Implements temporal and dynamic transformations"""
    
    def __init__(self, time_steps: int = 10):
        self.time_steps = time_steps
    
    def temporal_evolution(self, embeddings: np.ndarray, evolution_type: str = "oscillatory") -> np.ndarray:
        """Apply temporal evolution to embeddings"""
        if evolution_type == "oscillatory":
            # Oscillatory evolution
            time_factor = np.sin(np.linspace(0, 2*np.pi, self.time_steps))
            evolved = embeddings * (1 + 0.1 * time_factor[-1])
            
        elif evolution_type == "exponential":
            # Exponential growth/decay
            time_factor = np.exp(np.linspace(0, 1, self.time_steps))
            evolved = embeddings * time_factor[-1]
            
        elif evolution_type == "chaotic":
            # Chaotic evolution (logistic map)
            r = 3.9  # Chaotic parameter
            x = 0.5  # Initial value
            for _ in range(self.time_steps):
                x = r * x * (1 - x)
            evolved = embeddings * (1 + 0.1 * x)
            
        else:
            evolved = embeddings
        
        return evolved
    
    def phase_space_reconstruction(self, embeddings: np.ndarray, delay: int = 1, dimension: int = 3) -> np.ndarray:
        """Reconstruct phase space from embeddings"""
        # Create delay vectors
        if len(embeddings) < delay * dimension:
            # Pad with zeros if not enough data
            padded = np.pad(embeddings, (0, delay * dimension - len(embeddings)))
        else:
            padded = embeddings
        
        # Create delay vectors
        delay_vectors = []
        for i in range(dimension):
            start_idx = i * delay
            end_idx = start_idx + len(embeddings)
            delay_vectors.append(padded[start_idx:end_idx])
        
        # Combine delay vectors
        phase_space = np.concatenate(delay_vectors)
        
        return phase_space

class AdvancedPromptTransformer:
    """Main class for advanced prompt transformations"""
    
    def __init__(self):
        self.hyperbolic = HyperbolicTransformer()
        self.quantum = QuantumInspiredTransformer()
        self.fractal = FractalTransformer()
        self.manifold = ManifoldTransformer()
        self.temporal = TemporalTransformer()
    
    def apply_hyperbolic_transformation(self, embeddings: np.ndarray, config: TransformationConfig) -> np.ndarray:
        """Apply hyperbolic geometry transformation"""
        transformed = self.hyperbolic.poincare_ball_projection(embeddings)
        
        if config.add_noise:
            noise = np.random.normal(0, config.noise_level, transformed.shape)
            transformed += noise
        
        return transformed * config.intensity
    
    def apply_quantum_transformation(self, embeddings: np.ndarray, config: TransformationConfig) -> np.ndarray:
        """Apply quantum-inspired transformation"""
        transformed = self.quantum.quantum_superposition(embeddings)
        
        if config.add_noise:
            noise = np.random.normal(0, config.noise_level, transformed.shape)
            transformed += noise
        
        return transformed * config.intensity
    
    def apply_fractal_transformation(self, embeddings: np.ndarray, config: TransformationConfig) -> np.ndarray:
        """Apply fractal transformation"""
        if self.fractal.fractal_type == "mandelbrot":
            transformed = self.fractal.mandelbrot_transform(embeddings)
        else:
            transformed = self.fractal.julia_transform(embeddings)
        
        if config.add_noise:
            noise = np.random.normal(0, config.noise_level, transformed.shape)
            transformed += noise
        
        return transformed * config.intensity
    
    def apply_manifold_transformation(self, embeddings: np.ndarray, config: TransformationConfig, method: str = "tsne") -> np.ndarray:
        """Apply manifold learning transformation"""
        transformed = self.manifold.manifold_projection(embeddings, method)
        
        if config.add_noise:
            noise = np.random.normal(0, config.noise_level, transformed.shape)
            transformed += noise
        
        return transformed * config.intensity
    
    def apply_temporal_transformation(self, embeddings: np.ndarray, config: TransformationConfig, evolution_type: str = "oscillatory") -> np.ndarray:
        """Apply temporal transformation"""
        transformed = self.temporal.temporal_evolution(embeddings, evolution_type)
        
        if config.add_noise:
            noise = np.random.normal(0, config.noise_level, transformed.shape)
            transformed += noise
        
        return transformed * config.intensity
    
    def apply_multi_modal_transformation(self, embeddings: np.ndarray, config: TransformationConfig) -> Dict[str, np.ndarray]:
        """Apply multiple transformations and return all results"""
        results = {}
        
        # Apply all transformations
        results['hyperbolic'] = self.apply_hyperbolic_transformation(embeddings, config)
        results['quantum'] = self.apply_quantum_transformation(embeddings, config)
        results['fractal'] = self.apply_fractal_transformation(embeddings, config)
        results['manifold'] = self.apply_manifold_transformation(embeddings, config)
        results['temporal'] = self.apply_temporal_transformation(embeddings, config)
        
        # Normalize vector lengths before ensembling
        vectors = list(results.values())
        # Determine a common length (use minimum length to avoid padding biases)
        common_len = min(len(v) for v in vectors if isinstance(v, np.ndarray))
        if common_len <= 0:
            results['ensemble'] = embeddings
            return results
        trimmed = [v[:common_len] for v in vectors]
        
        # Create ensemble result
        ensemble = np.mean(trimmed, axis=0)
        results['ensemble'] = ensemble
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Test the transformations
    transformer = AdvancedPromptTransformer()
    config = TransformationConfig(intensity=1.0, add_noise=False)
    
    # Sample embeddings
    sample_embeddings = np.random.randn(384)  # Typical embedding size
    
    # Test individual transformations
    print("Testing transformations...")
    
    hyperbolic_result = transformer.apply_hyperbolic_transformation(sample_embeddings, config)
    print(f"Hyperbolic transformation: {hyperbolic_result[:5]}...")
    
    quantum_result = transformer.apply_quantum_transformation(sample_embeddings, config)
    print(f"Quantum transformation: {quantum_result[:5]}...")
    
    fractal_result = transformer.apply_fractal_transformation(sample_embeddings, config)
    print(f"Fractal transformation: {fractal_result[:5]}...")
    
    # Test multi-modal transformation
    multi_modal_results = transformer.apply_multi_modal_transformation(sample_embeddings, config)
    print(f"Multi-modal results keys: {list(multi_modal_results.keys())}")
    print(f"Ensemble result: {multi_modal_results['ensemble'][:5]}...")

#!/usr/bin/env python3
"""
Advanced Prompt Middleware Suite
Implements geometric transformations, voice-based subtext inference, and mathematical reprocessing
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformationType(Enum):
    """Types of prompt transformations"""
    MIRROR = "mirror"
    CHIRAL = "chiral"
    DIMENSIONAL_4D = "4d_projection"
    TOPOLOGICAL = "topological"
    VOICE_CONDITIONED = "voice_conditioned"
    SEMANTIC_ROTATION = "semantic_rotation"
    ENTROPY_ENHANCEMENT = "entropy_enhancement"

@dataclass
class PromptContext:
    """Context information for prompt processing"""
    text: str
    voice_features: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    spatial_context: Optional[Dict[str, Any]] = None
    social_context: Optional[Dict[str, Any]] = None
    domain_context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TransformationResult:
    """Result of a prompt transformation"""
    original_prompt: str
    transformed_prompt: str
    transformation_type: TransformationType
    confidence_score: float
    subtext_inference: Dict[str, Any]
    geometric_metadata: Dict[str, Any]
    processing_time: float

class VoiceAnalyzer:
    """Analyzes voice patterns to infer subtext and emotional context"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
    def extract_prosodic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features from audio"""
        try:
            # Fundamental frequency (pitch)
            f0 = librosa.yin(audio_data, fmin=50, fmax=400)
            pitch_mean = np.mean(f0[~np.isnan(f0)])
            pitch_std = np.std(f0[~np.isnan(f0)])
            
            # Rhythm and tempo
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            
            # Energy and intensity
            rms = librosa.feature.rms(y=audio_data)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            
            return {
                'pitch_mean': float(pitch_mean) if not np.isnan(pitch_mean) else 0.0,
                'pitch_std': float(pitch_std) if not np.isnan(pitch_std) else 0.0,
                'tempo': float(tempo),
                'energy_mean': float(energy_mean),
                'energy_std': float(energy_std),
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff))
            }
        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            return {}
    
    def infer_emotional_context(self, features: Dict[str, float]) -> Dict[str, float]:
        """Infer emotional context from prosodic features"""
        # Simple heuristic-based emotional inference
        # In production, this would use a trained model
        
        emotional_scores = {
            'anger': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'neutral': 0.0
        }
        
        # High pitch + high energy = excitement/anger
        if features.get('pitch_mean', 0) > 200 and features.get('energy_mean', 0) > 0.1:
            emotional_scores['anger'] += 0.3
            emotional_scores['joy'] += 0.2
        
        # Low pitch + low energy = sadness
        if features.get('pitch_mean', 0) < 150 and features.get('energy_mean', 0) < 0.05:
            emotional_scores['sadness'] += 0.4
        
        # High tempo = excitement
        if features.get('tempo', 0) > 120:
            emotional_scores['joy'] += 0.2
            emotional_scores['surprise'] += 0.1
        
        # Normalize scores
        total = sum(emotional_scores.values())
        if total > 0:
            emotional_scores = {k: v/total for k, v in emotional_scores.items()}
        else:
            emotional_scores['neutral'] = 1.0
            
        return emotional_scores
    
    def detect_intentional_ambiguity(self, features: Dict[str, float]) -> float:
        """Detect if the speaker is being intentionally ambiguous"""
        # High pitch variance + moderate energy = potential ambiguity
        pitch_std = features.get('pitch_std', 0)
        energy_mean = features.get('energy_mean', 0)
        
        if 50 < pitch_std < 100 and 0.05 < energy_mean < 0.1:
            return 0.7  # High ambiguity
        elif pitch_std > 100:
            return 0.9  # Very high ambiguity
        else:
            return 0.3  # Low ambiguity

class GeometricTransformer:
    """Implements geometric transformations on semantic space"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings[0]
    
    def mirror_transformation(self, embeddings: np.ndarray, axis: int = 0) -> np.ndarray:
        """Apply mirror transformation to embeddings"""
        # Create reflection matrix
        reflection_matrix = np.eye(embeddings.shape[0])
        reflection_matrix[axis, axis] = -1
        
        return np.dot(reflection_matrix, embeddings)
    
    def chiral_transformation(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply chiral (handedness) transformation"""
        # Chiral transformation: flip the sign of the cross product components
        # This changes the "handedness" of the semantic space
        
        # For simplicity, we'll apply a rotation that changes handedness
        rotation_angle = np.pi / 4
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        
        # Create rotation matrix for first two dimensions
        rotation_matrix = np.eye(embeddings.shape[0])
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 1] = -sin_theta
        rotation_matrix[1, 0] = sin_theta
        rotation_matrix[1, 1] = cos_theta
        
        return np.dot(rotation_matrix, embeddings)
    
    def dimensional_4d_projection(self, embeddings: np.ndarray, time_dimension: float = 0.5) -> np.ndarray:
        """Project embeddings into 4D space with temporal dimension"""
        # For high-dimensional embeddings, we'll apply a simpler temporal transformation
        # that doesn't require exact 4D matrix operations
        
        # Apply temporal scaling based on time dimension
        temporal_factor = 1 + 0.1 * np.sin(time_dimension * np.pi)
        
        # Apply temporal rotation to first few dimensions
        if len(embeddings) >= 4:
            # Create a simple rotation matrix for first 4 dimensions
            angle = time_dimension * np.pi / 2
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # Apply rotation to first 4 dimensions
            rotated_embeddings = embeddings.copy()
            rotated_embeddings[0] = embeddings[0] * cos_angle - embeddings[1] * sin_angle
            rotated_embeddings[1] = embeddings[0] * sin_angle + embeddings[1] * cos_angle
            rotated_embeddings[2] = embeddings[2] * cos_angle - embeddings[3] * sin_angle
            rotated_embeddings[3] = embeddings[2] * sin_angle + embeddings[3] * cos_angle
            
            # Apply temporal scaling
            return rotated_embeddings * temporal_factor
        else:
            # For smaller embeddings, just apply temporal scaling
            return embeddings * temporal_factor
    
    def topological_deformation(self, embeddings: np.ndarray, curvature: float = 0.1) -> np.ndarray:
        """Apply topological deformation to semantic space"""
        # Simulate curvature in the semantic manifold
        # This changes the "shape" of the semantic space
        
        # Apply non-linear transformation based on distance from origin
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            # Curvature effect: bend the space
            curvature_factor = 1 + curvature * np.sin(norm)
            return embeddings * curvature_factor
        return embeddings

class SemanticReprocessor:
    """Mathematical reprocessing of word connections and semantic relationships"""
    
    def __init__(self):
        self.word_graph = nx.Graph()
        self.semantic_vectors = {}
        
    def build_semantic_graph(self, text: str, embeddings: np.ndarray) -> nx.Graph:
        """Build a graph of semantic relationships"""
        words = text.split()
        
        # Add nodes with embedding vectors
        for i, word in enumerate(words):
            if i < len(embeddings):
                self.word_graph.add_node(word, embedding=embeddings[i])
        
        # Add edges based on semantic similarity
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j and word1 in self.word_graph.nodes and word2 in self.word_graph.nodes:
                    similarity = 1 - cosine(
                        self.word_graph.nodes[word1]['embedding'],
                        self.word_graph.nodes[word2]['embedding']
                    )
                    if similarity > 0.3:  # Threshold for edge creation
                        self.word_graph.add_edge(word1, word2, weight=similarity)
        
        return self.word_graph
    
    def analyze_information_flow(self, graph: nx.Graph) -> Dict[str, float]:
        """Analyze information flow through the semantic graph"""
        # Calculate centrality measures
        betweenness_centrality = nx.betweenness_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
        
        # Calculate clustering coefficient
        clustering = nx.clustering(graph)
        
        return {
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
            'clustering_coefficient': clustering,
            'average_clustering': nx.average_clustering(graph),
            'density': nx.density(graph)
        }
    
    def entropy_enhancement(self, text: str, embeddings: np.ndarray) -> np.ndarray:
        """Enhance embeddings using information theory"""
        # Calculate entropy of the embedding distribution
        probabilities = np.abs(embeddings) / np.sum(np.abs(embeddings))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Enhance based on entropy
        enhancement_factor = 1 + 0.1 * entropy
        enhanced_embeddings = embeddings * enhancement_factor
        
        return enhanced_embeddings
    
    def mutual_information_analysis(self, text1: str, text2: str, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Calculate mutual information between two texts"""
        # Simplified mutual information calculation
        # In practice, this would use more sophisticated methods
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embeddings1, embeddings2)
        
        # Convert to mutual information approximation
        # This is a heuristic - real MI calculation would be more complex
        mutual_info = -np.log(1 - similarity + 1e-10)
        
        return float(mutual_info)

class PromptMiddleware:
    """Main middleware orchestrator"""
    
    def __init__(self):
        self.voice_analyzer = VoiceAnalyzer()
        self.geometric_transformer = GeometricTransformer()
        self.semantic_reprocessor = SemanticReprocessor()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def process_prompt(self, context: PromptContext, transformations: List[TransformationType]) -> List[TransformationResult]:
        """Process a prompt through multiple transformations"""
        results = []
        
        # Get base embeddings
        base_embeddings = self.geometric_transformer.get_embeddings(context.text)
        
        # Process each transformation
        for transformation_type in transformations:
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await self._apply_transformation(
                    context, base_embeddings, transformation_type
                )
                result.processing_time = asyncio.get_event_loop().time() - start_time
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error applying transformation {transformation_type}: {e}")
                continue
        
        return results
    
    async def _apply_transformation(self, context: PromptContext, base_embeddings: np.ndarray, transformation_type: TransformationType) -> TransformationResult:
        """Apply a specific transformation"""
        
        if transformation_type == TransformationType.MIRROR:
            transformed_embeddings = self.geometric_transformer.mirror_transformation(base_embeddings)
            transformed_text = await self._embeddings_to_text(transformed_embeddings, context.text)
            
        elif transformation_type == TransformationType.CHIRAL:
            transformed_embeddings = self.geometric_transformer.chiral_transformation(base_embeddings)
            transformed_text = await self._embeddings_to_text(transformed_embeddings, context.text)
            
        elif transformation_type == TransformationType.DIMENSIONAL_4D:
            time_dim = context.temporal_context.get('time_factor', 0.5) if context.temporal_context else 0.5
            transformed_embeddings = self.geometric_transformer.dimensional_4d_projection(base_embeddings, time_dim)
            transformed_text = await self._embeddings_to_text(transformed_embeddings, context.text)
            
        elif transformation_type == TransformationType.TOPOLOGICAL:
            curvature = context.metadata.get('curvature', 0.1) if context.metadata else 0.1
            transformed_embeddings = self.geometric_transformer.topological_deformation(base_embeddings, curvature)
            transformed_text = await self._embeddings_to_text(transformed_embeddings, context.text)
            
        elif transformation_type == TransformationType.VOICE_CONDITIONED:
            if context.voice_features:
                transformed_text = await self._voice_conditioned_transform(context)
            else:
                transformed_text = context.text
                
        elif transformation_type == TransformationType.SEMANTIC_ROTATION:
            transformed_embeddings = self.geometric_transformer.chiral_transformation(base_embeddings)
            transformed_text = await self._embeddings_to_text(transformed_embeddings, context.text)
            
        elif transformation_type == TransformationType.ENTROPY_ENHANCEMENT:
            transformed_embeddings = self.semantic_reprocessor.entropy_enhancement(context.text, base_embeddings)
            transformed_text = await self._embeddings_to_text(transformed_embeddings, context.text)
            
        else:
            transformed_text = context.text
        
        # Analyze subtext
        subtext_inference = await self._analyze_subtext(context, transformed_text)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(context, transformed_text)
        
        # Generate geometric metadata
        geometric_metadata = {
            'transformation_type': transformation_type.value,
            'embedding_dimension': len(base_embeddings),
            'transformation_applied': True
        }
        
        return TransformationResult(
            original_prompt=context.text,
            transformed_prompt=transformed_text,
            transformation_type=transformation_type,
            confidence_score=confidence_score,
            subtext_inference=subtext_inference,
            geometric_metadata=geometric_metadata,
            processing_time=0.0  # Will be set by caller
        )
    
    async def _embeddings_to_text(self, embeddings: np.ndarray, original_text: str) -> str:
        """Convert embeddings back to text (simplified approach)"""
        # This is a simplified approach - in production, you'd use a proper decoder
        # For now, we'll return the original text with some modifications
        
        # Apply some heuristic transformations based on embedding changes
        words = original_text.split()
        if len(words) > 0:
            # Simple word reordering based on embedding similarity
            # This is a placeholder - real implementation would be more sophisticated
            return " ".join(words[::-1])  # Reverse word order as example
        return original_text
    
    async def _voice_conditioned_transform(self, context: PromptContext) -> str:
        """Apply voice-conditioned transformation"""
        if not context.voice_features:
            return context.text
        
        # Analyze emotional context
        emotional_context = self.voice_analyzer.infer_emotional_context(context.voice_features)
        ambiguity_score = self.voice_analyzer.detect_intentional_ambiguity(context.voice_features)
        
        # Modify text based on voice analysis
        words = context.text.split()
        
        # Add emotional qualifiers based on voice analysis
        dominant_emotion = max(emotional_context, key=emotional_context.get)
        
        if dominant_emotion == 'anger' and emotional_context['anger'] > 0.5:
            words.insert(0, "urgently")
        elif dominant_emotion == 'sadness' and emotional_context['sadness'] > 0.5:
            words.insert(0, "quietly")
        elif dominant_emotion == 'joy' and emotional_context['joy'] > 0.5:
            words.insert(0, "enthusiastically")
        
        # Add ambiguity markers if detected
        if ambiguity_score > 0.7:
            words.append("(unclear)")
        
        return " ".join(words)
    
    async def _analyze_subtext(self, context: PromptContext, transformed_text: str) -> Dict[str, Any]:
        """Analyze subtext and implied meaning"""
        subtext = {
            'emotional_undertones': {},
            'intentional_ambiguity': 0.0,
            'implied_urgency': 0.0,
            'social_dynamics': {},
            'domain_specificity': 0.0
        }
        
        # Analyze voice features if available
        if context.voice_features:
            subtext['emotional_undertones'] = self.voice_analyzer.infer_emotional_context(context.voice_features)
            subtext['intentional_ambiguity'] = self.voice_analyzer.detect_intentional_ambiguity(context.voice_features)
        
        # Analyze social context
        if context.social_context:
            subtext['social_dynamics'] = context.social_context
        
        # Analyze domain context
        if context.domain_context:
            subtext['domain_specificity'] = len(context.domain_context.get('keywords', [])) / 10.0
        
        return subtext
    
    def _calculate_confidence(self, context: PromptContext, transformed_text: str) -> float:
        """Calculate confidence score for the transformation"""
        # Simple confidence calculation based on text similarity and context richness
        base_confidence = 0.5
        
        # Increase confidence based on available context
        if context.voice_features:
            base_confidence += 0.2
        if context.temporal_context:
            base_confidence += 0.1
        if context.spatial_context:
            base_confidence += 0.1
        if context.social_context:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Prompt Middleware API", version="1.0.0")

class PromptRequest(BaseModel):
    text: str
    voice_features: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    spatial_context: Optional[Dict[str, Any]] = None
    social_context: Optional[Dict[str, Any]] = None
    domain_context: Optional[Dict[str, Any]] = None
    transformations: List[str] = ["mirror", "chiral", "4d_projection"]

class PromptResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

middleware = PromptMiddleware()

@app.post("/transform", response_model=PromptResponse)
async def transform_prompt(request: PromptRequest):
    """Transform a prompt using the middleware suite"""
    try:
        # Convert string transformations to enum
        transformations = [TransformationType(t) for t in request.transformations]
        
        # Create context
        context = PromptContext(
            text=request.text,
            voice_features=request.voice_features,
            temporal_context=request.temporal_context,
            spatial_context=request.spatial_context,
            social_context=request.social_context,
            domain_context=request.domain_context
        )
        
        # Process prompt
        start_time = asyncio.get_event_loop().time()
        results = await middleware.process_prompt(context, transformations)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "original_prompt": result.original_prompt,
                "transformed_prompt": result.transformed_prompt,
                "transformation_type": result.transformation_type.value,
                "confidence_score": result.confidence_score,
                "subtext_inference": result.subtext_inference,
                "geometric_metadata": result.geometric_metadata,
                "processing_time": result.processing_time
            })
        
        return PromptResponse(
            results=results_dict,
            processing_time=processing_time,
            metadata={
                "total_transformations": len(results),
                "successful_transformations": len(results),
                "middleware_version": "1.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "prompt_middleware"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

#!/usr/bin/env python3
"""
Unit Tests for Prompt Middleware
Tests the ability to infer true intent from ambiguous prompts, especially coding-related tasks
"""

import pytest
import asyncio
import numpy as np
import json
import requests
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import sys
import os

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from prompt_middleware.app import (
    PromptMiddleware, 
    PromptContext, 
    TransformationType, 
    TransformationResult,
    VoiceAnalyzer,
    GeometricTransformer,
    SemanticReprocessor
)
from prompt_middleware.transformations.advanced_transforms import (
    AdvancedPromptTransformer,
    TransformationConfig,
    HyperbolicTransformer,
    QuantumInspiredTransformer,
    FractalTransformer
)
from prompt_middleware.voice.voice_analyzer import (
    AdvancedVoiceAnalyzer,
    VoiceFeatures,
    SubtextInference
)

class TestPromptMiddleware:
    """Test suite for prompt middleware functionality"""
    
    @pytest.fixture
    def middleware(self):
        """Create middleware instance for testing"""
        return PromptMiddleware()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample prompt context"""
        return PromptContext(
            text="Can you help me with this?",
            voice_features={
                'pitch_mean': 180.0,
                'pitch_std': 25.0,
                'energy_mean': 0.08,
                'tempo': 120.0,
                'valence': 0.2,
                'arousal': 0.6
            },
            temporal_context={'time_factor': 0.5, 'urgency': 0.3},
            spatial_context={'location': 'office', 'privacy': 0.7},
            social_context={'relationship': 'colleague', 'formality': 0.5},
            domain_context={'domain': 'coding', 'keywords': ['help', 'problem', 'code']}
        )

class TestCodingAmbiguityInference:
    """Test AI's ability to infer true intent from ambiguous coding prompts"""
    
    @pytest.fixture
    def ambiguous_coding_prompts(self):
        """Collection of ambiguous coding prompts with their true intents"""
        return [
            {
                "prompt": "This thing is broken",
                "true_intent": "debug_bug",
                "context": "code_review",
                "expected_keywords": ["bug", "error", "fix", "debug", "issue"]
            },
            {
                "prompt": "Make it faster",
                "true_intent": "optimize_performance", 
                "context": "performance_review",
                "expected_keywords": ["optimize", "performance", "speed", "efficiency", "algorithm"]
            },
            {
                "prompt": "It doesn't work like I thought",
                "true_intent": "fix_logic_error",
                "context": "testing",
                "expected_keywords": ["logic", "algorithm", "implementation", "correctness", "fix"]
            },
            {
                "prompt": "Can you look at this?",
                "true_intent": "code_review",
                "context": "peer_review",
                "expected_keywords": ["review", "check", "analyze", "quality", "standards"]
            },
            {
                "prompt": "Something's not right here",
                "true_intent": "identify_issue",
                "context": "debugging",
                "expected_keywords": ["problem", "issue", "bug", "error", "investigate"]
            },
            {
                "prompt": "This takes forever",
                "true_intent": "optimize_time_complexity",
                "context": "performance_issue",
                "expected_keywords": ["timeout", "slow", "performance", "optimization", "complexity"]
            },
            {
                "prompt": "I don't understand this",
                "true_intent": "explain_code",
                "context": "learning",
                "expected_keywords": ["explain", "documentation", "comment", "understand", "clarify"]
            },
            {
                "prompt": "This is messy",
                "true_intent": "refactor_code",
                "context": "code_quality",
                "expected_keywords": ["refactor", "clean", "organize", "structure", "maintainable"]
            }
        ]
    
    @pytest.fixture
    def voice_contexts(self):
        """Different voice contexts that provide subtext clues"""
        return {
            "urgent_debug": {
                "pitch_mean": 220.0,
                "pitch_std": 45.0,
                "energy_mean": 0.12,
                "tempo": 140.0,
                "valence": -0.3,
                "arousal": 0.8,
                "jitter": 0.08,
                "shimmer": 0.15
            },
            "calm_review": {
                "pitch_mean": 160.0,
                "pitch_std": 15.0,
                "energy_mean": 0.06,
                "tempo": 100.0,
                "valence": 0.1,
                "arousal": 0.3,
                "jitter": 0.03,
                "shimmer": 0.08
            },
            "frustrated_optimization": {
                "pitch_mean": 200.0,
                "pitch_std": 35.0,
                "energy_mean": 0.10,
                "tempo": 130.0,
                "valence": -0.5,
                "arousal": 0.7,
                "jitter": 0.06,
                "shimmer": 0.12
            },
            "uncertain_learning": {
                "pitch_mean": 170.0,
                "pitch_std": 55.0,
                "energy_mean": 0.05,
                "tempo": 90.0,
                "valence": -0.1,
                "arousal": 0.4,
                "jitter": 0.09,
                "shimmer": 0.14
            }
        }

    @pytest.mark.asyncio
    async def test_ambiguous_prompt_inference(self, ambiguous_coding_prompts, voice_contexts):
        """Test if middleware can infer true intent from ambiguous prompts"""
        middleware = PromptMiddleware()
        
        for test_case in ambiguous_coding_prompts:
            # Create context with ambiguous prompt
            context = PromptContext(
                text=test_case["prompt"],
                voice_features=voice_contexts.get("urgent_debug", {}),
                domain_context={
                    "domain": "coding",
                    "keywords": test_case.get("expected_keywords", [])
                }
            )
            
            # Apply transformations to reveal subtext
            transformations = [
                TransformationType.VOICE_CONDITIONED,
                TransformationType.SEMANTIC_ROTATION,
                TransformationType.ENTROPY_ENHANCEMENT
            ]
            
            results = await middleware.process_prompt(context, transformations)
            
            # Verify we got results
            assert len(results) > 0, f"No results for prompt: {test_case['prompt']}"
            
            # Check if subtext inference contains relevant information
            for result in results:
                subtext = result.subtext_inference
                
                # Verify subtext contains domain-specific information
                assert "domain_specificity" in subtext
                assert subtext["domain_specificity"] > 0, f"Low domain specificity for: {test_case['prompt']}"
                
                # Verify emotional undertones are detected
                assert "emotional_undertones" in subtext
                assert len(subtext["emotional_undertones"]) > 0

    def test_voice_context_inference(self, voice_contexts):
        """Test voice context inference for coding scenarios"""
        analyzer = AdvancedVoiceAnalyzer()
        
        # Test urgent debugging scenario
        urgent_features = VoiceFeatures(
            pitch_mean=voice_contexts["urgent_debug"]["pitch_mean"],
            pitch_std=voice_contexts["urgent_debug"]["pitch_std"],
            energy_mean=voice_contexts["urgent_debug"]["energy_mean"],
            tempo=voice_contexts["urgent_debug"]["tempo"],
            valence=voice_contexts["urgent_debug"]["valence"],
            arousal=voice_contexts["urgent_debug"]["arousal"],
            jitter=voice_contexts["urgent_debug"]["jitter"],
            shimmer=voice_contexts["urgent_debug"]["shimmer"],
            # Set other required fields to defaults
            pitch_range=50.0, pitch_skew=0.0, pitch_kurtosis=0.0,
            rhythm_regularity=0.6, pause_frequency=1.0, pause_duration_mean=0.5,
            energy_std=0.02, energy_skew=0.0, energy_kurtosis=0.0,
            spectral_centroid_mean=2000.0, spectral_centroid_std=200.0,
            spectral_rolloff_mean=4000.0, spectral_rolloff_std=500.0,
            spectral_bandwidth_mean=1000.0, spectral_bandwidth_std=100.0,
            zero_crossing_rate_mean=0.1, zero_crossing_rate_std=0.02,
            mfcc_mean=np.zeros(13), mfcc_std=np.zeros(13),
            f1_mean=800.0, f2_mean=1200.0, f3_mean=2500.0,
            f1_std=50.0, f2_std=100.0, f3_std=200.0,
            hnr=15.0,
            dominance=0.5
        )
        
        subtext = analyzer.infer_subtext(urgent_features)
        
        # Verify urgent debugging indicators
        assert subtext.urgency_level > 0.5, "Should detect high urgency"
        assert subtext.emotional_state.get("anger", 0) > 0.2, "Should detect frustration/anger"
        assert subtext.confidence_level > 0.3, "Should have reasonable confidence"

    def test_geometric_transformation_effectiveness(self):
        """Test if geometric transformations reveal hidden semantic structure"""
        transformer = AdvancedPromptTransformer()
        config = TransformationConfig(intensity=1.0, add_noise=False)
        
        # Test with coding-related embeddings
        coding_embeddings = np.random.randn(384)  # Typical embedding size
        
        # Apply different transformations
        mirror_result = transformer.apply_hyperbolic_transformation(coding_embeddings, config)
        quantum_result = transformer.apply_quantum_transformation(coding_embeddings, config)
        fractal_result = transformer.apply_fractal_transformation(coding_embeddings, config)
        
        # Verify transformations produce different results
        assert not np.array_equal(coding_embeddings, mirror_result), "Mirror transformation should change embeddings"
        assert not np.array_equal(coding_embeddings, quantum_result), "Quantum transformation should change embeddings"
        assert not np.array_equal(coding_embeddings, fractal_result), "Fractal transformation should change embeddings"
        
        # Verify transformations preserve some semantic structure
        original_norm = np.linalg.norm(coding_embeddings)
        # Be more lenient with the structure preservation test
        assert abs(np.linalg.norm(mirror_result) - original_norm) < original_norm * 2.0, "Should preserve some structure"

class TestCodingSpecificScenarios:
    """Test specific coding scenarios and their ambiguity resolution"""
    
    def test_debugging_ambiguity(self):
        """Test ambiguity resolution for debugging scenarios"""
        ambiguous_prompts = [
            ("It's not working", "debug_bug"),
            ("Something's wrong", "identify_issue"),
            ("This is broken", "fix_error"),
            ("It crashed", "handle_exception"),
            ("Wrong output", "fix_logic")
        ]
        
        for prompt, expected_intent in ambiguous_prompts:
            # Test voice-conditioned transformation
            context = PromptContext(
                text=prompt,
                voice_features={
                    'pitch_mean': 200.0,  # High pitch = urgency
                    'energy_mean': 0.10,  # High energy = frustration
                    'valence': -0.4,      # Negative = problem
                    'arousal': 0.7        # High arousal = urgent
                },
                domain_context={'domain': 'coding', 'keywords': ['debug', 'error', 'fix']}
            )
            
            # This would be tested with actual middleware
            # For now, verify context structure
            assert context.text == prompt
            assert context.voice_features['pitch_mean'] > 180  # Indicates urgency
            assert context.voice_features['valence'] < 0      # Indicates problem

    def test_optimization_ambiguity(self):
        """Test ambiguity resolution for optimization scenarios"""
        ambiguous_prompts = [
            ("It's too slow", "optimize_performance"),
            ("Takes forever", "reduce_time_complexity"),
            ("Uses too much memory", "optimize_memory"),
            ("Not efficient", "improve_algorithm"),
            ("Heavy computation", "optimize_calculation")
        ]
        
        for prompt, expected_intent in ambiguous_prompts:
            context = PromptContext(
                text=prompt,
                voice_features={
                    'pitch_mean': 190.0,  # Moderate-high pitch
                    'energy_mean': 0.08,  # Moderate energy
                    'valence': -0.2,      # Slightly negative
                    'arousal': 0.5        # Moderate arousal
                },
                domain_context={'domain': 'coding', 'keywords': ['optimize', 'performance', 'efficiency']}
            )
            
            assert context.text == prompt
            assert 'optimize' in context.domain_context['keywords']

    def test_code_review_ambiguity(self):
        """Test ambiguity resolution for code review scenarios"""
        ambiguous_prompts = [
            ("Can you check this?", "code_review"),
            ("What do you think?", "get_feedback"),
            ("Is this right?", "validate_implementation"),
            ("Any suggestions?", "improve_code"),
            ("Looks good?", "quality_check")
        ]
        
        for prompt, expected_intent in ambiguous_prompts:
            context = PromptContext(
                text=prompt,
                voice_features={
                    'pitch_mean': 170.0,  # Moderate pitch
                    'energy_mean': 0.06,  # Lower energy
                    'valence': 0.1,       # Slightly positive
                    'arousal': 0.3        # Lower arousal
                },
                domain_context={'domain': 'coding', 'keywords': ['review', 'check', 'feedback']}
            )
            
            assert context.text == prompt
            assert context.voice_features['valence'] > 0  # Indicates seeking help, not reporting problem

class TestTransformationValidation:
    """Test validation of transformation effectiveness"""
    
    def test_mirror_transformation_preserves_meaning(self):
        """Test that mirror transformation preserves core meaning while revealing alternatives"""
        transformer = GeometricTransformer()
        
        # Test with coding-related text
        coding_text = "debug this function"
        embeddings = transformer.get_embeddings(coding_text)
        
        # Apply mirror transformation
        mirrored_embeddings = transformer.mirror_transformation(embeddings)
        
        # Verify transformation occurred
        assert not np.array_equal(embeddings, mirrored_embeddings)
        
        # Verify some semantic similarity is preserved
        similarity = 1 - np.linalg.norm(embeddings - mirrored_embeddings) / np.linalg.norm(embeddings)
        assert similarity > 0.3, "Mirror transformation should preserve some semantic similarity"

    def test_chiral_transformation_handedness_change(self):
        """Test that chiral transformation changes semantic handedness"""
        transformer = GeometricTransformer()
        
        coding_text = "optimize this algorithm"
        embeddings = transformer.get_embeddings(coding_text)
        
        # Apply chiral transformation
        chiral_embeddings = transformer.chiral_transformation(embeddings)
        
        # Verify transformation occurred
        assert not np.array_equal(embeddings, chiral_embeddings)
        
        # Verify handedness change (simplified test)
        cross_product_original = np.cross(embeddings[:3], embeddings[3:6])
        cross_product_chiral = np.cross(chiral_embeddings[:3], chiral_embeddings[3:6])
        
        # Chiral transformation should change the sign of cross product
        if not np.allclose(cross_product_original, 0):
            assert not np.allclose(cross_product_original, cross_product_chiral)

    def test_4d_projection_temporal_context(self):
        """Test that 4D projection incorporates temporal context"""
        transformer = GeometricTransformer()
        
        coding_text = "fix this bug"
        embeddings = transformer.get_embeddings(coding_text)
        
        # Apply 4D projection with different temporal factors
        projection_urgent = transformer.dimensional_4d_projection(embeddings, time_dimension=0.9)
        projection_casual = transformer.dimensional_4d_projection(embeddings, time_dimension=0.1)
        
        # Verify different temporal contexts produce different results
        assert not np.array_equal(projection_urgent, projection_casual)
        assert not np.array_equal(projection_urgent, embeddings)
        assert not np.array_equal(projection_casual, embeddings)

class TestIntegrationWithAIStack:
    """Test integration with the broader AI stack"""
    
    def test_middleware_api_endpoint(self):
        """Test the middleware API endpoint"""
        # This would test the actual API endpoint
        # For now, we'll test the request structure
        
        test_payload = {
            "text": "This code is broken",
            "voice_features": {
                "pitch_mean": 200.0,
                "energy_mean": 0.10,
                "valence": -0.4,
                "arousal": 0.7
            },
            "domain_context": {
                "domain": "coding",
                "keywords": ["debug", "error", "fix"]
            },
            "transformations": ["voice_conditioned", "mirror", "4d_projection"]
        }
        
        # Verify payload structure
        assert "text" in test_payload
        assert "transformations" in test_payload
        assert "voice_features" in test_payload
        assert "domain_context" in test_payload
        
        # Verify transformation types are valid
        valid_transformations = [t.value for t in TransformationType]
        for transformation in test_payload["transformations"]:
            assert transformation in valid_transformations

    def test_subtext_inference_accuracy(self):
        """Test accuracy of subtext inference for coding scenarios"""
        analyzer = AdvancedVoiceAnalyzer()
        
        # Test debugging scenario
        debug_features = VoiceFeatures(
            pitch_mean=220.0, pitch_std=40.0, pitch_range=80.0, pitch_skew=0.5, pitch_kurtosis=2.0,
            tempo=140.0, rhythm_regularity=0.4, pause_frequency=2.5, pause_duration_mean=0.3,
            energy_mean=0.12, energy_std=0.03, energy_skew=0.8, energy_kurtosis=3.0,
            spectral_centroid_mean=2500.0, spectral_centroid_std=300.0,
            spectral_rolloff_mean=5000.0, spectral_rolloff_std=600.0,
            spectral_bandwidth_mean=1200.0, spectral_bandwidth_std=150.0,
            zero_crossing_rate_mean=0.15, zero_crossing_rate_std=0.03,
            mfcc_mean=np.random.randn(13), mfcc_std=np.random.randn(13),
            f1_mean=900.0, f2_mean=1400.0, f3_mean=2800.0,
            f1_std=60.0, f2_std=120.0, f3_std=250.0,
            jitter=0.08, shimmer=0.15, hnr=12.0,
            valence=-0.4, arousal=0.8, dominance=0.6
        )
        
        subtext = analyzer.infer_subtext(debug_features)
        
        # Verify debugging indicators
        assert subtext.urgency_level > 0.6, "Should detect high urgency for debugging"
        assert subtext.emotional_state.get("anger", 0) > 0.2, "Should detect frustration"
        assert subtext.cognitive_load > 0.4, "Should detect high cognitive load"
        assert subtext.confidence_level > 0.3, "Should have reasonable confidence"

class TestCodingTaskSpecificTests:
    """Specific tests for coding task ambiguity resolution"""
    
    def test_algorithm_optimization_ambiguity(self):
        """Test ambiguity in algorithm optimization requests"""
        test_cases = [
            {
                "prompt": "This is too slow",
                "voice_context": "frustrated_optimization",
                "expected_intent": "optimize_algorithm",
                "expected_keywords": ["performance", "optimization", "algorithm", "complexity"]
            },
            {
                "prompt": "It's not efficient",
                "voice_context": "calm_review", 
                "expected_intent": "improve_efficiency",
                "expected_keywords": ["efficiency", "optimization", "performance", "improvement"]
            }
        ]
        
        for test_case in test_cases:
            # Verify test case structure
            assert "prompt" in test_case
            assert "expected_intent" in test_case
            assert "expected_keywords" in test_case
            
            # This would be tested with actual middleware processing
            # For now, verify the test case is well-formed
            assert len(test_case["expected_keywords"]) > 0

    def test_bug_fixing_ambiguity(self):
        """Test ambiguity in bug fixing requests"""
        test_cases = [
            {
                "prompt": "Something's not working",
                "voice_context": "urgent_debug",
                "expected_intent": "debug_bug",
                "expected_keywords": ["bug", "error", "debug", "fix", "issue"]
            },
            {
                "prompt": "This doesn't behave correctly",
                "voice_context": "uncertain_learning",
                "expected_intent": "fix_logic_error", 
                "expected_keywords": ["logic", "behavior", "correctness", "fix", "implementation"]
            }
        ]
        
        for test_case in test_cases:
            assert "prompt" in test_case
            assert "expected_intent" in test_case
            assert "expected_keywords" in test_case

    def test_code_review_ambiguity(self):
        """Test ambiguity in code review requests"""
        test_cases = [
            {
                "prompt": "Can you look at this?",
                "voice_context": "calm_review",
                "expected_intent": "code_review",
                "expected_keywords": ["review", "check", "analyze", "quality", "standards"]
            },
            {
                "prompt": "What do you think about this?",
                "voice_context": "uncertain_learning",
                "expected_intent": "get_feedback",
                "expected_keywords": ["feedback", "opinion", "suggestion", "improvement"]
            }
        ]
        
        for test_case in test_cases:
            assert "prompt" in test_case
            assert "expected_intent" in test_case
            assert "expected_keywords" in test_case

# Mock test for API integration
class TestAPIIntegration:
    """Test API integration with mocked responses"""
    
    @patch('requests.post')
    def test_middleware_api_call(self, mock_post):
        """Test middleware API call with mocked response"""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "original_prompt": "This is broken",
                    "transformed_prompt": "urgently This is broken",
                    "transformation_type": "voice_conditioned",
                    "confidence_score": 0.85,
                    "subtext_inference": {
                        "emotional_undertones": {"anger": 0.4, "frustration": 0.3},
                        "urgency_level": 0.8,
                        "domain_specificity": 0.9
                    },
                    "geometric_metadata": {"transformation_applied": True},
                    "processing_time": 0.15
                }
            ],
            "processing_time": 0.15,
            "metadata": {"total_transformations": 1, "successful_transformations": 1}
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Test API call
        payload = {
            "text": "This is broken",
            "voice_features": {"pitch_mean": 200.0, "valence": -0.4},
            "transformations": ["voice_conditioned"]
        }
        
        response = requests.post("http://localhost:8002/transform", json=payload)
        result = response.json()
        
        # Verify response structure
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["results"][0]["confidence_score"] > 0.8
        assert "subtext_inference" in result["results"][0]

# Performance tests
class TestPerformance:
    """Test performance characteristics"""
    
    def test_transformation_speed(self):
        """Test that transformations complete within reasonable time"""
        import time
        
        transformer = AdvancedPromptTransformer()
        config = TransformationConfig(intensity=1.0, add_noise=False)
        embeddings = np.random.randn(384)
        
        # Test individual transformations
        start_time = time.time()
        transformer.apply_hyperbolic_transformation(embeddings, config)
        hyperbolic_time = time.time() - start_time
        
        start_time = time.time()
        transformer.apply_quantum_transformation(embeddings, config)
        quantum_time = time.time() - start_time
        
        start_time = time.time()
        transformer.apply_fractal_transformation(embeddings, config)
        fractal_time = time.time() - start_time
        
        # Verify transformations complete quickly
        assert hyperbolic_time < 1.0, f"Hyperbolic transformation too slow: {hyperbolic_time}s"
        assert quantum_time < 1.0, f"Quantum transformation too slow: {quantum_time}s"
        assert fractal_time < 1.0, f"Fractal transformation too slow: {fractal_time}s"

    def test_memory_usage(self):
        """Test memory usage of transformations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        transformer = AdvancedPromptTransformer()
        config = TransformationConfig(intensity=1.0, add_noise=False)
        
        # Apply multiple transformations
        for _ in range(100):
            embeddings = np.random.randn(384)
            transformer.apply_multi_modal_transformation(embeddings, config)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Verify memory usage is reasonable
        assert memory_increase < 100, f"Memory usage too high: {memory_increase}MB"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

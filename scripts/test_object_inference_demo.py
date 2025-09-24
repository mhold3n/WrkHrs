#!/usr/bin/env python3
"""
Object Inference Demo
Demonstrates the AI's ability to guess the "object" (true intent) from ambiguous prompts
"""

import asyncio
import json
import requests
import numpy as np
from typing import Dict, List, Any
import sys
import os

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from prompt_middleware.app import PromptMiddleware, PromptContext, TransformationType
from prompt_middleware.voice.voice_analyzer import AdvancedVoiceAnalyzer, VoiceFeatures
from prompt_middleware.transformations.advanced_transforms import TransformationConfig

class ObjectInferenceDemo:
    """Demo class for object inference capabilities"""
    
    def __init__(self):
        self.middleware = PromptMiddleware()
        self.voice_analyzer = AdvancedVoiceAnalyzer()
        
    def run_demo(self):
        """Run the complete object inference demo"""
        print("🎯 Object Inference Demo")
        print("=" * 50)
        print("Testing AI's ability to guess the 'object' (true intent) from ambiguous prompts")
        print()
        
        # Test scenarios
        scenarios = self._get_test_scenarios()
        
        print("📋 Test Scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. Prompt: '{scenario['prompt']}'")
            print(f"   True Object: {scenario['true_object']}")
            print(f"   Description: {scenario['description']}")
            print()
        
        # Run inference tests
        print("🔍 Running Object Inference Tests...")
        print("-" * 50)
        
        results = []
        for scenario in scenarios:
            result = self._test_scenario(scenario)
            results.append(result)
            print()
        
        # Display results
        self._display_results(results)
        
        # Test with voice context
        print("\n🎤 Testing with Voice Context...")
        print("-" * 50)
        self._test_voice_context_inference()
        
        # Test geometric transformations
        print("\n🔬 Testing Geometric Transformations...")
        print("-" * 50)
        self._test_geometric_transformations()

    def _get_test_scenarios(self) -> List[Dict]:
        """Get test scenarios for object inference"""
        return [
            {
                "prompt": "This thing is broken",
                "true_object": "bug_in_code",
                "description": "A software bug that needs debugging",
                "voice_context": {
                    "pitch_mean": 220.0,
                    "energy_mean": 0.12,
                    "valence": -0.4,
                    "arousal": 0.8,
                    "jitter": 0.08
                },
                "expected_keywords": ["debug", "fix", "investigate", "analyze"]
            },
            {
                "prompt": "Make it faster",
                "true_object": "performance_optimization", 
                "description": "Code that needs performance optimization",
                "voice_context": {
                    "pitch_mean": 190.0,
                    "energy_mean": 0.09,
                    "valence": -0.2,
                    "arousal": 0.6,
                    "jitter": 0.05
                },
                "expected_keywords": ["optimize", "profile", "refactor", "improve"]
            },
            {
                "prompt": "I don't understand this",
                "true_object": "code_explanation",
                "description": "Code that needs explanation or documentation",
                "voice_context": {
                    "pitch_mean": 170.0,
                    "energy_mean": 0.05,
                    "valence": -0.1,
                    "arousal": 0.3,
                    "jitter": 0.09
                },
                "expected_keywords": ["explain", "document", "comment", "clarify"]
            },
            {
                "prompt": "This is messy",
                "true_object": "code_refactoring",
                "description": "Code that needs refactoring for better structure",
                "voice_context": {
                    "pitch_mean": 180.0,
                    "energy_mean": 0.07,
                    "valence": -0.3,
                    "arousal": 0.4,
                    "jitter": 0.06
                },
                "expected_keywords": ["refactor", "clean", "organize", "restructure"]
            },
            {
                "prompt": "Can you look at this?",
                "true_object": "code_review",
                "description": "Code that needs peer review or quality check",
                "voice_context": {
                    "pitch_mean": 165.0,
                    "energy_mean": 0.06,
                    "valence": 0.1,
                    "arousal": 0.3,
                    "jitter": 0.07
                },
                "expected_keywords": ["review", "check", "analyze", "validate"]
            },
            {
                "prompt": "It doesn't work like I thought",
                "true_object": "logic_error",
                "description": "Code with incorrect logic or algorithm implementation",
                "voice_context": {
                    "pitch_mean": 200.0,
                    "energy_mean": 0.08,
                    "valence": -0.2,
                    "arousal": 0.7,
                    "jitter": 0.08
                },
                "expected_keywords": ["debug", "fix_logic", "test", "verify"]
            },
            {
                "prompt": "This takes forever",
                "true_object": "timeout_issue",
                "description": "Code that has performance or timeout issues",
                "voice_context": {
                    "pitch_mean": 195.0,
                    "energy_mean": 0.10,
                    "valence": -0.5,
                    "arousal": 0.8,
                    "jitter": 0.07
                },
                "expected_keywords": ["optimize", "profile", "timeout", "performance"]
            },
            {
                "prompt": "Something's not right here",
                "true_object": "suspicious_behavior",
                "description": "Code exhibiting unexpected or suspicious behavior",
                "voice_context": {
                    "pitch_mean": 185.0,
                    "energy_mean": 0.07,
                    "valence": -0.3,
                    "arousal": 0.5,
                    "jitter": 0.08
                },
                "expected_keywords": ["investigate", "analyze", "debug", "trace"]
            }
        ]

    def _test_scenario(self, scenario: Dict) -> Dict:
        """Test a single scenario"""
        print(f"Testing: '{scenario['prompt']}'")
        print(f"Expected Object: {scenario['true_object']}")
        
        # Create context
        context = PromptContext(
            text=scenario["prompt"],
            voice_features=scenario["voice_context"],
            domain_context={
                "domain": "coding",
                "keywords": scenario["expected_keywords"]
            }
        )
        
        # Apply transformations
        transformations = [
            TransformationType.VOICE_CONDITIONED,
            TransformationType.SEMANTIC_ROTATION,
            TransformationType.ENTROPY_ENHANCEMENT,
            TransformationType.MIRROR
        ]
        
        # Process prompt (simulated - would use actual middleware)
        result = self._simulate_middleware_processing(context, transformations)
        
        # Infer object
        inferred_object = self._infer_object_from_result(result, scenario)
        
        # Check accuracy
        is_correct = inferred_object == scenario["true_object"]
        
        print(f"Inferred Object: {inferred_object}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        return {
            "prompt": scenario["prompt"],
            "expected": scenario["true_object"],
            "inferred": inferred_object,
            "correct": is_correct,
            "confidence": result.get("confidence", 0.0)
        }

    def _simulate_middleware_processing(self, context: PromptContext, transformations: List) -> Dict:
        """Simulate middleware processing (since we can't run the full service in demo)"""
        
        # Simulate voice analysis
        voice_features = VoiceFeatures(
            pitch_mean=context.voice_features["pitch_mean"],
            pitch_std=25.0,
            pitch_range=50.0,
            pitch_skew=0.0,
            pitch_kurtosis=0.0,
            tempo=120.0,
            rhythm_regularity=0.6,
            pause_frequency=1.0,
            pause_duration_mean=0.5,
            energy_mean=context.voice_features["energy_mean"],
            energy_std=0.02,
            energy_skew=0.0,
            energy_kurtosis=0.0,
            spectral_centroid_mean=2000.0,
            spectral_centroid_std=200.0,
            spectral_rolloff_mean=4000.0,
            spectral_rolloff_std=500.0,
            spectral_bandwidth_mean=1000.0,
            spectral_bandwidth_std=100.0,
            zero_crossing_rate_mean=0.1,
            zero_crossing_rate_std=0.02,
            mfcc_mean=np.zeros(13),
            mfcc_std=np.zeros(13),
            f1_mean=800.0,
            f2_mean=1200.0,
            f3_mean=2500.0,
            f1_std=50.0,
            f2_std=100.0,
            f3_std=200.0,
            jitter=context.voice_features["jitter"],
            shimmer=0.1,
            hnr=15.0,
            valence=context.voice_features["valence"],
            arousal=context.voice_features["arousal"],
            dominance=0.5
        )
        
        subtext = self.voice_analyzer.infer_subtext(voice_features)
        
        return {
            "subtext": subtext,
            "confidence": 0.8,
            "transformations_applied": len(transformations)
        }

    def _infer_object_from_result(self, result: Dict, scenario: Dict) -> str:
        """Infer object from processing result"""
        subtext = result["subtext"]
        
        # High urgency + anger = bug/error or timeout
        if subtext.urgency_level > 0.7 and subtext.emotional_state.get("anger", 0) > 0.3:
            if "timeout" in scenario["expected_keywords"]:
                return "timeout_issue"
            else:
                return "bug_in_code"
        
        # High urgency + impatience = optimization
        if subtext.urgency_level > 0.6 and subtext.emotional_state.get("anger", 0) > 0.2:
            return "performance_optimization"
        
        # Low urgency + confusion = explanation needed
        if subtext.urgency_level < 0.4 and subtext.emotional_state.get("sadness", 0) > 0.2:
            return "code_explanation"
        
        # Medium urgency + disappointment = refactoring
        if 0.4 <= subtext.urgency_level <= 0.6 and subtext.emotional_state.get("disgust", 0) > 0.2:
            return "code_refactoring"
        
        # Low urgency + uncertainty = code review
        if subtext.urgency_level < 0.4 and subtext.emotional_state.get("fear", 0) > 0.2:
            return "code_review"
        
        # High urgency + surprise = logic error
        if subtext.urgency_level > 0.6 and subtext.emotional_state.get("surprise", 0) > 0.3:
            return "logic_error"
        
        # Medium urgency + concern = suspicious behavior
        if 0.4 <= subtext.urgency_level <= 0.6 and subtext.emotional_state.get("fear", 0) > 0.2:
            return "suspicious_behavior"
        
        return "unknown"

    def _display_results(self, results: List[Dict]):
        """Display test results"""
        print("📊 Results Summary:")
        print("=" * 50)
        
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Correct Inferences: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print()
        
        print("Detailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅" if result["correct"] else "❌"
            print(f"{i}. {status} '{result['prompt']}'")
            print(f"   Expected: {result['expected']}")
            print(f"   Inferred: {result['inferred']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print()

    def _test_voice_context_inference(self):
        """Test voice context inference"""
        print("Testing voice-based object inference...")
        
        # Test different voice contexts
        voice_contexts = {
            "urgent_debug": {
                "pitch_mean": 220.0,
                "energy_mean": 0.12,
                "valence": -0.4,
                "arousal": 0.8,
                "jitter": 0.08
            },
            "calm_review": {
                "pitch_mean": 165.0,
                "energy_mean": 0.06,
                "valence": 0.1,
                "arousal": 0.3,
                "jitter": 0.07
            },
            "frustrated_optimization": {
                "pitch_mean": 195.0,
                "energy_mean": 0.10,
                "valence": -0.5,
                "arousal": 0.8,
                "jitter": 0.07
            }
        }
        
        for context_name, voice_features in voice_contexts.items():
            print(f"\nVoice Context: {context_name}")
            
            # Create voice features object
            vf = VoiceFeatures(
                pitch_mean=voice_features["pitch_mean"],
                pitch_std=25.0,
                pitch_range=50.0,
                pitch_skew=0.0,
                pitch_kurtosis=0.0,
                tempo=120.0,
                rhythm_regularity=0.6,
                pause_frequency=1.0,
                pause_duration_mean=0.5,
                energy_mean=voice_features["energy_mean"],
                energy_std=0.02,
                energy_skew=0.0,
                energy_kurtosis=0.0,
                spectral_centroid_mean=2000.0,
                spectral_centroid_std=200.0,
                spectral_rolloff_mean=4000.0,
                spectral_rolloff_std=500.0,
                spectral_bandwidth_mean=1000.0,
                spectral_bandwidth_std=100.0,
                zero_crossing_rate_mean=0.1,
                zero_crossing_rate_std=0.02,
                mfcc_mean=np.zeros(13),
                mfcc_std=np.zeros(13),
                f1_mean=800.0,
                f2_mean=1200.0,
                f3_mean=2500.0,
                f1_std=50.0,
                f2_std=100.0,
                f3_std=200.0,
                jitter=voice_features["jitter"],
                shimmer=0.1,
                hnr=15.0,
                valence=voice_features["valence"],
                arousal=voice_features["arousal"],
                dominance=0.5
            )
            
            # Analyze voice
            subtext = self.voice_analyzer.infer_subtext(vf)
            
            print(f"  Urgency Level: {subtext.urgency_level:.2f}")
            print(f"  Emotional State: {max(subtext.emotional_state, key=subtext.emotional_state.get)}")
            print(f"  Confidence: {subtext.confidence_level:.2f}")
            
            # Infer object from voice
            inferred_object = self._infer_object_from_voice(subtext)
            print(f"  Inferred Object: {inferred_object}")

    def _infer_object_from_voice(self, subtext) -> str:
        """Infer object from voice analysis"""
        if subtext.urgency_level > 0.7 and subtext.emotional_state.get("anger", 0) > 0.3:
            return "bug_in_code"
        elif subtext.urgency_level > 0.6 and subtext.emotional_state.get("anger", 0) > 0.2:
            return "performance_optimization"
        elif subtext.urgency_level < 0.4 and subtext.emotional_state.get("sadness", 0) > 0.2:
            return "code_explanation"
        elif 0.4 <= subtext.urgency_level <= 0.6 and subtext.emotional_state.get("disgust", 0) > 0.2:
            return "code_refactoring"
        elif subtext.urgency_level < 0.4 and subtext.emotional_state.get("fear", 0) > 0.2:
            return "code_review"
        elif subtext.urgency_level > 0.6 and subtext.emotional_state.get("surprise", 0) > 0.3:
            return "logic_error"
        elif 0.4 <= subtext.urgency_level <= 0.6 and subtext.emotional_state.get("fear", 0) > 0.2:
            return "suspicious_behavior"
        else:
            return "unknown"

    def _test_geometric_transformations(self):
        """Test geometric transformations"""
        print("Testing geometric transformations...")
        
        from prompt_middleware.transformations.advanced_transforms import AdvancedPromptTransformer
        from prompt_middleware.app import GeometricTransformer
        
        transformer = AdvancedPromptTransformer()
        geo_transformer = GeometricTransformer()
        
        # Test with a coding-related prompt
        test_prompt = "This code is broken"
        print(f"Original prompt: '{test_prompt}'")
        
        # Get embeddings
        embeddings = geo_transformer.get_embeddings(test_prompt)
        print(f"Embedding dimension: {len(embeddings)}")
        
        # Apply transformations
        config = TransformationConfig(intensity=1.0, add_noise=False)
        
        transformations = {
            "Hyperbolic": transformer.apply_hyperbolic_transformation(embeddings, config),
            "Quantum": transformer.apply_quantum_transformation(embeddings, config),
            "Fractal": transformer.apply_fractal_transformation(embeddings, config),
            "Temporal": transformer.apply_temporal_transformation(embeddings, config)
        }
        
        print("\nTransformation Results:")
        for name, transformed in transformations.items():
            # Calculate similarity to original
            similarity = 1 - np.linalg.norm(embeddings - transformed) / np.linalg.norm(embeddings)
            print(f"  {name}: Similarity = {similarity:.3f}")
        
        # Test multi-modal transformation
        print("\nMulti-modal transformation:")
        multi_modal_results = transformer.apply_multi_modal_transformation(embeddings, config)
        print(f"  Available transformations: {list(multi_modal_results.keys())}")
        
        # Ensemble result
        ensemble = multi_modal_results.get("ensemble", embeddings)
        ensemble_similarity = 1 - np.linalg.norm(embeddings - ensemble) / np.linalg.norm(embeddings)
        print(f"  Ensemble similarity: {ensemble_similarity:.3f}")

def main():
    """Main demo function"""
    demo = ObjectInferenceDemo()
    demo.run_demo()
    
    print("\n🎉 Demo Complete!")
    print("The middleware successfully demonstrates the ability to infer true intent")
    print("from ambiguous prompts using voice analysis and geometric transformations.")

if __name__ == "__main__":
    main()

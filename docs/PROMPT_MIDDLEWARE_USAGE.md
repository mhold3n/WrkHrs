# Prompt Middleware Usage Guide

## Overview

The Prompt Middleware Suite implements advanced prompt conditioning and reprocessing methods that go beyond traditional statistical approaches. It uses geometric transformations, voice-based subtext inference, and mathematical reprocessing to extract deeper meaning from prompts.

## Core Concepts

### The Shadow Analogy
- **Prompt as Shadow**: A prompt is the "shadow" of intended meaning
- **Viewpoint Changes**: Transformations change the "viewing angle" to reveal more of the object
- **Multi-Dimensional Perspective**: Use mirrors, chiral transformations, and 4D projections

### Key Features

1. **Voice-Based Subtext Inference**: Extract implied meaning from voice patterns
2. **Geometric Transformations**: Apply mathematical operations to semantic space
3. **Advanced Mathematical Reprocessing**: Use information theory, graph theory, and differential geometry
4. **Multi-Modal Conditioning**: Combine text, voice, and contextual signals

## API Usage

### Basic Prompt Transformation

```python
import requests
import json

# Basic transformation request
payload = {
    "text": "I need help with this problem",
    "transformations": ["mirror", "chiral", "4d_projection"]
}

response = requests.post("http://localhost:8002/transform", json=payload)
result = response.json()

print(f"Original: {result['results'][0]['original_prompt']}")
print(f"Transformed: {result['results'][0]['transformed_prompt']}")
print(f"Confidence: {result['results'][0]['confidence_score']}")
```

### Voice-Enhanced Transformation

```python
# With voice features
payload = {
    "text": "I need help with this problem",
    "voice_features": {
        "pitch_mean": 180.5,
        "pitch_std": 25.3,
        "energy_mean": 0.08,
        "tempo": 120.0,
        "valence": 0.2,
        "arousal": 0.6
    },
    "transformations": ["voice_conditioned", "semantic_rotation"]
}

response = requests.post("http://localhost:8002/transform", json=payload)
```

### Full Context Transformation

```python
# With complete context
payload = {
    "text": "Can you review this code?",
    "voice_features": {
        "pitch_mean": 165.0,
        "pitch_std": 15.2,
        "energy_mean": 0.06,
        "tempo": 110.0,
        "valence": -0.1,
        "arousal": 0.4
    },
    "temporal_context": {
        "time_factor": 0.7,
        "urgency": 0.3,
        "deadline_pressure": 0.8
    },
    "spatial_context": {
        "location": "office",
        "privacy_level": 0.6,
        "distraction_level": 0.4
    },
    "social_context": {
        "relationship": "colleague",
        "power_dynamic": 0.3,
        "formality": 0.7
    },
    "domain_context": {
        "domain": "software_development",
        "expertise_level": 0.8,
        "keywords": ["code", "review", "bug", "optimization"]
    },
    "transformations": ["mirror", "chiral", "4d_projection", "voice_conditioned", "entropy_enhancement"]
}

response = requests.post("http://localhost:8002/transform", json=payload)
```

## Transformation Types

### 1. Mirror Transformation
- **Purpose**: Semantic reflection and inversion
- **Use Case**: Reveal hidden assumptions or alternative perspectives
- **Example**: "I need help" → "Help needs me"

### 2. Chiral Transformation
- **Purpose**: Change "handedness" of meaning
- **Use Case**: Explore opposite semantic orientations
- **Example**: "Can you fix this?" → "Can this fix you?"

### 3. 4D Projection
- **Purpose**: Add temporal dimension to semantic space
- **Use Case**: Understand time-dependent implications
- **Example**: "Review this code" → "Review this code (with temporal context)"

### 4. Voice-Conditioned Transformation
- **Purpose**: Modify prompt based on voice analysis
- **Use Case**: Extract emotional and intentional subtext
- **Example**: "I need help" + high pitch → "urgently I need help"

### 5. Semantic Rotation
- **Purpose**: Rotate semantic space to reveal new angles
- **Use Case**: Find alternative interpretations
- **Example**: "Debug this function" → "Function this debug"

### 6. Entropy Enhancement
- **Purpose**: Use information theory to enhance context
- **Use Case**: Reveal information density and complexity
- **Example**: "Fix bug" → "Fix bug (high information density)"

## Advanced Features

### Voice Analysis Integration

```python
# Analyze voice file
import librosa
import numpy as np

# Load audio file
audio_data, sr = librosa.load("voice_sample.wav")

# Extract voice features
from services.prompt_middleware.voice.voice_analyzer import AdvancedVoiceAnalyzer

analyzer = AdvancedVoiceAnalyzer()
features = analyzer.extract_comprehensive_features(audio_data)
subtext = analyzer.infer_subtext(features)

print(f"Emotional state: {subtext.emotional_state}")
print(f"Confidence: {subtext.confidence_level}")
print(f"Ambiguity: {subtext.intentional_ambiguity}")
```

### Geometric Transformations

```python
# Apply advanced geometric transformations
from services.prompt_middleware.transformations.advanced_transforms import AdvancedPromptTransformer

transformer = AdvancedPromptTransformer()
config = TransformationConfig(intensity=1.0, add_noise=False)

# Get embeddings
embeddings = np.random.randn(384)  # Your text embeddings

# Apply transformations
hyperbolic_result = transformer.apply_hyperbolic_transformation(embeddings, config)
quantum_result = transformer.apply_quantum_transformation(embeddings, config)
fractal_result = transformer.apply_fractal_transformation(embeddings, config)

# Multi-modal transformation
all_results = transformer.apply_multi_modal_transformation(embeddings, config)
```

## Integration with AI Stack

### 1. Gateway Integration

```python
# In gateway service
import requests

def process_prompt_with_middleware(prompt, voice_data=None, context=None):
    # Transform prompt through middleware
    middleware_payload = {
        "text": prompt,
        "voice_features": voice_data,
        "temporal_context": context.get("temporal"),
        "spatial_context": context.get("spatial"),
        "social_context": context.get("social"),
        "domain_context": context.get("domain"),
        "transformations": ["voice_conditioned", "4d_projection", "entropy_enhancement"]
    }
    
    response = requests.post("http://prompt-middleware:8002/transform", json=middleware_payload)
    results = response.json()
    
    # Use the most confident transformation
    best_result = max(results["results"], key=lambda x: x["confidence_score"])
    
    return best_result["transformed_prompt"], best_result["subtext_inference"]
```

### 2. Orchestrator Integration

```python
# In orchestrator service
def enhanced_prompt_processing(user_input, voice_context=None):
    # Process through middleware
    enhanced_prompt, subtext = process_prompt_with_middleware(
        user_input, 
        voice_context,
        {
            "temporal": {"time_factor": 0.5, "urgency": 0.3},
            "spatial": {"location": "office", "privacy": 0.7},
            "social": {"relationship": "user", "formality": 0.5},
            "domain": {"keywords": ["coding", "help", "problem"]}
        }
    )
    
    # Use enhanced prompt for LLM
    llm_response = call_llm(enhanced_prompt)
    
    # Apply subtext context to response
    contextualized_response = apply_subtext_context(llm_response, subtext)
    
    return contextualized_response
```

## Performance Optimization

### 1. Caching
```python
# Cache transformation results
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_transform(text, transformation_type, voice_hash=None):
    # Cache based on text + transformation + voice features hash
    pass
```

### 2. Batch Processing
```python
# Process multiple prompts in batch
def batch_transform(prompts, transformations):
    # Group similar prompts for efficient processing
    # Use vectorized operations where possible
    pass
```

### 3. GPU Acceleration
```python
# Use GPU for embedding operations
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## Monitoring and Debugging

### 1. Health Checks
```bash
# Check middleware health
curl http://localhost:8002/health
```

### 2. Performance Metrics
```python
# Monitor transformation performance
import time

start_time = time.time()
result = transform_prompt(prompt, transformations)
processing_time = time.time() - start_time

print(f"Transformation time: {processing_time:.3f}s")
print(f"Confidence score: {result['confidence_score']}")
```

### 3. Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Log transformation details
logger.info(f"Transforming prompt: {prompt}")
logger.info(f"Applied transformations: {transformations}")
logger.info(f"Result confidence: {confidence}")
```

## Best Practices

### 1. Transformation Selection
- **High Confidence**: Use voice-conditioned and semantic rotation
- **Creative Tasks**: Use mirror and chiral transformations
- **Time-Sensitive**: Use 4D projection and entropy enhancement
- **Complex Context**: Use all transformations and ensemble results

### 2. Context Utilization
- **Always provide voice features** when available
- **Include temporal context** for time-sensitive tasks
- **Add social context** for relationship-aware responses
- **Specify domain context** for specialized tasks

### 3. Quality Assurance
- **Monitor confidence scores** - low confidence may indicate poor transformations
- **Validate subtext inference** - check if emotional/intentional analysis makes sense
- **Test with known examples** - verify transformations work as expected
- **A/B test results** - compare transformed vs. original prompts

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Check if voice features are properly extracted
   - Verify context information is complete
   - Try different transformation combinations

2. **Poor Transformations**
   - Ensure text is properly tokenized
   - Check embedding model compatibility
   - Verify transformation parameters

3. **Performance Issues**
   - Enable GPU acceleration
   - Use caching for repeated transformations
   - Optimize batch processing

### Debug Mode
```python
# Enable debug mode
import os
os.environ["DEBUG"] = "true"

# Detailed transformation logging
transformer = AdvancedPromptTransformer(debug=True)
```

## Future Enhancements

1. **Machine Learning Integration**: Train models on transformation effectiveness
2. **Real-time Processing**: Optimize for live voice analysis
3. **Multi-language Support**: Extend to multiple languages
4. **Custom Transformations**: Allow user-defined transformation functions
5. **Federated Learning**: Learn from multiple users while preserving privacy

This middleware suite provides a powerful foundation for advanced prompt processing that goes far beyond traditional statistical methods, enabling deeper understanding and more effective AI interactions.

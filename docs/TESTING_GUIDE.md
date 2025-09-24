# Prompt Middleware Testing Guide

## Overview

This guide covers comprehensive testing of the Prompt Middleware Suite, focusing on the AI's ability to infer the true "object" (intent) from ambiguous prompts, particularly for coding-related tasks.

## Test Structure

### 1. Unit Tests (`test_prompt_middleware.py`)
- **Purpose**: Test individual components and transformations
- **Coverage**: Voice analysis, geometric transformations, semantic reprocessing
- **Focus**: Component functionality and edge cases

### 2. Object Inference Tests (`test_object_inference.py`)
- **Purpose**: Test AI's ability to guess true intent from ambiguous prompts
- **Coverage**: Coding-specific scenarios with voice context
- **Focus**: Accuracy of intent inference

### 3. AI Model Integration Tests (`test_ai_model_integration.py`)
- **Purpose**: Test integration with actual AI models
- **Coverage**: End-to-end workflow from prompt to AI response
- **Focus**: Real-world performance and accuracy

## Running Tests

### Quick Start
```bash
# Run all tests
python scripts/run_prompt_tests.py

# Run specific test types
python scripts/run_prompt_tests.py --type unit
python scripts/run_prompt_tests.py --type object_inference
python scripts/run_prompt_tests.py --type ai_integration

# Run with coverage
python scripts/run_prompt_tests.py --coverage

# Run the demo
python scripts/run_prompt_tests.py --type demo
```

### Individual Test Files
```bash
# Run specific test files
pytest tests/test_prompt_middleware.py -v
pytest tests/test_object_inference.py -v
pytest tests/test_ai_model_integration.py -v

# Run with markers
pytest -m "voice" -v
pytest -m "geometric" -v
pytest -m "ai_integration" -v
```

## Test Scenarios

### Ambiguous Coding Prompts

The tests use carefully crafted ambiguous prompts that hide the true intent:

1. **"This thing is broken"** → `bug_in_code`
   - **Voice Context**: High pitch, negative valence, high arousal
   - **Expected AI Response**: Debugging assistance
   - **Keywords**: debug, fix, investigate, analyze

2. **"Make it faster"** → `performance_optimization`
   - **Voice Context**: Moderate-high pitch, negative valence, moderate arousal
   - **Expected AI Response**: Performance optimization
   - **Keywords**: optimize, profile, refactor, improve

3. **"I don't understand this"** → `code_explanation`
   - **Voice Context**: Low pitch, negative valence, low arousal
   - **Expected AI Response**: Code explanation
   - **Keywords**: explain, document, comment, clarify

4. **"This is messy"** → `code_refactoring`
   - **Voice Context**: Moderate pitch, negative valence, moderate arousal
   - **Expected AI Response**: Refactoring suggestions
   - **Keywords**: refactor, clean, organize, restructure

5. **"Can you look at this?"** → `code_review`
   - **Voice Context**: Low pitch, positive valence, low arousal
   - **Expected AI Response**: Code review
   - **Keywords**: review, check, analyze, validate

6. **"It doesn't work like I thought"** → `logic_error`
   - **Voice Context**: High pitch, negative valence, high arousal
   - **Expected AI Response**: Logic debugging
   - **Keywords**: debug, fix_logic, test, verify

7. **"This takes forever"** → `timeout_issue`
   - **Voice Context**: High pitch, very negative valence, high arousal
   - **Expected AI Response**: Performance/timeout resolution
   - **Keywords**: optimize, profile, timeout, performance

8. **"Something's not right here"** → `suspicious_behavior`
   - **Voice Context**: Moderate pitch, negative valence, moderate arousal
   - **Expected AI Response**: Investigation assistance
   - **Keywords**: investigate, analyze, debug, trace

## Test Categories

### Voice Analysis Tests
- **Prosodic Feature Extraction**: Pitch, rhythm, energy analysis
- **Emotional State Inference**: Anger, joy, sadness, fear, surprise, disgust
- **Subtext Detection**: Intentional ambiguity, urgency, social dominance
- **Cultural Markers**: Regional and social linguistic patterns

### Geometric Transformation Tests
- **Mirror Operations**: Semantic reflection and inversion
- **Chiral Transformations**: Handedness changes in meaning
- **4D Projections**: Temporal and dimensional expansions
- **Topological Mapping**: Continuous deformation of semantic space

### Object Inference Tests
- **Accuracy Measurement**: Percentage of correct intent inferences
- **Confidence Scoring**: Reliability of inference results
- **Multi-Modal Fusion**: Combining voice, text, and context signals
- **Ensemble Methods**: Multiple inference approaches

### AI Integration Tests
- **End-to-End Workflow**: Prompt → Transformation → AI Response
- **Response Analysis**: AI understanding of transformed prompts
- **Performance Metrics**: Response time and throughput
- **Error Handling**: Graceful failure modes

## Performance Benchmarks

### Accuracy Targets
- **Object Inference**: ≥ 70% accuracy
- **Voice Analysis**: ≥ 60% accuracy
- **AI Integration**: ≥ 80% accuracy
- **Ensemble Methods**: ≥ 75% accuracy

### Performance Targets
- **Transformation Time**: < 100ms per prompt
- **Voice Analysis**: < 50ms per audio sample
- **AI Response Time**: < 2s per request
- **Throughput**: > 100 prompts/second

### Memory Usage
- **Peak Memory**: < 2GB for full test suite
- **Memory Growth**: < 100MB per 1000 transformations
- **GPU Memory**: < 4GB for geometric transformations

## Test Data

### Voice Contexts
```python
voice_contexts = {
    "urgent_debug": {
        "pitch_mean": 220.0,
        "pitch_std": 45.0,
        "energy_mean": 0.12,
        "valence": -0.3,
        "arousal": 0.8
    },
    "calm_review": {
        "pitch_mean": 160.0,
        "pitch_std": 15.0,
        "energy_mean": 0.06,
        "valence": 0.1,
        "arousal": 0.3
    },
    "frustrated_optimization": {
        "pitch_mean": 200.0,
        "pitch_std": 35.0,
        "energy_mean": 0.10,
        "valence": -0.5,
        "arousal": 0.7
    }
}
```

### Expected AI Responses
```python
expected_responses = {
    "bug_in_code": "I can help you debug this issue. Let me analyze the code to identify the problem.",
    "performance_optimization": "I'll help you optimize the performance. Let me profile the code and suggest improvements.",
    "code_explanation": "I'll explain this code for you. Let me break it down step by step.",
    "code_refactoring": "I can help you refactor this code. Let me suggest ways to improve the structure.",
    "code_review": "I'll review this code for you. Let me check for potential issues and best practices.",
    "logic_error": "I can help you fix the logic error. Let me analyze the algorithm and identify the issue.",
    "timeout_issue": "I'll help you resolve the timeout issue. Let me analyze the performance bottlenecks.",
    "suspicious_behavior": "I can help you investigate this suspicious behavior. Let me trace through the code."
}
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Prompt Middleware Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: python scripts/run_prompt_tests.py --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: prompt-tests
        name: Run prompt middleware tests
        entry: python scripts/run_prompt_tests.py --type unit
        language: system
        pass_filenames: false
        always_run: true
```

## Debugging Tests

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure services directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/services"
   ```

2. **Missing Dependencies**
   ```bash
   # Check dependencies
   python scripts/run_prompt_tests.py --check-deps
   
   # Install missing packages
   pip install pytest numpy scipy torch transformers librosa networkx matplotlib
   ```

3. **GPU Memory Issues**
   ```bash
   # Run without GPU tests
   pytest -m "not geometric" -v
   ```

4. **Voice Analysis Failures**
   ```bash
   # Run without voice tests
   pytest -m "not voice" -v
   ```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run specific test with debug output
pytest tests/test_object_inference.py::TestObjectInference::test_object_inference_accuracy -v -s
```

## Test Results Interpretation

### Success Criteria
- **Green Tests**: All assertions pass
- **Accuracy**: Meets minimum thresholds
- **Performance**: Within time limits
- **Coverage**: > 80% code coverage

### Failure Analysis
- **Red Tests**: Identify specific failures
- **Accuracy Below Threshold**: Review inference logic
- **Performance Issues**: Profile bottlenecks
- **Coverage Gaps**: Add missing tests

### Reporting
```bash
# Generate detailed report
pytest --html=report.html --self-contained-html

# Generate coverage report
pytest --cov=services/prompt_middleware --cov-report=html

# Generate performance report
pytest --durations=10 -v
```

## Best Practices

### Writing Tests
1. **Clear Test Names**: Describe what is being tested
2. **Isolated Tests**: Each test should be independent
3. **Realistic Data**: Use representative test cases
4. **Edge Cases**: Test boundary conditions
5. **Error Handling**: Test failure modes

### Maintaining Tests
1. **Regular Updates**: Keep tests current with code changes
2. **Performance Monitoring**: Track test execution time
3. **Coverage Tracking**: Maintain high coverage
4. **Documentation**: Keep test documentation current

### Test Data Management
1. **Version Control**: Track test data changes
2. **Data Privacy**: Ensure no sensitive data in tests
3. **Data Size**: Keep test datasets manageable
4. **Data Quality**: Validate test data accuracy

This testing framework ensures the Prompt Middleware Suite can reliably infer true intent from ambiguous prompts, providing a solid foundation for advanced AI interactions.
